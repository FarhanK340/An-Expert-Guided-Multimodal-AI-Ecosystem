"""
Report generation and management views.
Implements the full pipeline: structured_findings → JSON descriptor → LLM → Report.
"""

import json
import io
from datetime import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from cases.models import Case, SegmentationResult
from .models import Report, ReportEdit
from .serializers import ReportSerializer, ReportUpdateSerializer
from .llm_utils import generate_json_descriptor, generate_report_from_descriptor


class GenerateReportView(APIView):
    """
    POST /api/reports/generate/<case_id>/
    Reads SegmentationResult.structured_findings, builds a JSON descriptor,
    calls LLM, and creates/overwrites the Report for this case.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, case_id):
        # --- Fetch case ---
        try:
            case = Case.objects.get(case_id=case_id)
        except Case.DoesNotExist:
            return Response({'error': 'Case not found'}, status=status.HTTP_404_NOT_FOUND)

        if case.created_by != request.user and not request.user.is_staff:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        # --- Fetch segmentation result ---
        try:
            seg_result = SegmentationResult.objects.get(case=case)
        except SegmentationResult.DoesNotExist:
            return Response(
                {'error': 'No segmentation result found. Please run inference first.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- Generate JSON descriptor ---
        patient_info = {
            'patient_id': case.patient_id,
            'age': case.age,
            'sex': case.sex,
            'clinical_history': case.clinical_history,
            'indication': case.indication,
            'scan_date': str(case.scan_date) if case.scan_date else None,
        }

        json_descriptor = generate_json_descriptor(
            structured_findings=seg_result.structured_findings,
            patient_info=patient_info,
            case_id=str(case.case_id)
        )

        # --- Generate report text via LLM ---
        report_text = generate_report_from_descriptor(json_descriptor)

        # --- Persist report ---
        report, created = Report.objects.update_or_create(
            case=case,
            defaults={
                'ai_generated_text': report_text,
                'finalized_text': report_text,
                'findings_json': json_descriptor,
                'status': 'draft',
                'last_edited_by': request.user,
            }
        )

        # Update case status if not already completed
        if case.status != 'completed':
            case.status = 'completed'
            case.save()

        serializer = ReportSerializer(report)
        return Response({
            'message': 'Report generated successfully',
            'report': serializer.data,
            'created': created,
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class ReportListView(APIView):
    """
    GET /api/reports/
    Returns reports belonging to the authenticated user's cases.
    Admin sees all reports.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role == 'admin' or request.user.is_staff:
            reports = Report.objects.all().order_by('-generated_at')
        elif request.user.role == 'patient':
            # Show finalized/reviewed reports for cases linked to this patient
            reports = Report.objects.filter(
                case__patient_user=request.user
            ).exclude(status='draft').order_by('-generated_at')
        else:
            # Show reports for cases created by this user
            reports = Report.objects.filter(
                case__created_by=request.user
            ).order_by('-generated_at')

        serializer = ReportSerializer(reports, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ReportDetailView(APIView):
    """GET /api/reports/<report_id>/"""
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        try:
            report = Report.objects.get(report_id=report_id)
        except Report.DoesNotExist:
            return Response({'error': 'Report not found'}, status=status.HTTP_404_NOT_FOUND)

        if request.user.is_staff or request.user.role == 'admin':
            pass
        elif request.user.role == 'patient':
            if report.case.patient_user != request.user or report.status == 'draft':
                return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)
        else:
            if report.case.created_by != request.user:
                return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        serializer = ReportSerializer(report)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ReportUpdateView(APIView):
    """
    PATCH /api/reports/<report_id>/update/
    Clinician edits the finalized text. Saves a ReportEdit audit record.
    """
    permission_classes = [IsAuthenticated]

    def patch(self, request, report_id):
        try:
            report = Report.objects.get(report_id=report_id)
        except Report.DoesNotExist:
            return Response({'error': 'Report not found'}, status=status.HTTP_404_NOT_FOUND)

        if report.case.created_by != request.user and not request.user.is_staff:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        serializer = ReportUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        new_text = serializer.validated_data['finalizedText']
        edit_reason = serializer.validated_data.get('editReason', '')

        # Save audit trail
        ReportEdit.objects.create(
            report=report,
            edited_by=request.user,
            original_text=report.finalized_text,
            edited_text=new_text,
            section='full_report',
            edit_reason=edit_reason,
            character_change_count=abs(len(new_text) - len(report.finalized_text)),
        )

        report.finalized_text = new_text
        report.edit_count += 1
        report.status = 'reviewed'
        report.reviewed_at = datetime.now()
        report.last_edited_by = request.user
        report.save()

        return Response({
            'message': 'Report updated successfully',
            'report': ReportSerializer(report).data,
        }, status=status.HTTP_200_OK)


class ExportPDFView(APIView):
    """
    POST /api/reports/<report_id>/export/
    Generates a PDF of the report using reportlab and returns it as a download.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, report_id):
        try:
            report = Report.objects.get(report_id=report_id)
        except Report.DoesNotExist:
            return Response({'error': 'Report not found'}, status=status.HTTP_404_NOT_FOUND)

        if request.user.is_staff or request.user.role == 'admin':
            pass
        elif request.user.role == 'patient':
            if report.case.patient_user != request.user or report.status == 'draft':
                return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)
        else:
            if report.case.created_by != request.user:
                return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        try:
            pdf_bytes = self._generate_pdf(report)
        except Exception as e:
            return Response({'error': f'PDF generation failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        from django.http import HttpResponse
        response = HttpResponse(pdf_bytes, content_type='application/pdf')
        filename = f"report_{report.case.patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    def _generate_pdf(self, report: Report) -> bytes:
        """Generate PDF bytes using reportlab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            raise ImportError("reportlab is not installed. Run: pip install reportlab")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'Title', parent=styles['Heading1'],
            fontSize=18, textColor=colors.HexColor('#1a1a2e'),
            alignment=TA_CENTER, spaceAfter=6
        )
        story.append(Paragraph("Brain Tumor Analysis Report", title_style))
        story.append(Paragraph("DRAFT - FOR RESEARCH PURPOSES ONLY", styles['Normal']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.4*cm))

        # Patient & Case info
        case = report.case
        findings = report.findings_json or {}
        patient_info = findings.get('patient_info', {})

        info_data = [
            ['Patient ID', case.patient_id or 'N/A'],
            ['Age', str(patient_info.get('age', case.age) or 'N/A')],
            ['Sex', str(patient_info.get('sex', case.sex) or 'N/A')],
            ['Scan Date', str(case.scan_date or 'N/A')],
            ['Report Date', datetime.now().strftime('%Y-%m-%d')],
            ['Report Status', report.status.capitalize()],
        ]
        info_table = Table(info_data, colWidths=[5*cm, 12*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4ff')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a1a2e')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8faff')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.6*cm))

        # Tumor Metrics table
        metrics = findings.get('tumor_metrics', {})
        if metrics:
            story.append(Paragraph("Segmentation Metrics", styles['Heading2']))
            metrics_data = [['Region', 'Volume (mm³)', 'Confidence']]
            for region_key, label in [('whole_tumor', 'Whole Tumor'), ('tumor_core', 'Tumor Core'), ('enhancing_tumor', 'Enhancing Tumor')]:
                vol = metrics.get('volumes', {}).get(region_key, 'N/A')
                conf = metrics.get('confidence_scores', {}).get(region_key, 'N/A')
                metrics_data.append([label,
                                      f"{vol:.1f}" if isinstance(vol, float) else str(vol),
                                      f"{conf:.2%}" if isinstance(conf, float) else str(conf)])
            m_table = Table(metrics_data, colWidths=[6*cm, 5.5*cm, 5.5*cm])
            m_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (1, 0), (-1, -1), [colors.white, colors.HexColor('#f8faff')]),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(m_table)
            story.append(Spacer(1, 0.6*cm))

        # Report narrative
        story.append(Paragraph("Radiological Report", styles['Heading2']))
        report_text = report.finalized_text or report.ai_generated_text or ''
        for paragraph in report_text.split('\n\n'):
            paragraph = paragraph.strip()
            if paragraph:
                if paragraph.startswith('**') and paragraph.endswith('**'):
                    story.append(Paragraph(paragraph.strip('*'), styles['Heading3']))
                else:
                    story.append(Paragraph(paragraph.replace('\n', '<br/>'), styles['Normal']))
                story.append(Spacer(1, 0.2*cm))

        doc.build(story)
        return buffer.getvalue()


class PDFListView(APIView):
    """GET /api/reports/<report_id>/pdfs/"""
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        return Response({'message': 'PDF versions are generated on demand via /export/'})


class TraceabilityView(APIView):
    """GET /api/reports/<report_id>/traceability/"""
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        try:
            report = Report.objects.get(report_id=report_id)
        except Report.DoesNotExist:
            return Response({'error': 'Report not found'}, status=status.HTTP_404_NOT_FOUND)

        links = list(report.traceability_links.values(
            'sentence_text', 'sentence_index', 'section', 'evidence_path', 'evidence_value', 'confidence_score'
        ))
        return Response({'traceability_links': links}, status=status.HTTP_200_OK)


class TemplateListView(APIView):
    """GET /api/reports/templates/"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from .models import ReportTemplate
        templates = ReportTemplate.objects.filter(is_active=True).values(
            'name', 'template_type', 'version', 'is_default'
        )
        return Response(list(templates), status=status.HTTP_200_OK)


class TemplateDetailView(APIView):
    """GET /api/reports/templates/<pk>/"""
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        from .models import ReportTemplate
        try:
            t = ReportTemplate.objects.get(pk=pk)
            return Response({
                'name': t.name, 'template_type': t.template_type,
                'version': t.version, 'template_text': t.template_text,
            })
        except ReportTemplate.DoesNotExist:
            return Response({'error': 'Template not found'}, status=status.HTTP_404_NOT_FOUND)
