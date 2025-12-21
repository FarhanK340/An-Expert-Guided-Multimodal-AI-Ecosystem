"""
Case management views.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import Case
from .serializers import CaseSerializer


class CaseListCreateView(APIView):
    """List all cases or create new case."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get all cases (all users can see all cases for now)."""
        # Show all cases to all users
        cases = Case.objects.all().order_by('-created_at')
        
        serializer = CaseSerializer(cases, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        """Create new case."""
        serializer = CaseSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CaseDetailView(APIView):
    """Get case details."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, case_id):
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            serializer = CaseSerializer(case)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class CaseUpdateView(APIView):
    """Update case."""
    permission_classes = [IsAuthenticated]
    
    def patch(self, request, case_id):
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            serializer = CaseSerializer(case, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class CaseDeleteView(APIView):
    """Delete case."""
    permission_classes = [IsAuthenticated]
    
    def delete(self, request, case_id):
        try:
            case = Case.objects.get(case_id=case_id)
            
            # Check permission
            if not request.user.is_admin and case.created_by != request.user:
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            case.delete()
            return Response(
                {'message': 'Case deleted successfully'},
                status=status.HTTP_200_OK
            )
        except Case.DoesNotExist:
            return Response(
                {'error': 'Case not found'},
                status=status.HTTP_404_NOT_FOUND
            )