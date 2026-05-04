"""
Celery tasks for asynchronous ML inference.

This module exposes @shared_task wrappers around the synchronous
InferenceEngine so that inference runs in a Celery worker process
instead of blocking an HTTP request thread.
"""

from pathlib import Path
import logging

from celery import shared_task

from cases.models import Case

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name="inference.tasks.run_segmentation",
    max_retries=0,
    time_limit=600,        # hard kill after 10 min
    soft_time_limit=540,   # raises SoftTimeLimitExceeded at 9 min
)
def run_segmentation_task(self, case_id: str, case_dir_str: str) -> dict:
    """
    Run MoME+ segmentation for the given case asynchronously.

    Args:
        case_id:      UUID string of the Case record.
        case_dir_str: Absolute path to the directory containing the
                      uploaded NIfTI files (as a string, JSON-serialisable).

    Returns:
        dict with volumes, confidence_scores, gating_weights, mask_files.
    """
    from .inference_utils import InferenceEngine

    logger.info(f"[Celery] Starting segmentation task for case {case_id}")

    try:
        case = Case.objects.get(case_id=case_id)
    except Case.DoesNotExist:
        logger.error(f"[Celery] Case {case_id} not found.")
        raise

    try:
        engine = InferenceEngine()
        result = engine.run_inference(case_id, Path(case_dir_str))
        logger.info(f"[Celery] Segmentation complete for case {case_id}")
        return result
    except Exception as exc:
        logger.exception(f"[Celery] Segmentation failed for case {case_id}: {exc}")
        case.status = "failed"
        case.error_message = str(exc)
        case.save(update_fields=["status", "error_message"])
        raise
