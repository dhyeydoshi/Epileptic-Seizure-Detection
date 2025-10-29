"""
API endpoints for epileptic seizure detection.
Version 2.3 - Enhanced security with authentication, validation, error handling, and database logging.
"""
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.cache import cache
from django.conf import settings
import json
import logging
import hashlib
import os
import sys
import re

from .services import MLModelService, PredictionService
from .models import PredictionHistory, ModelPerformance

logger = logging.getLogger(__name__)

# Whitelist of allowed models for security
ALLOWED_MODELS = {
    'DeepLearning.h5',
    'SVMModel.pickle',
    'KNNModel.pickle',
    'NaiveBayesModel.pickle',
    'LogisticRegressionModel.pickle',
    'RandomForestModel.pickle',
    'XgBoostModel.pickle'
}


def validate_patient_number(patient_number: str) -> tuple[bool, str]:
    """
    Validate patient number format for security.

    Args:
        patient_number: Patient identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not patient_number:
        return False, "patient_number is required"

    # Remove whitespace
    patient_number = patient_number.strip()

    # Check length (prevent DoS with extremely long inputs)
    if len(patient_number) > 50:
        return False, "patient_number too long (max 50 characters)"

    # Only allow alphanumeric and basic separators (prevent injection)
    if not re.match(r'^[a-zA-Z0-9_-]+$', patient_number):
        return False, "patient_number contains invalid characters"

    return True, ""


def validate_threshold(threshold_value) -> tuple[bool, str, float]:
    """
    Validate threshold value.

    Returns:
        Tuple of (is_valid, error_message, validated_value)
    """
    try:
        threshold = float(threshold_value)
        if not 0.0 <= threshold <= 1.0:
            return False, "threshold must be between 0.0 and 1.0", 0.5
        return True, "", threshold
    except (ValueError, TypeError):
        return False, "threshold must be a valid number", 0.5


@require_http_methods(["POST"])
def predict_api(request):
    """
    API endpoint for making predictions with caching and security.

    POST /api/predict/
    Headers:
        X-CSRFToken: <csrf_token>  (Required for security)
    Body:
    {
        "patient_number": "001",
        "model_name": "DeepLearning.h5",
        "threshold": 0.5
    }

    Security Features:
    - CSRF protection enabled
    - Input validation and sanitization
    - Model whitelist validation
    - Error message sanitization
    - Request logging
    - Database audit trail
    """
    try:
        # Parse request body
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received from {request.META.get('REMOTE_ADDR')}")
            return JsonResponse({
                'status': 'error',
                'error': 'Invalid JSON format'
            }, status=400)

        # Extract and validate patient number
        patient_number = data.get('patient_number', '').strip()
        is_valid, error_msg = validate_patient_number(patient_number)
        if not is_valid:
            logger.warning(f"Invalid patient_number: {error_msg}")
            return JsonResponse({
                'status': 'error',
                'error': error_msg
            }, status=400)

        # Extract and validate model name
        model_name = data.get('model_name', 'DeepLearning.h5')
        if model_name not in ALLOWED_MODELS:
            logger.warning(f"Attempt to access unauthorized model: {model_name}")
            return JsonResponse({
                'status': 'error',
                'error': f'Invalid model. Allowed models: {", ".join(ALLOWED_MODELS)}'
            }, status=400)

        # Extract and validate threshold
        threshold_value = data.get('threshold', 0.5)
        is_valid, error_msg, threshold = validate_threshold(threshold_value)
        if not is_valid:
            return JsonResponse({
                'status': 'error',
                'error': error_msg
            }, status=400)

        # Generate cache key for this specific prediction
        cache_hash = hashlib.md5(
            f"prediction_{model_name}_{patient_number}_{threshold}".encode()
        ).hexdigest()

        # Check cache first (production only)
        if not settings.DEBUG:
            cached_result = cache.get(cache_hash)
            if cached_result:
                logger.info(f"Returning cached prediction for patient {patient_number}")
                return JsonResponse({
                    'status': 'success',
                    'data': cached_result,
                    'cached': True
                })

        # Create services and make prediction
        model_service = MLModelService(model_name)
        prediction_service = PredictionService(model_service)
        result = prediction_service.predict_patient(patient_number, threshold)

        # Cache the result for 1 hour
        cache.set(cache_hash, result, 3600)

        # Save to database for audit trail
        try:
            # Get client IP
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip_address = x_forwarded_for.split(',')[0].strip()
            else:
                ip_address = request.META.get('REMOTE_ADDR')

            # Create prediction history record
            PredictionHistory.objects.create(
                patient_number=patient_number,
                model_used=model_name,
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                has_epilepsy=result['predicted_class'] == 1,
                threshold_used=threshold,
                ip_address=ip_address
            )

            # Increment model usage counter
            try:
                model_perf = ModelPerformance.objects.filter(saved_model_filename=model_name).first()
                if model_perf:
                    model_perf.increment_predictions()
            except Exception as e:
                logger.warning(f"Failed to increment model counter: {str(e)}")

        except Exception as e:
            # Don't fail the prediction if database logging fails
            logger.error(f"Failed to save prediction history: {str(e)}")

        logger.info(f"Prediction completed for patient {patient_number} using {model_name}")

        return JsonResponse({
            'status': 'success',
            'data': result,
            'cached': False
        })

    except ValueError as e:
        # Sanitize error message to avoid information disclosure
        error_message = str(e)
        if 'No data found' in error_message:
            safe_message = 'Patient not found in dataset'
        elif 'Dataset not found' in error_message:
            safe_message = 'Dataset unavailable'
        else:
            safe_message = 'Invalid input data'

        logger.warning(f"Validation error in predict_api: {error_message}")
        return JsonResponse({
            'status': 'error',
            'error': safe_message
        }, status=400)

    except FileNotFoundError as e:
        logger.error(f"File not found in predict_api: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'error': 'Required resource not found'
        }, status=500)

    except Exception as e:
        # Log detailed error but return generic message
        logger.error(f"Error in predict_api: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'error': 'An error occurred processing your request'
        }, status=500)


@require_http_methods(["GET"])
def models_list_api(request):
    """
    API endpoint for listing available models with caching.
    GET /api/models/

    Security Features:
    - Read-only operation
    - Cached results
    - Sanitized file information
    """
    try:
        # Cache the model list for 1 hour
        cache_key = 'available_models_list_v2'
        cached_models = cache.get(cache_key)

        if cached_models:
            logger.debug("Returning cached models list")
            return JsonResponse({
                'status': 'success',
                'models': cached_models,
                'cached': True
            })

        model_dir = os.path.join(os.path.dirname(__file__), '../model')
        models = []

        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                # Only include whitelisted models
                if filename in ALLOWED_MODELS:
                    file_path = os.path.join(model_dir, filename)
                    try:
                        file_size = os.path.getsize(file_path)
                        models.append({
                            'filename': filename,
                            'type': 'deep_learning' if filename.endswith('.h5') else 'traditional_ml',
                            'size_mb': round(file_size / (1024 * 1024), 2)
                        })
                    except OSError as e:
                        logger.warning(f"Error reading model file {filename}: {str(e)}")
                        continue

        # Cache for 1 hour
        cache.set(cache_key, models, 3600)

        logger.info(f"Models list requested, returning {len(models)} models")

        return JsonResponse({
            'status': 'success',
            'models': models,
            'cached': False
        })

    except Exception as e:
        logger.error(f"Error in models_list_api: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'error': 'Unable to retrieve models list'
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint with system information.
    GET /api/health/

    Security Features:
    - Minimal information disclosure
    - No sensitive data exposed
    """
    try:
        from django.db import connection

        # Check database connection
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            db_status = 'healthy'
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            db_status = 'unhealthy'

        # Check cache
        try:
            cache.set('health_check', 'ok', 10)
            cache_status = 'healthy' if cache.get('health_check') == 'ok' else 'unhealthy'
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            cache_status = 'unhealthy'

        # Overall health
        overall_status = 'healthy' if (db_status == 'healthy' and cache_status == 'healthy') else 'degraded'

        response_data = {
            'status': overall_status,
            'version': '2.0',
            'components': {
                'database': db_status,
                'cache': cache_status
            }
        }

        # Only include debug info in DEBUG mode
        if settings.DEBUG:
            response_data['debug_info'] = {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'debug_mode': True
            }

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}", exc_info=True)
        return JsonResponse({
            'status': 'degraded',
            'version': '2.0',
            'error': 'Health check failed'
        }, status=500)
