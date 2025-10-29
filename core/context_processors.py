"""
Custom context processors for epileptic seizure detection.
Version 2.0 - Application-wide context variables.
"""
from django.conf import settings
import os


def app_version(request):
    """Add application version to context."""
    return {
        'APP_VERSION': '2.0',
        'APP_NAME': 'Epileptic Seizure Detection'
    }


def models_info(request):
    """Add information about available models to context."""
    model_dir = os.path.join(settings.BASE_DIR, 'model')

    if not os.path.exists(model_dir):
        return {'MODELS_COUNT': 0, 'MODELS_AVAILABLE': False}

    models_count = len([
        f for f in os.listdir(model_dir)
        if f.endswith(('.pickle', '.h5'))
    ])

    return {
        'MODELS_COUNT': models_count,
        'MODELS_AVAILABLE': models_count > 0
    }
