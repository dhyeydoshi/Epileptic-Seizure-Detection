"""
Custom decorators for epileptic seizure detection.
Version 2.0 - View decorators for common functionality.
"""
from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)


def dataset_required(view_func):
    """
    Decorator to ensure dataset is loaded before accessing a view.
    Usage: @dataset_required
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('dataset_loaded', False):
            messages.warning(request, 'Please upload a dataset first.')
            logger.warning(f"Dataset not loaded for {request.path}")
            return redirect('core:dashboard')
        return view_func(request, *args, **kwargs)
    return wrapper


def log_view_access(view_func):
    """
    Decorator to log view access.
    Usage: @log_view_access
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        logger.info(f"View accessed: {view_func.__name__} by {request.META.get('REMOTE_ADDR')}")
        return view_func(request, *args, **kwargs)
    return wrapper


def cache_response(timeout=300):
    """
    Decorator to cache view responses.
    Usage: @cache_response(timeout=600)
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            # Generate cache key from view name and args
            cache_key = f"view_{view_func.__name__}_{str(args)}_{str(kwargs)}"

            # Try to get from cache
            response = cache.get(cache_key)
            if response is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return response

            # Generate and cache response
            response = view_func(request, *args, **kwargs)
            cache.set(cache_key, response, timeout)
            logger.debug(f"Cached response for {cache_key}")

            return response
        return wrapper
    return decorator
