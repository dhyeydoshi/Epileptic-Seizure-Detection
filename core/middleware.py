"""
Custom middleware for epileptic seizure detection application.
Version 2.0 - Enhanced security and performance middleware.
"""
import time
import logging
from django.utils.deprecation import MiddlewareMixin
from django.core.cache import cache
from django.http import JsonResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(MiddlewareMixin):
    """Log all requests with timing information."""

    def process_request(self, request):
        """Mark the start time of the request."""
        request._start_time = time.time()
        return None

    def process_response(self, request, response):
        """Log request completion with duration."""
        if hasattr(request, '_start_time'):
            duration = time.time() - request._start_time

            # Log level based on duration (warn if slow)
            if duration > 5.0:
                logger.warning(
                    f"SLOW REQUEST: {request.method} {request.path} - "
                    f"Status: {response.status_code} - "
                    f"Duration: {duration:.2f}s"
                )
            else:
                logger.info(
                    f"{request.method} {request.path} - "
                    f"Status: {response.status_code} - "
                    f"Duration: {duration:.2f}s"
                )
        return response


class RateLimitMiddleware(MiddlewareMixin):
    """Enhanced rate limiting middleware with different limits for API and web."""

    # Rate limits: (max_requests, time_window_seconds)
    RATE_LIMITS = {
        'api': (100, 3600),      # 100 requests per hour for API
        'web': (1000, 3600),     # 1000 requests per hour for web
        'health': (1000, 60),    # 1000 requests per minute for health checks
    }

    def process_request(self, request):
        """Check rate limit for the client with enhanced logic."""
        # Determine request type
        path = request.path

        if path.startswith('/api/health'):
            limit_type = 'health'
        elif path.startswith('/api/'):
            limit_type = 'api'
        elif path.startswith('/admin/'):
            # Don't rate limit admin (has its own auth)
            return None
        elif path.startswith('/static/') or path.startswith('/media/'):
            # Don't rate limit static files
            return None
        else:
            limit_type = 'web'

        # Get client IP
        ip = self.get_client_ip(request)

        # Get rate limit for this type
        max_requests, time_window = self.RATE_LIMITS[limit_type]

        # Create cache key
        cache_key = f'rate_limit_{limit_type}_{ip}'

        # Get current request count
        requests = cache.get(cache_key, 0)

        if requests >= max_requests:
            logger.warning(
                f"Rate limit exceeded for {ip} on {limit_type} "
                f"({requests}/{max_requests} requests)"
            )

            # Return appropriate response based on request type
            if path.startswith('/api/'):
                return JsonResponse({
                    'status': 'error',
                    'error': 'Rate limit exceeded. Please try again later.',
                    'retry_after': time_window
                }, status=429)
            else:
                from django.http import HttpResponse
                return HttpResponse(
                    'Rate limit exceeded. Please try again later.',
                    status=429
                )

        # Increment counter
        cache.set(cache_key, requests + 1, time_window)

        # Add rate limit info to request for debugging
        request.rate_limit_info = {
            'requests': requests + 1,
            'limit': max_requests,
            'window': time_window
        }

        return None

    @staticmethod
    def get_client_ip(request):
        """Get the client's IP address, accounting for proxies."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            # Take the first IP in the chain (client IP)
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', 'unknown')
        return ip


class SecurityHeadersMiddleware(MiddlewareMixin):
    """Add comprehensive security headers to all responses."""

    def process_response(self, request, response):
        """Add security headers."""
        # Prevent MIME type sniffing
        response['X-Content-Type-Options'] = 'nosniff'

        # Prevent clickjacking
        response['X-Frame-Options'] = 'DENY'

        # XSS Protection (legacy browsers)
        response['X-XSS-Protection'] = '1; mode=block'

        # Referrer policy
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Content Security Policy (CSP)
        # Adjust based on your actual needs
        if not request.path.startswith('/admin/'):
            response['Content-Security-Policy'] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self';"
            )

        # Permissions Policy (formerly Feature Policy)
        response['Permissions-Policy'] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )

        return response

