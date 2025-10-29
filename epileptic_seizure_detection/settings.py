"""
Django settings for epileptic_seizure_detection project.
Version 2.0 - Updated for Python 3.12 and Django 5.2 compatibility.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.2/ref/settings/
"""

from pathlib import Path
import logging
import os
import platform
import secrets

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/
# SECURITY WARNING: keep the secret key used in production secret!
check_file = os.path.isfile(os.path.join(BASE_DIR, 'secret_key.txt'))
if check_file:
    with open(os.path.join(BASE_DIR, 'secret_key.txt'), 'r') as f:
        SECRET_KEY = f.read().strip()
        # Validate secret key length
        if len(SECRET_KEY) < 50 or SECRET_KEY.startswith('django-insecure-'):
            raise ValueError("SECRET_KEY must be at least 50 characters and not start with 'django-insecure-'")
else:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        # Generate a strong secret key for development only
        if os.environ.get('DEBUG', 'False') == 'True':
            SECRET_KEY = secrets.token_urlsafe(50)
            print("WARNING: Using auto-generated SECRET_KEY. Set a permanent one in production!")
        else:
            raise ValueError("SECRET_KEY environment variable must be set in production")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = platform.node().lower() in os.environ.get('DEBUG', 'False') == 'True'

ALLOWED_HOSTS = ['epileptic-seizure-detection.onrender.com', '127.0.0.1', 'localhost']
if DEBUG:
    ALLOWED_HOSTS.append('*')

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'django.forms',

    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # Custom middleware for enhanced security and performance (Version 2.0)
    'core.middleware.RequestLoggingMiddleware',
    'core.middleware.RateLimitMiddleware',
    'core.middleware.SecurityHeadersMiddleware',

]

ROOT_URLCONF = 'epileptic_seizure_detection.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'epileptic_seizure_detection.wsgi.application'

# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'epileptic-seizure-cache',
        'TIMEOUT': 3600,  # 1 hour
        'OPTIONS': {
            'MAX_ENTRIES': 1000
        }
    }
}

# Session configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.cached_db'
SESSION_CACHE_ALIAS = 'default'
SESSION_COOKIE_AGE = 86400  # 24 hours

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs/app.log',
            'maxBytes': 1024 * 1024 * 10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'core': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}

# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'static_root/'

STATICFILES_DIRS = [
    BASE_DIR / "static/",
]

# Django 5.2+ uses STORAGES instead of deprecated STATICFILES_STORAGE
# This is the new way to configure static and media file storage
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Security settings
CSRF_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
CSRF_COOKIE_SAMESITE = 'Strict'
SESSION_COOKIE_SAMESITE = 'Strict'


# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB

# HTTPS/SSL Settings (fixes security.W008)
SECURE_SSL_REDIRECT = not DEBUG  # Redirect all HTTP to HTTPS in production

# For reverse proxy/load balancer support
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# HSTS - HTTP Strict Transport Security (fixes security.W004)
# Forces browsers to use HTTPS for all future requests
if not DEBUG:
    SECURE_HSTS_SECONDS = 31536000  # 1 year (recommended: 31536000)
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True  # Apply to all subdomains
    SECURE_HSTS_PRELOAD = True  # Allow preloading in browsers
