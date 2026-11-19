"""Django settings for the distracted driver detection web app."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # webapp/
PROJECT_DIR = BASE_DIR.parent                           # repository root

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "detector",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "driver_site.urls"
WSGI_APPLICATION = "driver_site.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    },
]

# The app has no user accounts or stored records, so no database is needed.
DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
# Expose the repository's figures (images/) to the templates
STATICFILES_DIRS = [("figures", PROJECT_DIR / "images")]

DATA_UPLOAD_MAX_MEMORY_SIZE = 6 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 6 * 1024 * 1024

# Machine learning: the published TensorFlow.js model and the files written by scripts/
MODEL_DIR = PROJECT_DIR / "models" / "tfjs"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
EVALUATION_PATH = PROJECT_DIR / "models" / "evaluation.json"    # optional, from scripts/evaluate.py
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "original_bangkit_capstone.ipynb"
SAMPLES_DIR = BASE_DIR / "detector" / "static" / "detector" / "samples"
