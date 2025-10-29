"""
Management command to initialize the application.
Version 2.0 - Setup script for first-time deployment.
"""
from django.core.management.base import BaseCommand
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Initialize the epileptic seizure detection application'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Initializing Application...'))

        # Create logs directory
        logs_dir = settings.BASE_DIR / 'logs'
        if not logs_dir.exists():
            logs_dir.mkdir(parents=True)
            self.stdout.write(self.style.SUCCESS(f'Created logs directory: {logs_dir}'))

        # Create media directory
        media_dir = settings.BASE_DIR / 'media'
        if not media_dir.exists():
            media_dir.mkdir(parents=True)
            self.stdout.write(self.style.SUCCESS(f'Created media directory: {media_dir}'))

        # Verify model directory
        model_dir = settings.BASE_DIR / 'model'
        if model_dir.exists():
            models = list(model_dir.glob('*.pickle')) + list(model_dir.glob('*.h5'))
            self.stdout.write(self.style.SUCCESS(f'Found {len(models)} models in model directory'))
        else:
            self.stdout.write(self.style.WARNING('Model directory not found!'))

        # Verify data directory
        data_dir = settings.BASE_DIR / 'data'
        if data_dir.exists():
            csv_files = list(data_dir.glob('*.csv'))
            self.stdout.write(self.style.SUCCESS(f'Found {len(csv_files)} CSV files in data directory'))
        else:
            self.stdout.write(self.style.WARNING('Data directory not found!'))

        self.stdout.write(self.style.SUCCESS('Initialization complete!'))

