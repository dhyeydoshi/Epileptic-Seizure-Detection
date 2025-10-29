"""
Django models for storing predictions and user data.
Version 2.3 - Activated and integrated with the application.
"""
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class PredictionHistory(models.Model):
    """Store prediction history for auditing and analysis."""

    patient_number = models.CharField(max_length=50, db_index=True)
    model_used = models.CharField(max_length=100)
    predicted_class = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]  # Fixed: Should be 0.0-1.0, not 0-100
    )
    has_epilepsy = models.BooleanField()
    threshold_used = models.FloatField(default=0.5)  # Added: Track threshold
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)  # Added: Track source

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Prediction Histories"
        indexes = [
            models.Index(fields=['patient_number', 'created_at']),
            models.Index(fields=['model_used', 'created_at']),
        ]

    def __str__(self):
        return f"Patient {self.patient_number} - {self.model_used} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    @property
    def confidence_percentage(self):
        """Return confidence as percentage."""
        return self.confidence * 100


class DatasetUpload(models.Model):
    """Track dataset uploads."""

    data_file_name = models.CharField(max_length=255)
    labels_file_name = models.CharField(max_length=255)
    upload_date = models.DateTimeField(default=timezone.now, db_index=True)
    total_samples = models.IntegerField(null=True, blank=True)
    uploaded_by_ip = models.GenericIPAddressField(null=True, blank=True)  # Added: Track uploader

    class Meta:
        ordering = ['-upload_date']
        verbose_name = "Dataset Upload"
        verbose_name_plural = "Dataset Uploads"

    def __str__(self):
        return f"Dataset: {self.data_file_name} ({self.upload_date.strftime('%Y-%m-%d')})"


class ModelPerformance(models.Model):
    """Store model performance metrics."""

    model_name = models.CharField(max_length=100, unique=True)
    saved_model_filename = models.CharField(max_length=255, default='')  # Added: Link to actual file
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField()
    total_predictions = models.IntegerField(default=0)  # Added: Usage tracking
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-accuracy']
        verbose_name = "Model Performance"
        verbose_name_plural = "Model Performances"

    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.2f}%"

    def increment_predictions(self):
        """Increment prediction counter."""
        self.total_predictions += 1
        self.save(update_fields=['total_predictions'])
