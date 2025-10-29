"""
Django models for storing predictions and user data.
Version 2.0 - Added database models for tracking.
"""
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class PredictionHistory(models.Model):
    """Store prediction history for auditing and analysis."""

    patient_number = models.CharField(max_length=50)
    model_used = models.CharField(max_length=100)
    predicted_class = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    has_epilepsy = models.BooleanField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Prediction Histories"
        indexes = [
            models.Index(fields=['patient_number']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Patient {self.patient_number} - {self.model_used} - {self.created_at}"


class DatasetUpload(models.Model):
    """Track dataset uploads."""

    data_file_name = models.CharField(max_length=255)
    labels_file_name = models.CharField(max_length=255)
    upload_date = models.DateTimeField(default=timezone.now)
    total_samples = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['-upload_date']

    def __str__(self):
        return f"Dataset uploaded on {self.upload_date}"


class ModelPerformance(models.Model):
    """Store model performance metrics."""

    model_name = models.CharField(max_length=100, unique=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-accuracy']
        verbose_name_plural = "Model Performances"

    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy}%"

