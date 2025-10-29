"""
Django admin configuration for epileptic seizure detection.
Version 2.3 - Activated admin interface for tracking models.
"""
from django.contrib import admin
from django.utils.html import format_html
from .models import PredictionHistory, DatasetUpload, ModelPerformance


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    """Admin interface for prediction history."""

    list_display = ['patient_number', 'model_used', 'result_display', 'confidence_display', 'created_at']
    list_filter = ['model_used', 'has_epilepsy', 'created_at']
    search_fields = ['patient_number', 'ip_address']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'

    def result_display(self, obj):
        """Display result with color coding."""
        if obj.has_epilepsy:
            return format_html('<span style="color: red; font-weight: bold;">Epileptic</span>')
        return format_html('<span style="color: green; font-weight: bold;">Normal</span>')
    result_display.short_description = 'Result'

    def confidence_display(self, obj):
        """Display confidence as percentage."""
        percentage = obj.confidence * 100
        color = 'green' if percentage > 80 else 'orange' if percentage > 60 else 'red'
        return format_html(f'<span style="color: {color};">{percentage:.2f}%</span>')
    confidence_display.short_description = 'Confidence'


@admin.register(DatasetUpload)
class DatasetUploadAdmin(admin.ModelAdmin):
    """Admin interface for dataset uploads."""

    list_display = ['data_file_name', 'labels_file_name', 'total_samples', 'upload_date']
    list_filter = ['upload_date']
    search_fields = ['data_file_name', 'labels_file_name']
    readonly_fields = ['upload_date']
    date_hierarchy = 'upload_date'


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    """Admin interface for model performance."""

    list_display = ['model_name', 'accuracy_display', 'precision', 'recall', 'f1_score', 'total_predictions', 'last_updated']
    list_filter = ['last_updated']
    search_fields = ['model_name']
    readonly_fields = ['last_updated', 'total_predictions']

    def accuracy_display(self, obj):
        """Display accuracy with color coding."""
        color = 'green' if obj.accuracy > 90 else 'orange' if obj.accuracy > 80 else 'red'
        return format_html(f'<span style="color: {color}; font-weight: bold;">{obj.accuracy:.2f}%</span>')
    accuracy_display.short_description = 'Accuracy'
