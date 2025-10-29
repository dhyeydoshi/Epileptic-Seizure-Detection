"""
Form validators and custom forms.
Version 2.0 - Added form validation.
"""
from django import forms
from django.core.exceptions import ValidationError
import pandas as pd


class DatasetUploadForm(forms.Form):
    """Form for uploading dataset files with validation."""

    data_file = forms.FileField(
        label='Data File (CSV)',
        help_text='Upload the EEG data CSV file',
        required=True
    )
    labels_file = forms.FileField(
        label='Labels File (CSV)',
        help_text='Upload the labels CSV file',
        required=True
    )

    def clean_data_file(self):
        """Validate data file format."""
        data_file = self.cleaned_data['data_file']

        if not data_file.name.endswith('.csv'):
            raise ValidationError('Data file must be a CSV file.')

        # Check file size (max 100MB)
        if data_file.size > 100 * 1024 * 1024:
            raise ValidationError('File size must be less than 100MB.')

        return data_file

    def clean_labels_file(self):
        """Validate labels file format."""
        labels_file = self.cleaned_data['labels_file']

        if not labels_file.name.endswith('.csv'):
            raise ValidationError('Labels file must be a CSV file.')

        # Check file size (max 10MB)
        if labels_file.size > 10 * 1024 * 1024:
            raise ValidationError('File size must be less than 10MB.')

        return labels_file


class PatientPredictionForm(forms.Form):
    """Form for patient prediction input."""

    patient_number = forms.CharField(
        max_length=50,
        label='Patient Number',
        help_text='Enter the patient ID',
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 001'
        })
    )

    threshold = forms.FloatField(
        initial=0.5,
        min_value=0.0,
        max_value=1.0,
        label='Classification Threshold',
        help_text='Threshold for epilepsy classification (0.0 - 1.0)',
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01'
        })
    )

    def clean_patient_number(self):
        """Validate patient number format."""
        patient_number = self.cleaned_data['patient_number']

        # Remove leading/trailing whitespace
        patient_number = patient_number.strip()

        if not patient_number:
            raise ValidationError('Patient number cannot be empty.')

        return patient_number

