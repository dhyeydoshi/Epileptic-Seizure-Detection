"""
Views for epileptic seizure detection application.
Version 2.0 - Optimized with proper cleanup and caching.
"""
import os
import pandas as pd
from django.shortcuts import render, redirect
from django.views import View
from django.views.generic import TemplateView
from django.contrib import messages
from django.core.cache import cache
from django.conf import settings
import logging
import tempfile

from .services import (
    DatasetProcessor, ModelEvaluator, MLModelService,
    PredictionService, ModelCache
)
from .utils import PlotGenerator
from .forms import DatasetUploadForm, PatientPredictionForm

logger = logging.getLogger(__name__)

# Global dataset instance (stored in session in production)
_dataset_cache = {}

def _save_uploaded_to_temp(uploaded_file):
    """
    Save an UploadedFile (InMemoryUploadedFile or TemporaryUploadedFile)
    to a NamedTemporaryFile and return its path.
    """
    suffix = os.path.splitext(uploaded_file.name)[1] or ''
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except Exception:
            pass
        raise


class DashboardView(View):
    """Main dashboard for uploading datasets."""

    template_name = 'core/dashboard.html'

    def get(self, request, *args, **kwargs):
        """Display dashboard with upload form."""
        form = DatasetUploadForm()
        messages_to_display = messages.get_messages(request)
        temporary_message = None
        for message in messages_to_display:
            temporary_message = message

        context = {
            'form': form,
            'temporary_message': temporary_message
        }
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        """Handle dataset upload."""
        form = DatasetUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, self.template_name, {'error': True, 'form': form})

        created_temp_files = []
        try:
            uploaded_data_file = form.cleaned_data['data_file']
            uploaded_labels_file = form.cleaned_data['labels_file']

            # Determine file paths (use temporary_file_path if available)
            if hasattr(uploaded_data_file, 'temporary_file_path'):
                df_data_path = uploaded_data_file.temporary_file_path()
            else:
                df_data_path = _save_uploaded_to_temp(uploaded_data_file)
                created_temp_files.append(df_data_path)

            if hasattr(uploaded_labels_file, 'temporary_file_path'):
                df_labels_path = uploaded_labels_file.temporary_file_path()
            else:
                df_labels_path = _save_uploaded_to_temp(uploaded_labels_file)
                created_temp_files.append(df_labels_path)

            # Process dataset
            dataset = DatasetProcessor(data_path=df_data_path, label_path=df_labels_path)

            # Store in session (use cache or database in production)
            request.session['dataset_loaded'] = True
            _dataset_cache['current'] = dataset

            # Save upload to database for tracking
            try:
                from .models import DatasetUpload

                # Get client IP
                x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
                if x_forwarded_for:
                    ip_address = x_forwarded_for.split(',')[0].strip()
                else:
                    ip_address = request.META.get('REMOTE_ADDR')

                DatasetUpload.objects.create(
                    data_file_name=uploaded_data_file.name,
                    labels_file_name=uploaded_labels_file.name,
                    total_samples=len(dataset.data),
                    uploaded_by_ip=ip_address
                )
            except Exception as e:
                logger.warning(f"Failed to save dataset upload record: {str(e)}")

            messages.success(request, 'Dataset uploaded and processed successfully!')
            logger.info(f"Dataset uploaded: {uploaded_data_file.name}, {uploaded_labels_file.name}")

            return render(request, self.template_name, {'success': True, 'form': DatasetUploadForm()})

        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            messages.error(request, f'Error processing dataset: {str(e)}')
            return render(request, self.template_name, {'error': True, 'form': form})

        finally:
            # Clean up any temp files we created
            for path in created_temp_files:
                try:
                    os.unlink(path)
                    logger.debug(f"Cleaned up temp file: {path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temp file {path}: {cleanup_error}")


class BaseModelView(TemplateView):
    """Base view for all model evaluation views."""
    
    model_file_name = None
    model_display_name = None
    cache_timeout = 300  # 5 minutes cache
    
    def get_dataset(self, request):
        """Get the current dataset from cache."""
        if 'current' not in _dataset_cache:
            messages.warning(request, 'Warning: Data and Label not uploaded.')
            return None
        return _dataset_cache['current']
    
    def get(self, request, *args, **kwargs):
        """Evaluate model and display results with caching."""
        # Check cache first
        cache_key = f'model_eval_{self.model_file_name}_{request.session.session_key}'
        cached_result = cache.get(cache_key)
        
        if cached_result and settings.DEBUG is False:
            logger.info(f"Returning cached result for {self.model_display_name}")
            return render(request, self.template_name, cached_result)
        
        dataset = self.get_dataset(request)
        if dataset is None:
            return redirect('core:dashboard')
        
        try:
            # Load model using service
            model_service = MLModelService(self.model_file_name)
            model_service.load_model()
            
            # Make predictions
            if self.model_file_name.endswith('.h5'):
                # Deep learning model - lazy import
                predictions = model_service.predict(dataset.X_val)
                y_pred_labels = [1 if x > 0.5 else 0 for x in predictions.flatten()]
                probs = predictions.flatten()
            else:
                # Traditional ML model
                y_pred_labels = model_service.predict(dataset.X_val)
                probs = model_service.predict_proba(dataset.X_val)
            
            # Calculate metrics
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_metrics(dataset.y_test, y_pred_labels)
            
            # Generate plots
            plot_gen = PlotGenerator()
            confusion_matrix_plot = plot_gen.plot_confusion_matrix(dataset.y_test, y_pred_labels)
            roc_curve_plot = plot_gen.plot_roc_curve(dataset.y_test, probs, self.model_display_name)
            
            # Get classification report
            classification_report = evaluator.get_classification_report(dataset.y_test, y_pred_labels)
            
            # Prepare context
            context = {
                'model_name': self.model_display_name,
                'classificationReport': classification_report,
                'confusion_matrix_string': confusion_matrix_plot,
                'roc_curve_string': roc_curve_plot,
                **metrics
            }
            
            # Cache the result
            cache.set(cache_key, context, self.cache_timeout)
            
            logger.info(f"{self.model_display_name} evaluation completed: {metrics}")
            return render(request, self.template_name, context)
            
        except Exception as e:
            logger.error(f"Error evaluating {self.model_display_name}: {str(e)}")
            messages.error(request, f'Error evaluating model: {str(e)}')
            return redirect('core:dashboard')


class CNNView(BaseModelView):
    """View for CNN/Deep Learning model."""
    template_name = 'core/CNN.html'
    model_file_name = 'DeepLearning.h5'
    model_display_name = 'CNN'


class SVMView(BaseModelView):
    """View for SVM model."""
    template_name = 'core/svm.html'
    model_file_name = 'SVMModel.pickle'
    model_display_name = 'SVM'


class KNNView(BaseModelView):
    """View for KNN model."""
    template_name = 'core/knn.html'
    model_file_name = 'KNNModel.pickle'
    model_display_name = 'KNN'


class NaiveBayesView(BaseModelView):
    """View for Naive Bayes model."""
    template_name = 'core/naive.html'
    model_file_name = 'NaiveBayesModel.pickle'
    model_display_name = 'Naive Bayes'


class RandomForestView(BaseModelView):
    """View for Random Forest model."""
    template_name = 'core/random_forest.html'
    model_file_name = 'RandomForestModel.pickle'
    model_display_name = 'Random Forest'


class XgBoostView(BaseModelView):
    """View for XGBoost model."""
    template_name = 'core/xgboost.html'
    model_file_name = 'XgBoostModel.pickle'
    model_display_name = 'XGBoost'


class LogisticView(BaseModelView):
    """View for Logistic Regression model."""
    template_name = 'core/logistic.html'
    model_file_name = 'LogisticRegressionModel.pickle'
    model_display_name = 'Logistic Regression'


class VisualizationView(TemplateView):
    """View for data visualization."""
    template_name = 'core/visualization.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add visualization data if needed
        return context


class BestModelView(TemplateView):
    """View for best model prediction."""
    template_name = 'core/best_model.html'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models_dataframe = self._load_model_performance()

    def _load_model_performance(self):
        """Load model performance data."""
        try:
            module_dir = os.path.dirname(__file__)
            csv_path = os.path.join(module_dir, '../data/model_acc_dataframe.csv')
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading model performance data: {str(e)}")
            return pd.DataFrame()

    def get(self, request, *args, **kwargs):
        """Display best model form."""
        form = PatientPredictionForm()

        if self.models_dataframe.empty:
            best_model_name = "CNN"
        else:
            best_model_name = self.models_dataframe.iloc[0]['Model']

        context = {
            'form': form,
            'success': False,
            'best_model_name': best_model_name
        }
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        """Handle patient prediction request."""
        form = PatientPredictionForm(request.POST)

        if self.models_dataframe.empty:
            best_model_name = "CNN"
            saved_model_name = "DeepLearning.h5"
        else:
            best_model_name = self.models_dataframe.iloc[0]['Model']
            saved_model_name = self.models_dataframe.iloc[0]['SavedModelName']

        if form.is_valid():
            try:
                patient_number = form.cleaned_data['patient_number']
                threshold = form.cleaned_data.get('threshold', 0.5)

                # Create model service and prediction service
                model_service = MLModelService(saved_model_name)
                prediction_service = PredictionService(model_service)

                # Make prediction
                result = prediction_service.predict_patient(patient_number, threshold)

                context = {
                    'form': form,
                    'output_string': result['message'],
                    'confidence': result['confidence'],
                    'success': True,
                    'best_model_name': best_model_name,
                    'predicted_class': result['predicted_class']
                }

                logger.info(f"Prediction made for patient {patient_number}: {result}")
                return render(request, self.template_name, context)

            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                messages.error(request, f'Error making prediction: {str(e)}')
                context = {
                    'form': form,
                    'success': False,
                    'best_model_name': best_model_name,
                    'error_message': str(e)
                }
                return render(request, self.template_name, context)
        else:
            context = {
                'form': form,
                'success': False,
                'best_model_name': best_model_name
            }
            return render(request, self.template_name, context)
