"""
Business logic and model services for epileptic seizure detection.
Version 2.0 - Optimized with lazy loading and better resource management.
"""
import os
import pickle
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_curve, f1_score
)
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """Singleton cache for loaded models to avoid repeated disk I/O."""

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_name: str, model_path: str):
        """Get model from cache or load it."""
        if model_name not in self._models:
            logger.info(f"Loading model: {model_name}")
            if model_name.endswith('.h5'):
                # Lazy import TensorFlow only when needed
                try:
                    from tensorflow.keras.models import load_model
                    self._models[model_name] = load_model(model_path)
                except ImportError as e:
                    logger.error(f"TensorFlow not available: {e}")
                    raise ImportError("TensorFlow is required for loading .h5 models. Install with: pip install tensorflow")
            else:
                with open(model_path, 'rb') as f:
                    self._models[model_name] = pickle.load(f)
        return self._models[model_name]

    def clear(self):
        """Clear all cached models."""
        logger.info(f"Clearing {len(self._models)} cached models")
        self._models.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            'cached_models_count': len(self._models),
            'cached_model_names': list(self._models.keys())
        }


class DatasetProcessor:
    """Handle dataset loading, preprocessing, and splitting."""

    def __init__(self, data_path: Optional[str] = None, label_path: Optional[str] = None):
        self.data_path = data_path
        self.label_path = label_path
        self.scaler = StandardScaler()

        if data_path and label_path:
            self._load_and_process()

    def _load_and_process(self):
        """Load and process the dataset."""
        try:
            # Use low_memory=False to avoid dtype warnings
            self.og_data = pd.read_csv(self.data_path, low_memory=False)
            self.og_labels = pd.read_csv(self.label_path, low_memory=False)
            self._split_data()
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _split_data(self):
        """Split data into train, validation, and test sets."""
        # Initial split: 80% train, 20% test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.og_data, self.og_labels,
            test_size=0.2, random_state=42, stratify=self.og_labels
        )

        # Split train into train and validation: 60% train, 20% val, 20% test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.25, random_state=42, stratify=self.y_train
        )

        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info(f"Dataset split - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")


class ModelEvaluator:
    """Evaluate model performance and generate metrics."""

    @staticmethod
    def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        try:
            # Convert to numpy arrays if needed
            y_test = np.asarray(y_test).flatten()
            y_pred = np.asarray(y_pred).flatten()

            metrics = {
                'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
                'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
                'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
                'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
                'roc_auc': round(roc_auc_score(y_test, y_pred) * 100, 2),
            }

            if settings.DEBUG:
                logger.debug(f"Metrics calculated: {metrics}")

            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    @staticmethod
    def get_classification_report(y_test: np.ndarray, y_pred: np.ndarray) -> str:
        """Get formatted classification report."""
        return classification_report(y_true=y_test, y_pred=y_pred).replace('\n', '<br>')

    @staticmethod
    def get_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_test, y_pred)

    @staticmethod
    def get_roc_curve_data(y_test: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc = roc_auc_score(y_test, predictions)
        return fpr, tpr, auc


class MLModelService:
    """Service for loading and running ML models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_cache = ModelCache()
        self.model_path = self._get_model_path()
        self.model = None

    def _get_model_path(self) -> str:
        """Get the full path to the model file."""
        model_dir = os.path.join(os.path.dirname(__file__), '../model')
        return os.path.join(model_dir, self.model_name)

    def load_model(self):
        """Load the model from cache or disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = self.model_cache.get_model(self.model_name, self.model_path)
        return self.model

    def predict(self, X_data: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model."""
        if self.model is None:
            self.load_model()

        return self.model.predict(X_data)

    def predict_proba(self, X_data: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            self.load_model()

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_data)[:, 1]
        else:
            # For models without predict_proba (like neural networks)
            predictions = self.model.predict(X_data)
            return predictions.flatten() if len(predictions.shape) > 1 else predictions


class PredictionService:
    """Service for making predictions on patient data."""

    def __init__(self, model_service: MLModelService):
        self.model_service = model_service

    def predict_patient(self, patient_number: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict epilepsy for a specific patient.

        Args:
            patient_number: Patient identifier
            threshold: Classification threshold

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load patient data
            data_path = os.path.join(
                os.path.dirname(__file__),
                '../data/Epileptic Seizure Recognition.csv'
            )

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}")

            # Use chunksize for large files
            dataset = pd.read_csv(data_path, low_memory=False)

            # Filter patient data
            patient_data = dataset[
                dataset['Unnamed'].str.split('.').str[2] == patient_number
            ].copy()

            if patient_data.empty:
                raise ValueError(f"No data found for patient {patient_number}")

            # Prepare features
            X_data = patient_data.drop(['Unnamed', 'y'], axis=1).copy()
            scaler = StandardScaler()
            X_data_scaled = scaler.fit_transform(X_data)

            # Make predictions
            predictions = self.model_service.predict(X_data_scaled)

            # Convert to binary predictions
            binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions.flatten()]

            # Calculate final classification
            mean_prediction = np.mean(binary_predictions)
            predicted_class = 1 if mean_prediction >= threshold else 0

            result = {
                'patient_number': patient_number,
                'predicted_class': predicted_class,
                'confidence': float(round(mean_prediction * 100, 2)),
                'has_epilepsy': predicted_class == 1,
                'message': self._get_prediction_message(patient_number, predicted_class)
            }

            logger.info(f"Prediction for patient {patient_number}: {result}")
            return result

        except Exception as e:
            logger.error(f"Error predicting for patient {patient_number}: {str(e)}")
            raise

    @staticmethod
    def _get_prediction_message(patient_number: str, predicted_class: int) -> str:
        """Generate human-readable prediction message."""
        if predicted_class == 1:
            return f"Patient {patient_number} is predicted to have epilepsy."
        else:
            return f"Patient {patient_number} is predicted to not have epilepsy."
