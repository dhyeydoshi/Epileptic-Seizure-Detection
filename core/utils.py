"""
Utility functions for visualization and plotting.
Version 2.0 - Optimized plotting functions.
"""
import base64
import io
from typing import Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate plots for model evaluation."""

    @staticmethod
    def plot_roc_curve(y_test: np.ndarray, predictions: np.ndarray, label_name: str) -> str:
        """
        Generate ROC curve plot as base64 encoded string.

        Args:
            y_test: True labels
            predictions: Predicted probabilities
            label_name: Name for the legend

        Returns:
            Base64 encoded PNG image
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            auc = roc_auc_score(y_test, predictions)

            plt.figure(figsize=(10, 8), dpi=150)
            plt.title('ROC Curve', fontsize=16, fontweight='bold')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
            plt.plot(fpr, tpr, 'c', marker='.', linewidth=2,
                    label=f'{label_name.upper()} (AUC = {auc:.3f})')
            plt.legend(loc='lower right', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            return PlotGenerator._fig_to_base64()
        except Exception as e:
            logger.error(f"Error generating ROC curve: {str(e)}")
            raise

    @staticmethod
    def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate confusion matrix heatmap as base64 encoded string.

        Args:
            y_test: True labels
            y_pred: Predicted labels

        Returns:
            Base64 encoded PNG image
        """
        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(10, 8), dpi=150)
            sns.set(font_scale=1.2)
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=True,
                       annot_kws={'size': 16, 'weight': 'bold'},
                       cbar_kws={'label': 'Count'})
            plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold')
            plt.ylabel('True Labels', fontsize=14, fontweight='bold')
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()

            return PlotGenerator._fig_to_base64()
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            raise
        finally:
            sns.reset_defaults()

    @staticmethod
    def _fig_to_base64() -> str:
        """Convert current matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict) -> str:
        """
        Generate bar plot comparing different metrics.

        Args:
            metrics_dict: Dictionary of metric names and values

        Returns:
            Base64 encoded PNG image
        """
        try:
            plt.figure(figsize=(12, 6), dpi=150)
            metrics = list(metrics_dict.keys())
            values = list(metrics_dict.values())

            bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
            plt.xlabel('Metrics', fontsize=12, fontweight='bold')
            plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            return PlotGenerator._fig_to_base64()
        except Exception as e:
            logger.error(f"Error generating metrics comparison: {str(e)}")
            raise

