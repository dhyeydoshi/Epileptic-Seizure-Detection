import base64
import io

from django.shortcuts import render
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score , accuracy_score , precision_score, recall_score ,confusion_matrix, classification_report, roc_curve, f1_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from django.views import View
from django.views.generic import TemplateView


# Create your views here.
module_dir = os.path.dirname(__file__)
class BaseDatasetClass:

    def __init__(self, data_path=None, label_path=None):
        if data_path and label_path:
            self.data_path = data_path
            self.label_path = label_path
            self.update_data()

    def update_data(self):
        self.og_data = pd.read_csv(self.data_path)
        self.og_labels = pd.read_csv(self.label_path)
        self.split_data()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.og_data, self.og_labels, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

og_data = pd.read_csv(os.path.join(module_dir, '../','data/data.csv'))
og_labels = pd.read_csv(os.path.join(module_dir, '../','data/labels.csv'))

X_train, X_test, y_train, y_test = train_test_split(og_data,og_labels,test_size=0.2,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

class DashboardView(View):
    template_name = 'core/dashboard.html'

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get('file')
        request.session['file_name'] = uploaded_file.name
        if uploaded_file:
            report = {
                'success': True
            }
            return render(request, self.template_name, report)
        else:
            return render(request, self.template_name, {'error': True})

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)




class CNNView(TemplateView):
    template_name = 'core/CNN.html'

    def get(self, request, *args, **kwargs):
        algo_path = os.path.join(module_dir, '../', 'model', 'DeepLearning.h5')
        model = load_model(algo_path)

        # Evaluate the model on the val set
        test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test)

        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        predictions = model.predict(X_val)
        print("Predictions", predictions)


        # Convert predictions to binary class labels
        y_pred_labels = [1 if x > 0.5 else 0 for x in predictions]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred_labels).replace('\n', '<br>')

        data = {
            'classificationReport': classificationReport,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred_labels),
            'roc_curve_string': self.plot_roc_curve(predictions)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, predictions):
        fpr, tpr, thresholds = roc_curve(y_val, predictions)
        auc = roc_auc_score(y_val, predictions)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, 'c', marker='.', label=f'cnn = %0.3f' % auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class SVMView(TemplateView):
    template_name = 'core/svm.html'

    def get(self, request, *args, **kwargs):

        algo_path = os.path.join(module_dir, '../', 'model','SVMModel.pickle')
        with open(algo_path, 'rb') as f:
            algo = pickle.load(f)

        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred) * 100
        print(f"Accuracy with SVM: {accuracy:.2f}%")
        accuracy = round(accuracy, 2)

        # Calculate precision
        precision = precision_score(y_val, y_pred) * 100
        precision = round(precision, 2)
        print('Precision:', precision)

        # Calculate recall
        recall = recall_score(y_val, y_pred) * 100
        recall = round(recall, 2)
        print('Recall:', recall)

        # Calculate F1-score
        f1 = f1_score(y_val, y_pred) * 100
        f1 = round(f1, 2)
        print('F1-Score:', f1)

        # Calculate area under ROC curve
        roc_auc = roc_auc_score(y_val, y_pred) * 100
        roc_auc = round(roc_auc, 2)
        print('ROC AUC Score:', roc_auc)

        probs = algo.predict_proba(X_val)
        probs = probs[:, 1]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred).replace('\n', '<br>')
        data = {
            'classificationReport': classificationReport,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred),
            'roc_curve_string': self.plot_roc_curve(probs)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, probs):
        svm_fpr, svm_tpr, thresholds = roc_curve(y_val, probs)
        svm_auc = roc_auc_score(y_val, probs)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(svm_fpr, svm_tpr, 'c', marker='.', label=f'svm = %0.3f' % svm_auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class KNNView(TemplateView):
    template_name = 'core/knn.html'

    def get(self, request, *args, **kwargs):

        algo_path = os.path.join(module_dir, '../', 'model','KNNModel.pickle')
        with open(algo_path, 'rb') as f:
            algo = pickle.load(f)

        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        accuracy = round(accuracy, 2)

        # Calculate precision
        precision = precision_score(y_val, y_pred) * 100
        precision = round(precision, 2)
        print('Precision:', precision)

        # Calculate recall
        recall = recall_score(y_val, y_pred) * 100
        recall = round(recall, 2)
        print('Recall:', recall)

        # Calculate F1-score
        f1 = f1_score(y_val, y_pred) * 100
        f1 = round(f1, 2)
        print('F1-Score:', f1)

        # Calculate area under ROC curve
        roc_auc = roc_auc_score(y_val, y_pred) * 100
        roc_auc = round(roc_auc, 2)
        print('ROC AUC Score:', roc_auc)

        probs = algo.predict_proba(X_val)
        probs = probs[:, 1]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred).replace('\n', '<br>')
        data = {
            'classificationReport': classificationReport,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred),
            'roc_curve_string': self.plot_roc_curve(probs)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, probs):
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, 'c', marker='.', label=f'knn = %0.3f' % auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class NaiveBayesView(TemplateView):
    template_name = 'core/naive.html'

    def get(self, request, *args, **kwargs):

        algo_path = os.path.join(module_dir, '../', 'model','NaiveBayesModel.pickle')
        with open(algo_path, 'rb') as f:
            algo = pickle.load(f)

        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        accuracy = round(accuracy, 2)

        # Calculate precision
        precision = precision_score(y_val, y_pred) * 100
        precision = round(precision, 2)
        print('Precision:', precision)

        # Calculate recall
        recall = recall_score(y_val, y_pred) * 100
        recall = round(recall, 2)
        print('Recall:', recall)

        # Calculate F1-score
        f1 = f1_score(y_val, y_pred) * 100
        f1 = round(f1, 2)
        print('F1-Score:', f1)

        # Calculate area under ROC curve
        roc_auc = roc_auc_score(y_val, y_pred) * 100
        roc_auc = round(roc_auc, 2)
        print('ROC AUC Score:', roc_auc)

        probs = algo.predict_proba(X_val)
        probs = probs[:, 1]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred).replace('\n', '<br>')
        data = {
            'classificationReport': classificationReport,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred),
            'roc_curve_string': self.plot_roc_curve(probs)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, probs):
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, 'c', marker='.', label=f'naive = %0.3f' % auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class RandomForestView(TemplateView):
    template_name = 'core/random_forest.html'

    def get(self, request, *args, **kwargs):

        algo_path = os.path.join(module_dir, '../', 'model','RandomForestModel.pickle')
        with open(algo_path, 'rb') as f:
            algo = pickle.load(f)

        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        accuracy = round(accuracy, 2)

        # Calculate precision
        precision = precision_score(y_val, y_pred) * 100
        precision = round(precision, 2)
        print('Precision:', precision)

        # Calculate recall
        recall = recall_score(y_val, y_pred) * 100
        recall = round(recall, 2)
        print('Recall:', recall)

        # Calculate F1-score
        f1 = f1_score(y_val, y_pred) * 100
        f1 = round(f1, 2)
        print('F1-Score:', f1)

        # Calculate area under ROC curve
        roc_auc = roc_auc_score(y_val, y_pred) * 100
        roc_auc = round(roc_auc, 2)
        print('ROC AUC Score:', roc_auc)

        probs = algo.predict_proba(X_val)
        probs = probs[:, 1]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred).replace('\n', '<br>')
        data = {
            'classificationReport': classificationReport,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred),
            'roc_curve_string': self.plot_roc_curve(probs)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, probs):
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, 'c', marker='.', label=f'rf = %0.3f' % auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class XgBoostView(TemplateView):
    template_name = 'core/xgboost.html'

    def get(self, request, *args, **kwargs):

        algo_path = os.path.join(module_dir, '../', 'model','XgBoostModel.pickle')
        with open(algo_path, 'rb') as f:
            algo = pickle.load(f)

        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        accuracy = round(accuracy, 2)

        # Calculate precision
        precision = precision_score(y_val, y_pred) * 100
        precision = round(precision, 2)
        print('Precision:', precision)

        # Calculate recall
        recall = recall_score(y_val, y_pred) * 100
        recall = round(recall, 2)
        print('Recall:', recall)

        # Calculate F1-score
        f1 = f1_score(y_val, y_pred) * 100
        f1 = round(f1, 2)
        print('F1-Score:', f1)

        # Calculate area under ROC curve
        roc_auc = roc_auc_score(y_val, y_pred) * 100
        roc_auc = round(roc_auc, 2)
        print('ROC AUC Score:', roc_auc)

        probs = algo.predict_proba(X_val)
        probs = probs[:, 1]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred).replace('\n', '<br>')
        data = {
            'classificationReport': classificationReport,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred),
            'roc_curve_string': self.plot_roc_curve(probs)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, probs):
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, 'c', marker='.', label=f'xgboost = %0.3f' % auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class LogisticView(TemplateView):
    template_name = 'core/logistic.html'

    def get(self, request, *args, **kwargs):

        algo_path = os.path.join(module_dir, '../', 'model','LogisticRegressionModel.pickle')
        with open(algo_path, 'rb') as f:
            algo = pickle.load(f)

        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred) * 100
        print(f"Accuracy with Logistic: {accuracy:.2f}%")
        accuracy = round(accuracy, 2)

        # Calculate precision
        precision = precision_score(y_val, y_pred) * 100
        precision = round(precision, 2)
        print('Precision:', precision)

        # Calculate recall
        recall = recall_score(y_val, y_pred) * 100
        recall = round(recall, 2)
        print('Recall:', recall)

        # Calculate F1-score
        f1 = f1_score(y_val, y_pred) * 100
        f1 = round(f1, 2)
        print('F1-Score:', f1)

        # Calculate area under ROC curve
        roc_auc = roc_auc_score(y_val, y_pred) * 100
        roc_auc = round(roc_auc, 2)
        print('ROC AUC Score:', roc_auc)

        probs = algo.predict_proba(X_val)
        probs = probs[:, 1]
        classificationReport = classification_report(y_true=y_val, y_pred=y_pred).replace('\n', '<br>')
        data = {
            'classificationReport': classificationReport,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix_string': self.plot_confusion_matrix(y_val, y_pred),
            'roc_curve_string': self.plot_roc_curve(probs)
        }
        return render(request, self.template_name, data)

    def plot_roc_curve(self, probs):
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)

        plt.figure(figsize=(20, 20), dpi=300)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, 'c', marker='.', label=f'svm = %0.3f' % auc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def plot_confusion_matrix(self, y_val, y_pred):
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(20, 20), dpi=300)
        sns.set(font_scale=5)  # Adjust the font scale for better visibility
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode('utf-8')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class VisualizationView(TemplateView):
    template_name = 'core/visualization.html'

    def get(self, request, *args, **kwargs):

        context = self.get_context_data()
        return self.render_to_response(context=context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class BestModelView(TemplateView):
    template_name = 'core/best_model.html'

    def get(self, request, *args, **kwargs):
        models_dataframe = pd.read_csv(os.path.join(module_dir, '../', 'data/model_acc_dataframe.csv'))

        view = models_dataframe.iloc[0]['View']
        view = eval(view).as_view()

        return view(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        patient_number = request.POST.get('patient_number')

        algo_path = os.path.join(module_dir, '../', 'model', 'DeepLearning.h5')
        model = load_model(algo_path)

        data_path = os.path.join(module_dir, '../', 'Epileptic Seizure Recognition.csv')
        dataset = pd.read_csv(data_path)

        dataset = dataset[dataset['Unnamed'].str.split('.').str[2] == patient_number]
        data_x = dataset.drop(['Unnamed', 'y'], axis=1).copy()
        data_x = scaler.transform(data_x)
        data_y = dataset['y'].replace([2,3,4,5],0).copy()

        predictions = model.predict(data_x)
        binary_predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
        print(binary_predictions)
        # Threshold for classification
        threshold = 0.5

        # Apply threshold and classify patient's output
        predicted_class = 1 if np.mean(binary_predictions) >= threshold else 0

        print(predicted_class)
        # Print prediction
        if predicted_class == 1:
            output_string = f"The patient {patient_number} is predicted to have epilepsy."
        else:
            output_string = f"The patient {patient_number} is predicted to not have epilepsy."

        data = {
            'output_string': output_string,
            'success': True
        }
        return render(request, self.template_name, data)

