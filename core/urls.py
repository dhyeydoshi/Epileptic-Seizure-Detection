from django.urls import path

from core.views import DashboardView, CNNView, VisualizationView, SVMView, LogisticView, KNNView, NaiveBayesView, \
    RandomForestView, XgBoostView, BestModelView

app_name = 'core'
urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('deep-learning/', CNNView.as_view(), name='cnn'),
    path('svm/', SVMView.as_view(), name='svm'),
    path('knn/', KNNView.as_view(), name='knn'),
    path('naive/', NaiveBayesView.as_view(), name='naive'),
    path('logistic/', LogisticView.as_view(), name='logistic'),
    path('random-forest/', RandomForestView.as_view(), name='random_forest'),
    path('xgboost/', XgBoostView.as_view(), name='xgboost'),
    path('visualization/', VisualizationView.as_view(), name='charts'),
    path('best-model/', BestModelView.as_view(), name='best_model'),
]
