from django.urls import path
from .views import index, TrainFromCSVView

urlpatterns = [
    path('', index, name='index'),
    path('api/train_csv/', TrainFromCSVView.as_view(), name='train-from-csv'),
]
