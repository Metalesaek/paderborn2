from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    #path('', views.once_call, name='once_call'),
    path('general', views.general, name='general'),
    path('function_call', views.function_call, name='function_call'),
    path('arxiv', views.arxiv, name='arxiv'),
    path('bard', views.bard, name='bard'),


    ]

