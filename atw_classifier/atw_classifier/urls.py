"""atw_classifier URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from classification.views import classification, jupy_nb
from photos.views import photo_list, photo_list_classification
from results.views import results, analysis
from filter_results.views import filter_results, filter_result_table_page, photo_list_filtered

from django.views.generic import TemplateView


urlpatterns = [
    path('', classification),
    
    path('classification/', classification, name="classification_script"),
    path('jupy_nb/', jupy_nb, name="jupy_nb_script"),
    path('admin/', admin.site.urls, name='admin'),

    path('photo_viewer/', photo_list, name='photo_list'),
    path('photo_list_filtered/', photo_list_filtered, name='photo_list_filtered'),
    

    url(r'^$', TemplateView.as_view(template_name='home.html'), name='home'),
    url(r'^photos/', include('photos.urls', namespace='photos')),

    
    path('results/', results, name='results'),
    path('result_table', TemplateView.as_view(template_name="table.html"),name='result_table'),

    path('filter_result_table_page/', filter_result_table_page, name='filter_result_table_page'),
    path('filter_table', TemplateView.as_view(template_name="filter_result.html"),name='filter_table'),

    path('analysis/', analysis, name='analysis'),
    path('filter_results/', filter_results, name='filter_results'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

