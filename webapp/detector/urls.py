from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.detect_view, name="detect"),
    path("project/", views.project_view, name="project"),
    path("evaluation/", views.evaluation_view, name="evaluation"),
    path("notebook/", views.notebook_view, name="notebook"),
    path("notebook/download/", views.notebook_download, name="notebook_download"),
    path("certification/", views.certification_view, name="certification"),
    path("slides/", views.slides_view, name="slides"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
