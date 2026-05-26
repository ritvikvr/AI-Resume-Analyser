from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ResumeViewSet, JobDescriptionViewSet, AnalysisViewSet, SessionViewSet

router = DefaultRouter()
router.register(r'resumes', ResumeViewSet)
router.register(r'jobs', JobDescriptionViewSet)
router.register(r'analysis', AnalysisViewSet)
router.register(r'sessions', SessionViewSet, basename='session')

urlpatterns = [
    path('', include(router.urls)),
]
