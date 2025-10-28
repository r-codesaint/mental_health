from django.urls import path
from . import views

app_name = 'mental'

urlpatterns = [
    path("", views.index, name="index"),
    #path("<int:pk>/", views.detail, name="detail"),
    path("login/", views.login, name="login"),
    path("signup/", views.signup, name="signup"),
    path("survey/", views.survey, name="survey"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", views.logout_view, name="logout"),
    path("api/latest-score/", views.get_latest_score, name="get_latest_score")
]
