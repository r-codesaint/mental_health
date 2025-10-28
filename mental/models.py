from django.db import models


class User(models.Model):
    user_id = models.AutoField(primary_key=True, editable=False, unique=True)
    username = models.CharField(max_length=500, unique=True)
    first_name = models.CharField(max_length=500)
    last_name = models.CharField(max_length=500)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=500)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username

class SurveyResponse(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=50)
    country = models.CharField(max_length=100)
    occupation = models.CharField(max_length=100)
    self_employed = models.CharField(max_length=5)  # Yes/No
    family_history = models.CharField(max_length=5)  # Yes/No
    treatment = models.CharField(max_length=5)  # Yes/No
    days_indoors = models.CharField(max_length=50)
    growing_stress = models.CharField(max_length=50)
    changes_habits = models.CharField(max_length=50)
    Mood_Swings= models.CharField(max_length=5)
    Mental_Health_Hist= models.CharField(max_length=5)
    Coping_Struggles= models.CharField(max_length=5)
    Work_Interest= models.CharField(max_length=5)
    Social_Weakness= models.CharField(max_length=5)
    mental_health_interview= models.CharField(max_length=5)
    care_options= models.CharField(max_length=5)
    mood_prediction = models.CharField(max_length=50, null=True)  # To store ML prediction
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Survey Response from {self.user.username} - {self.created_at}"
