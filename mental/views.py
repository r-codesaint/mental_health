from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import User, SurveyResponse
import json
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# def home(request):
#     User=User.objects.all()

def index(request):
    # Landing page (index). User starts here and can go to login or signup.
    return render(request, "mental/index.html")


def dashboard(request):
    # Home/dashboard page shown after completing the survey
    # Require a logged in user (redirect to login if missing)
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('mental:login')

    try:
        # Get user data
        user = User.objects.get(email=user_email)
        # Get latest survey response
        latest_survey = SurveyResponse.objects.filter(user=user).order_by('-created_at').first()

        context = {
            'user': user,
            'full_name': f"{user.first_name} {user.last_name}",
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
            'username': user.username,
            'join_date': user.created.strftime('%B %Y'),
            'mood_prediction': latest_survey.mood_prediction if latest_survey else None,
            'mood_score': latest_survey.mood_score if latest_survey else 50,
            'survey_data': {
                'gender': latest_survey.gender if latest_survey else '',
                'occupation': latest_survey.occupation if latest_survey else '',
                'country': latest_survey.country if latest_survey else 'India',
                'self_employed': latest_survey.self_employed if latest_survey else '',
                'family_history': latest_survey.family_history if latest_survey else '',
                'treatment': latest_survey.treatment if latest_survey else '',
                'days_indoors': latest_survey.days_indoors if latest_survey else '',
                'growing_stress': latest_survey.growing_stress if latest_survey else '',
                'changes_habits': latest_survey.changes_habits if latest_survey else '',
            }
        }
        return render(request, "mental/home.html", context)
    except User.DoesNotExist:
        return redirect('mental:login')

@csrf_exempt
def login(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')

            if not email or not password:
                return JsonResponse({'status': 'error', 'message': 'Email and password are required'}, status=400)

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': 'Invalid email or password'}, status=400)

            # Note: passwords are stored in plain text in the current User model.
            # In production, use Django's auth system and hashed passwords.
            if user.password != password:
                return JsonResponse({'status': 'error', 'message': 'Invalid email or password'}, status=400)

            # Save user info into session so survey page can display name
            request.session['user_email'] = user.email
            request.session['user_name'] = f"{user.first_name} {user.last_name}"

            return JsonResponse({'status': 'success', 'redirect': '/survey/'})

        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return render(request, "mental/login.html")

@csrf_exempt
def signup(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            
            # Extract form data
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            first_name = data.get('firstName')
            last_name = data.get('lastName')
            
            # Validate required fields
            if not all([username, email, password, first_name, last_name]):
                return JsonResponse({
                    'status': 'error',
                    'message': 'All fields are required'
                }, status=400)
            
            # Check if username already exists
            if User.objects.filter(username=username).exists():
                return JsonResponse({
                    'status': 'error',
                    'message': 'Username already exists'
                }, status=400)
            
            # Check if email already exists
            if User.objects.filter(email=email).exists():
                return JsonResponse({
                    'status': 'error',
                    'message': 'Email already exists'
                }, status=400)
            
            # Create new user
            user = User.objects.create(
                username=username,
                email=email,
                password=password,  # In a real app, hash this password
                first_name=first_name,
                last_name=last_name
            )
            
            return JsonResponse({
                'status': 'success',
                'message': 'User created successfully',
                'redirect': '/login/'  # Frontend will handle redirect
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    # GET request - render signup page
    return render(request, "mental/signup.html")
@csrf_exempt
def survey(request):
    # Get user name from session and pass to template
    full_name = request.session.get('user_name')
    user_email = request.session.get('user_email')
    
    if not full_name or not user_email:
        # Not logged in; redirect to login
        return redirect('mental:login')

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            data['mood_swings'] = 'Medium' 
            data['Mental_Health_Hist']=data.get('treatment')
            data['Coping_Struggles'] = 'false'
            data['Work_Interest'] = 'No'
            data['Social_Weakness'] = 'No' 
            data['mental_health_interview'] =data.get('treatment')
            data['care_options'] ='No'
              # Default value for Mental_Health_Interview
            user = User.objects.get(email=user_email)

            # Create survey response
            survey_response = SurveyResponse.objects.create(
                user=user,
                gender=data.get('gender'),
                country=data.get('country'),
                occupation=data.get('occupation'),
                self_employed=data.get('self_employed'),
                family_history=data.get('family_history'),
                treatment=data.get('treatment'),
                days_indoors=data.get('days_indoors'),
                growing_stress=data.get('growing_stress'),
                changes_habits=data.get('changes_habits'),
                Mood_Swings=data.get('mood_swings'),
                Mental_Health_Hist=data.get('Mental_Health_Hist'),
                Coping_Struggles=data.get('Coping_Struggles'),
                Work_Interest=data.get('Work_Interest'),
                Social_Weakness=data.get('Social_Weakness'),
                mental_health_interview=data.get('mental_health_interview'),
                care_options=data.get('care_options')
            )

            # --- Prepare data for ML model (align feature names to training) ---
            meta_path = os.path.join(os.path.dirname(__file__), 'model_meta.joblib')
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                feature_cols = meta.get('columns', [])
                label_encoders = meta.get('label_encoders', {})
                country_map = meta.get('country_map', {})
            else:
                feature_cols = None
                label_encoders = {}
                country_map = {}

            # Build a zero-filled row matching the training columns
            if feature_cols:
                row = dict.fromkeys(feature_cols, 0)

                # Gender: training used LabelEncoder -> numeric
                if 'Gender' in row:
                    try:
                        le = label_encoders.get('Gender')
                        if le is not None:
                            row['Gender'] = int(le.transform([data.get('gender')])[0])
                        else:
                            # fallback: simple mapping
                            g = data.get('gender', '').lower()
                            row['Gender'] = 0 if g in ('male','m') else (1 if g in ('female','f') else 2)
                    except Exception:
                        row['Gender'] = 0

                # Country: use saved mapping if available
                if 'Country' in row:
                    c = data.get('country')
                    try:
                        if c in country_map:
                            row['Country'] = float(country_map[c])
                        else:
                            # fallback: numeric encode unknown countries to 0
                            row['Country'] = 0.0
                    except Exception:
                        row['Country'] = 0.0

                # Label-encoded booleans
                for col_name in ('self_employed', 'family_history', 'treatment', 'Coping_Struggles'):
                    if col_name in row:
                        raw = data.get(col_name) or data.get(col_name.lower())
                        if raw is None:
                            row[col_name] = 0
                        else:
                            try:
                                le = label_encoders.get(col_name)
                                if le is not None:
                                    row[col_name] = int(le.transform([raw])[0])
                                else:
                                    row[col_name] = 1 if str(raw).lower() in ('yes','y','true','1') else 0
                            except Exception:
                                row[col_name] = 1 if str(raw).lower() in ('yes','y','true','1') else 0

                # One-hot columns: Occupation_*, Days_Indoors_*, Growing_Stress_*, Changes_Habits_*
                def set_onehot(prefix, value):
                    if value is None:
                        return
                    for c in feature_cols:
                        if c.startswith(prefix + '_'):
                            suffix = c.split(prefix + '_', 1)[1]
                            # compare lowercased normalized forms
                            if suffix.replace(' ', '').replace('-', '').lower() == str(value).replace(' ', '').replace('-', '').lower():
                                row[c] = 1

                set_onehot('Occupation', data.get('occupation'))
                set_onehot('Days_Indoors', data.get('days_indoors'))
                set_onehot('Growing_Stress', data.get('growing_stress'))
                set_onehot('Changes_Habits', data.get('changes_habits'))

                # Numeric conveniences
                if 'Days_Indoors' in row and isinstance(data.get('days_indoors'), (int, float)):
                    # If training expects categories like '1-14 days', leave one-hot above to set correct bin
                    pass

                input_df = pd.DataFrame([row], columns=feature_cols)
            else:
                # No meta available — try a minimal DataFrame with keys we received
                input_df = pd.DataFrame([{
                    'Gender': data.get('gender'),
                    'Country': data.get('country'),
                    'Occupation': data.get('occupation'),
                    'self_employed': data.get('self_employed'),
                    'family_history': data.get('family_history'),
                    'treatment': data.get('treatment'),
                    'Days_Indoors': data.get('days_indoors'),
                    'Growing_Stress': data.get('growing_stress'),
                    'Changes_Habits': data.get('changes_habits'),
                    'Mood_Swings': data.get('mood_swings'),
                    'Mental_Health_Hist': data.get('Mental_Health_Hist'),
                    'Coping_Struggles': data.get('Coping_Struggles'),
                    'Work_Interest': data.get('Work_Interest'),
                    'Social_Weakness': data.get('Social_Weakness'),
                    'mental_health_interview': data.get('mental_health_interview'),
                    'care_options': data.get('care_options')
                }])

            # Initialize default values
            mood_level = 'Medium'  # default
            mood_score = 50  # default score out of 100
            
            try:
                # Get sleep hours and screen time from input data
                sleep_hours = float(data.get('sleep_hours', 0))
                watch_time = float(data.get('screen_hours', 0))

                # Calculate mood score based on sleep and screen time
                if sleep_hours < 6 and watch_time > 7:
                    # Poor sleep and high screen time - calculate score in 0-40 range
                    base_score = 40 - ((6 - sleep_hours) * 5) - ((watch_time - 7) * 2)
                    mood_score = max(0, min(40, base_score))
                    mood_level = 'Low'
                elif 6 <= sleep_hours <= 9 and watch_time < 4:
                    # Good sleep and low screen time - calculate score in 50-100 range
                    base_score = 75 + ((8 - sleep_hours) * 5) + ((4 - watch_time) * 5)
                    mood_score = max(50, min(100, base_score))
                    mood_level = 'High' if mood_score >= 80 else 'Good'
                else:
                    # Moderate conditions - score in 40-50 range
                    base_score = 45 + ((sleep_hours - 6) * 2) - ((watch_time - 4) * 2)
                    mood_score = max(40, min(50, base_score))
                    mood_level = 'Medium'
                
                # Round the final score to an integer
                mood_score = int(round(mood_score))

                logging.info(f'Calculated mood_score: {mood_score}, mood_level: {mood_level} based on sleep_hours: {sleep_hours}, watch_time: {watch_time}')
            except Exception as e:
                # Prediction failed — keep default and log
                print('Prediction error:', e)

            # Save prediction and score
            survey_response.mood_prediction = mood_level
            survey_response.mood_score = mood_score
            survey_response.save()

            # Store prediction and score in session for home page
            request.session['mood_prediction'] = mood_level
            request.session['mood_score'] = mood_score

            return JsonResponse({
                'status': 'success',
                'prediction': mood_level,
                'redirect': '/dashboard/'
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)

    return render(request, "mental/survey.html", {'full_name': full_name})


def logout_view(request):
    """Clear the session and redirect to login."""
    try:
        request.session.flush()
    except Exception:
        # ignore session errors
        pass
    return redirect('mental:login')


# AJAX endpoint to get latest mental health score
@csrf_exempt
def get_latest_score(request):
    """Get the latest mental health score for the current user."""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({
            'status': 'error',
            'message': 'Not logged in'
        }, status=401)

    try:
        user = User.objects.get(email=user_email)
        latest_survey = SurveyResponse.objects.filter(user=user).order_by('-created_at').first()
        
        if latest_survey:
            return JsonResponse({
                'status': 'success',
                'data': {
                    'mood_score': latest_survey.mood_score,
                    'mood_prediction': latest_survey.mood_prediction,
                    'created_at': latest_survey.created_at.strftime('%Y-%m-%d %H:%M:%S')
                }
            })
        else:
            return JsonResponse({
                'status': 'success',
                'data': {
                    'mood_score': 50,  # default score
                    'mood_prediction': 'Medium',
                    'created_at': None
                }
            })
            
    except User.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'User not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
