# MindWell: Mental Health Analysis and Prediction Platform

## Overview
MindWell is a comprehensive mental health analysis platform that uses machine learning to predict mood levels and provide personalized wellness recommendations. The system combines user survey data with advanced ML models to deliver real-time mental health assessments and actionable insights.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Data Structure](#data-structure)
- [Machine Learning Models](#machine-learning-models)
- [Workflow](#workflow)
- [API Documentation](#api-documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Features
- 🔐 Secure user authentication and profile management
- 📊 Comprehensive mental health survey system
- 🤖 ML-powered mood prediction
- 📈 Real-time mental health scoring (0-100 scale)
- 🎯 Personalized recommendations
- 📱 Responsive dashboard interface
- 📊 Historical data tracking
- 🔄 Automatic score updates

## Technology Stack
### Backend
- **Framework**: Django 
- **Database**: SQLite3
- **ML Libraries**: 
  - scikit-learn
  - TensorFlow/Keras
  - LightGBM
  - NumPy
  - Pandas

### Frontend
- **Framework**: Bootstrap 5
- **JavaScript**: Vanilla JS + jQuery
- **CSS**: Custom styling + Bootstrap classes
- **Visualization**: Chart.js

### Machine Learning
- **Models**: 
  - HistGradientBoostingClassifier
  - Neural Network (ANN)
  - MultiOutputClassifier
- **Features**: 15+ mental health indicators
- **Preprocessing**: StandardScaler, LabelEncoder
- **Validation**: ROC-AUC, Confusion Matrix

## System Architecture

### High-Level Architecture
```
MindWell Platform
├── Frontend Layer
│   ├── User Interface (Bootstrap + JS)
│   ├── Data Visualization
│   └── Real-time Updates
├── Backend Layer
│   ├── Django Views/Controllers
│   ├── Authentication System
│   └── Data Processing
├── ML Layer
│   ├── Model Training
│   ├── Prediction Engine
│   └── Score Calculation
└── Data Layer
    ├── User Data
    ├── Survey Responses
    └── ML Model States
```

## Data Structure

### Database Schema

#### User Model
\`\`\`python
class User(models.Model):
    user_id = AutoField(primary_key=True)
    username = CharField(unique=True)
    email = EmailField(unique=True)
    first_name = CharField()
    last_name = CharField()
    created = DateTimeField(auto_now_add=True)
\`\`\`

#### Survey Response Model
\`\`\`python
class SurveyResponse(models.Model):
    user = ForeignKey(User)
    gender = CharField()
    country = CharField()
    occupation = CharField()
    self_employed = CharField()
    family_history = CharField()
    treatment = CharField()
    days_indoors = CharField()
    growing_stress = CharField()
    changes_habits = CharField()
    mood_prediction = CharField()
    mood_score = IntegerField()
    created_at = DateTimeField()
\`\`\`

## Machine Learning Models

### Model Architecture
1. **Primary Model**: HistGradientBoostingClassifier
   - Multi-output classification
   - Handles High/Medium/Low mood predictions
   - Feature importance analysis

2. **Neural Network Model**
   - Dense layers with dropout
   - Binary classification
   - Early stopping implementation

### Feature Engineering
- Label Encoding for categorical variables
- One-hot encoding for multi-category features
- Leave-one-out encoding for country data
- Standardization for numerical features

## Workflow

### User Journey
1. **Registration/Login**
   - User creates account or logs in
   - Basic profile information collected

2. **Survey Process**
   - User completes mental health survey
   - Data validation and preprocessing

3. **Analysis Phase**
   - ML model processes survey data
   - Generates mood prediction and score
   - Calculates confidence levels

4. **Results Presentation**
   - Dashboard displays scores and predictions
   - Personalized recommendations generated
   - Historical data visualization

### Data Flow
```
User Input → Validation → Preprocessing → ML Prediction → Score Calculation → Result Display
```

## API Documentation

### Endpoints

#### Authentication
- POST `/api/signup/` - User registration
- POST `/api/login/` - User authentication
- GET `/api/logout/` - Session termination

#### Survey
- POST `/api/survey/` - Submit survey responses
- GET `/api/latest-score/` - Fetch latest mental health score

#### Dashboard
- GET `/api/dashboard/` - Fetch dashboard data
- GET `/api/recommendations/` - Get personalized recommendations

## Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/mindwell.git
cd mindwell
\`\`\`

2. Create virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Set up database:
\`\`\`bash
python manage.py migrate
\`\`\`

5. Run development server:
\`\`\`bash
python manage.py runserver
\`\`\`

## Usage

### Starting the Application
1. Ensure all dependencies are installed
2. Activate virtual environment
3. Run Django development server
4. Access platform at `http://localhost:8000`

### Running ML Training
\`\`\`bash
python train_ann_model.py  # Train neural network
python mentalhealth.py     # Train gradient boosting model
\`\`\`

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Mental Health Dataset contributors
- Open-source ML community
- Django and Bootstrap teams