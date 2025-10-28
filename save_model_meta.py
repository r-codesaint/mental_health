import pandas as pd
import joblib
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('Mental Health Dataset.csv')
# Basic cleaning similar to training
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Prepare label encoders
label_cols = ['Gender','self_employed','family_history','treatment','Coping_Struggles']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    try:
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    except Exception as e:
        print(f'Could not encode {col}: {e}')

# One-hot encode the categorical columns used in training
onehot_cols = ["Occupation", "Days_Indoors", "Growing_Stress",
            "Changes_Habits", "Mental_Health_History", "Work_Interest", "Social_Weakness",
            "mental_health_interview", "care_options"]

data = pd.get_dummies(data=df, columns=onehot_cols)
# Also dummy the target Mood_Swings
data = pd.get_dummies(data=data, columns=["Mood_Swings"]) if 'Mood_Swings' in df.columns else data

# Drop Timestamp if exists
if 'Timestamp' in data.columns:
    data.drop('Timestamp', axis=1, inplace=True)

# Fit leave-one-out encoder for Country if possible
leave_encoder = LeaveOneOutEncoder()
try:
    # choose a target column similar to training (first Mood_Swings_* column)
    target_cols = [c for c in data.columns if c.startswith('Mood_Swings')]
    if target_cols:
        leave_encoder.fit(data['Country'], data[target_cols[0]])
        country_encoded = leave_encoder.transform(data['Country'])
        data['Country'] = country_encoded
        # Build mapping for country -> encoded value (use first occurrence)
        country_map = dict(zip(data['Country'].index, data['Country']))
    else:
        # fallback: use mean encoding with counts
        codes, uniques = pd.factorize(data['Country'])
        data['Country'] = codes
        country_map = dict(zip(uniques, range(len(uniques))))
except Exception as e:
    print('LeaveOneOut fit failed:', e)
    codes, uniques = pd.factorize(data['Country'])
    data['Country'] = codes
    country_map = dict(zip(uniques, range(len(uniques))))

# Build X as in training: drop last three Mood_Swings columns if present
if any(c.startswith('Mood_Swings') for c in data.columns):
    # assume last 3 are the mood columns
    y_cols = [c for c in data.columns if c.startswith('Mood_Swings')]
    X = data.drop(columns=y_cols)
else:
    X = data

# Save columns and encoders
meta = {
    'columns': X.columns.tolist(),
    'label_encoders': {k: v for k, v in label_encoders.items()},
    'country_map': country_map
}
joblib.dump(meta, 'mental/model_meta.joblib')
print('Saved model_meta with', len(meta['columns']), 'columns to mental/model_meta.joblib')
