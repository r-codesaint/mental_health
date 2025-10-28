import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load dataset
df = pd.read_csv("Mental Health Dataset.csv")

# Inspect columns
print("Columns in dataset:", df.columns)

# ✅ Replace with your actual target column name
X = df.drop("Mental_Condition", axis=1)
y = df["Mental_Condition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build ANN
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save model and scaler
model.save("mindwell_ann_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")
