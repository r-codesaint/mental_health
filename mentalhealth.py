import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC, SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from category_encoders import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("Mental Health Dataset.csv")
df.head()

#print("✅ mentalhealth.py ran successfully!")
df.shape
df.info()
df.drop_duplicates(inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["self_employed"] = le.fit_transform(df["self_employed"])
df["family_history"] = le.fit_transform(df["family_history"])
df["treatment"] = le.fit_transform(df["treatment"])
df["Coping_Struggles"] = le.fit_transform(df["Coping_Struggles"])
data = pd.get_dummies(data=df, columns=["Occupation", "Days_Indoors", "Growing_Stress",
            "Changes_Habits", "Mental_Health_History", "Work_Interest", "Social_Weakness",
            "mental_health_interview", "care_options"])

data = pd.get_dummies(data=data, columns=["Mood_Swings"])
data.drop("Timestamp", axis=1, inplace=True)
data

leave_encoder = LeaveOneOutEncoder()
data["Country"] = leave_encoder.fit_transform(data["Country"], data.iloc[:, -3])
data.info()
cmap = sns.diverging_palette(275,150,  s=40, l=65, n=9)
corrmat = data.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,cmap= cmap,annot=True, square=True);
#plt.show()
y = data.iloc[:, -3:]
X = data.drop(data.iloc[:, -3:], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
X.shape
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
model_list = [LinearSVC(), LogisticRegression(), GradientBoostingClassifier(),
              AdaBoostClassifier(), HistGradientBoostingClassifier()]

roc_auc_list = []
accuracy_list = []

for model in model_list: 
    multi_class = MultiOutputClassifier(model, n_jobs=-1)
    for i in data.iloc[:, -3:]:
        multi_class.fit(X_train, y_train[[i]])
        y_pred = multi_class.predict(X_test)
        roc_auc = roc_auc_score(y_test[i], y_pred)
        accuracy = accuracy_score(y_test[[i]], y_pred)
        
        roc_auc_list.append(roc_auc)
        accuracy_list.append(accuracy)
accuracy_df = pd.DataFrame(accuracy_list, columns=["value"])
roc_auc_df = pd.DataFrame(roc_auc_list, columns=["value"])
list_models = ["LinearSVC", "LogisticRegression", "GradientBoostingClassifier",
               "AdaBoostClassifier", "HistGradientBoostingClassifier"]        
accuracy_averages = [accuracy_df.value[:3].mean(), accuracy_df.value[3:6].mean(), accuracy_df.value[6:9].mean(),
                    accuracy_df.value[9:12].mean(), accuracy_df.value[12:15].mean()]

roc_auc_averages = [roc_auc_df.value[:3].mean(), roc_auc_df.value[3:6].mean(), roc_auc_df.value[6:9].mean(),
                    roc_auc_df.value[9:12].mean(), roc_auc_df.value[12:15].mean()]

average_df = pd.DataFrame(accuracy_averages, columns=["accuracy_mean"], index=list_models)
average_df["roc_auc_mean"] = roc_auc_averages
print(average_df.head())
roc_auc_list = []
accuracy_list = []

hist_class = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1)
print("HistGradientBoostingClassifier SELECTED MODEL\n")

for i in data.iloc[:,-3:].columns:
    hist_class.fit(X_train, y_train[[i]])
    y_pred = hist_class.predict(X_test)
    roc_auc = roc_auc_score(y_test[[i]], y_pred)
    accuracy = accuracy_score(y_test[[i]], y_pred)
    
    roc_auc_list.append(roc_auc)
    accuracy_list.append(accuracy)
            
    print(f'Category name: {i}')
    print(f'{i} AUC ROC score is: {roc_auc:.3f}')
    print(f"accuracy score is: {accuracy:.3f}")
    print("\n", "-" * 50)
    k = 1
l = 2
i = 0
j = 3
for model_name in list_models:
    
    print(f"Accuracy scores of categories with {model_name}: {accuracy_df.value[i:j].mean()}")
    
    print(f"Roc_auc_scores of categories with {model_name}: {roc_auc_df.value[i:j].mean()}")
    
    plt.tight_layout()
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

model_list = [LinearSVC(), LogisticRegression(), GradientBoostingClassifier(),
              AdaBoostClassifier(), HistGradientBoostingClassifier()]

roc_auc_list = []
accuracy_list = []

for model in model_list: 
    multi_class = MultiOutputClassifier(model, n_jobs=-1)
    for i in data.iloc[:, -3:]:
        multi_class.fit(X_train, y_train[[i]])
        y_pred = multi_class.predict(X_test)
        roc_auc = roc_auc_score(y_test[i], y_pred)
        accuracy = accuracy_score(y_test[[i]], y_pred)
        
        roc_auc_list.append(roc_auc)
        accuracy_list.append(accuracy)
accuracy_df = pd.DataFrame(accuracy_list, columns=["value"])
roc_auc_df = pd.DataFrame(roc_auc_list, columns=["value"])
list_models = ["LinearSVC", "LogisticRegression", "GradientBoostingClassifier",
               "AdaBoostClassifier", "HistGradientBoostingClassifier"]
accuracy_averages = [accuracy_df.value[:3].mean(), accuracy_df.value[3:6].mean(), accuracy_df.value[6:9].mean(),
                    accuracy_df.value[9:12].mean(), accuracy_df.value[12:15].mean()]

roc_auc_averages = [roc_auc_df.value[:3].mean(), roc_auc_df.value[3:6].mean(), roc_auc_df.value[6:9].mean(),
                    roc_auc_df.value[9:12].mean(), roc_auc_df.value[12:15].mean()]

avarage_df = pd.DataFrame(accuracy_averages, columns=["accuracy_mean"], index=list_models)
avarage_df["roc_auc_mean"] = roc_auc_averages
print(avarage_df.head())
hist_high = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1).fit(X_train, y_train[["Mood_Swings_High"]])
hist_low = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1).fit(X_train, y_train[["Mood_Swings_Low"]])
hist_medium = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1).fit(X_train, y_train[["Mood_Swings_Medium"]])

pred_high = hist_high.predict(X_test)
pred_low = hist_low.predict(X_test)
pred_medium = hist_medium.predict(X_test)
# Mood Swings (High)
print(classification_report(y_test["Mood_Swings_High"], pred_high))

cm_high = confusion_matrix(y_test["Mood_Swings_High"], pred_high)
cm_low = confusion_matrix(y_test["Mood_Swings_Low"], pred_low)
cm_medium = confusion_matrix(y_test["Mood_Swings_Medium"], pred_medium)
plt.figure(figsize=(12,3.5))
sns.set(font_scale=0.8)
plt.subplot(1,3,1)
sns.heatmap(cm_high, annot=True, fmt='2g', cmap="Reds")
plt.title("Mood Swings (High)", fontsize=15)
plt.subplot(1,3,2)
sns.heatmap(cm_low, annot=True, fmt='2g', cmap="Greens")
plt.title("Mood Swings (Low)", fontsize=15)
plt.subplot(1,3,3)
sns.heatmap(cm_medium, annot=True, fmt='2g', cmap="Blues")
plt.title("Mood Swings (Medium)", fontsize=15)
plt.tight_layout()
plt.show()
y_prob_high = hist_high.predict_proba(X_test)[0][:, 1]

# Save the trained model
import joblib
model_path = 'mental/trained_model.joblib'
joblib.dump(hist_high, model_path)
print(f"Model saved to {model_path}")
y_prob_low = hist_low.predict_proba(X_test)[0][:, 1]
y_prob_medium = hist_medium.predict_proba(X_test)[0][:, 1]

fpr_h, tpr_h, thresholds_h = roc_curve(y_test["Mood_Swings_High"], y_prob_high)
fpr_l, tpr_l, thresholds_l = roc_curve(y_test["Mood_Swings_Low"], y_prob_low)
fpr_m, tpr_m, thresholds_m = roc_curve(y_test["Mood_Swings_Medium"], y_prob_medium)

roc_auc_h = auc(fpr_h, tpr_h)
roc_auc_l = auc(fpr_l, tpr_l)
roc_auc_m = auc(fpr_m, tpr_m)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(fpr_h, tpr_h, color="red", label= f"AUC: {roc_auc_h:.3f}")
plt.plot([0, 1], [0, 1], color="r", linestyle="--", linewidth=1.4)
plt.legend(fontsize=14)
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.title("ROC Curve (Mood Swings High)")
plt.subplot(1,3,2)
plt.plot(fpr_l, tpr_l, color="Green", label= f"AUC: {roc_auc_l:.3f}")
plt.plot([0, 1], [0, 1], color="r", linestyle="--", linewidth=1.4)
plt.legend(fontsize=14)
plt.xlabel("False positive rate (fpr)")
plt.title("ROC Curve (Mood Swings Low)")
plt.subplot(1,3,3)
plt.plot(fpr_m, tpr_m, color="darkblue", label= f"AUC: {roc_auc_m:.3f}")
plt.plot([0, 1], [0, 1], color="r", linestyle="--", linewidth=1.4)
plt.legend(fontsize=14)
plt.xlabel("False positive rate (fpr)")
plt.title("ROC Curve (Mood Swings Medium)")
plt.show()
data.shape
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Early Stopping
early_stopping = EarlyStopping(
    min_delta=0.001,  
    patience=20,  
    restore_best_weights=True
)

# Model Architecture
model = Sequential()
model.add(Dense(37, activation='relu', input_dim=X_train.shape[1]))  # Increase neurons
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))

# Output Layer
model.add(Dense(3, activation='sigmoid'))  # Use 'softmax' if labels are exclusive

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),  
              loss='binary_crossentropy',  # Use 'categorical_crossentropy' for exclusive labels
              metrics=['accuracy'])
# Train Model
history = model.fit(
    X_train_scaled, y_train,  
    batch_size=64, epochs=500,  
    callbacks=[early_stopping],  
    validation_split=0.2,
    class_weight={0: 1, 1: 2, 2: 2}  # Adjust if labels are imbalanced
)
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
from keras.optimizers import SGD
import tensorflow as tf

# Use SGD with momentum for stable updates
optimizer = SGD(learning_rate=0.01, momentum=0.9)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Sequential([
    Dense(37, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(3, activation='sigmoid')  # Use 'softmax' if labels are exclusive
])
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Custom Training Loop
epochs = 20
batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(batch_size)

for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    for X_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            loss = loss_fn(y_batch, y_pred)  # Compute loss
        
        gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update weights
        epoch_loss += loss.numpy()

        # Compute Accuracy (Fix Type Mismatch)
        y_pred_labels = tf.round(y_pred)  # Convert probabilities to 0 or 1
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(y_pred_labels, tf.cast(y_batch, tf.float32)), tf.float32)).numpy()
        total_samples += tf.size(y_batch).numpy()  # Total number of labels

    epoch_accuracy = correct_predictions / total_samples  # Multi-label accuracy
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")