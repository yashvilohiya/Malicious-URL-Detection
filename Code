# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 2. LOAD DATASET (DIRECT PATH)
# ==============================
# Make sure the file is already in your Colab environment

df = pd.read_csv('/content/balanced_urls.csv')

print(df.head())
print(df.columns)

# ==============================
# 3. DATA PREPROCESSING
# ==============================
df = df[['url', 'label']]
df.dropna(inplace=True)

# Convert labels to numeric
df['label'] = df['label'].map({'benign': 0, 'malicious': 1})

# ==============================
# 4. FEATURE EXTRACTION
# ==============================
def extract_features(url):
    features = []
    
    # URL length
    features.append(len(url))
    
    # Number of dots
    features.append(url.count('.'))
    
    # Presence of '@'
    features.append(1 if '@' in url else 0)
    
    # Number of '-'
    features.append(url.count('-'))
    
    # Presence of '//'
    features.append(1 if '//' in url else 0)
    
    # Number of digits
    features.append(sum(c.isdigit() for c in url))
    
    # HTTPS usage
    features.append(1 if url.startswith('https') else 0)
    
    return features

# Apply feature extraction
X = np.array(df['url'].apply(extract_features).tolist())
y = df['label']

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. MODEL TRAINING
# ==============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 7. MODEL EVALUATION
# ==============================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 8. PREDICTION FUNCTION
# ==============================
def predict_url(url):
    features = np.array(extract_features(url)).reshape(1, -1)
    prediction = model.predict(features)[0]
    
    return "Malicious 🚨" if prediction == 1 else "Benign ✅"

# Example predictions
print(predict_url("https://google.com"))
print(predict_url("http://login-secure-paypal.com@malicious.ru"))
