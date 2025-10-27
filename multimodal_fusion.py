import os
import re
import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# 1️⃣  Load Dataset
# ------------------------------------------------------------
DATASET_PATH = r"C:\Users\Dell\Downloads\Multimodal sentiment analysis\Datasets\text\combined_emotion.csv"

df = pd.read_csv(DATASET_PATH)
print("Columns in dataset:")
print(df.columns)
print(df.head())

# Handle missing data
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Ensure correct column names
text_col = "sentence" if "sentence" in df.columns else df.columns[0]
label_col = "emotion" if "emotion" in df.columns else df.columns[1]

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|[^A-Za-z0-9 ]+", "", str(text).lower())
    return text

df["clean_text"] = df[text_col].apply(clean_text)

print(f"Dataset shape: {df.shape}")
print(df[label_col].value_counts())

# ------------------------------------------------------------
# 2️⃣  Train Text Model
# ------------------------------------------------------------
X = df["clean_text"]
y = df[label_col]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Training RandomForest text classifier...")
text_clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
text_clf.fit(X_train, y_train)

y_pred = text_clf.predict(X_test)
print("\n--- Text Model Evaluation ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(text_clf, "models/text_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("✅ Text model and vectorizer saved.")

# ------------------------------------------------------------
# 3️⃣  Define Feature Extraction for Audio & Image
# ------------------------------------------------------------
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Audio feature extraction failed for {file_path}: {e}")
        return np.zeros(40)

def extract_image_features(file_path, image_model):
    try:
        img = image.load_img(file_path, target_size=(48, 48), color_mode="grayscale")
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = image_model.predict(img_array)
        return preds.flatten()
    except Exception as e:
        print(f"Image feature extraction failed for {file_path}: {e}")
        return np.zeros(7)

# ------------------------------------------------------------
# 4️⃣  Fusion Function
# ------------------------------------------------------------
def fuse_predictions(audio_feat, image_feat, text_feat, audio_model, weights=[0.3, 0.4, 0.3]):
    try:
        audio_pred = audio_model.predict_proba([audio_feat])[0]
        min_len = min(len(audio_pred), len(image_feat), len(text_feat))
        fused = (
            audio_pred[:min_len] * weights[0]
            + image_feat[:min_len] * weights[1]
            + text_feat[:min_len] * weights[2]
        )
        return fused
    except Exception as e:
        print(f"Fusion failed: {e}")
        return np.zeros(7)

# ------------------------------------------------------------
# 5️⃣  Mood Prediction Example
# ------------------------------------------------------------
def predict_mood(audio_path, image_path, text_input, image_model, audio_model):
    from tensorflow.keras.models import load_model

    mood_labels = ["joy", "sad", "anger", "fear", "love", "surprise"]

    # Extract features
    audio_feat = extract_audio_features(audio_path)
    image_feat = extract_image_features(image_path, image_model)
    clean_text = re.sub(r"http\S+|[^A-Za-z0-9 ]+", "", str(text_input).lower())
    text_feat = text_clf.predict_proba(vectorizer.transform([clean_text]))[0]

    fused_pred = fuse_predictions(audio_feat, image_feat, text_feat, audio_model)
    mood_index = np.argmax(fused_pred)
    mood = mood_labels[mood_index]

    actions = {
        "joy": "Play upbeat music, brighten lights 🌞",
        "sad": "Dim lights, play soft lo-fi 🌧️",
        "anger": "Play calming piano, cool light tone 💧",
        "fear": "Use warm light, play ambient sounds 🌙",
        "love": "Play romantic songs, pinkish hue ❤️",
        "surprise": "Keep lights vibrant 🌈"
    }

    return mood, actions.get(mood, "Maintain default ambience 🌤️")

# ------------------------------------------------------------
# 6️⃣  Example Run
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Example Multimodal Fusion Run ---")
    
    # Load existing models (replace paths if needed)
    try:
        image_model = load_model("models/image_model.h5")
        audio_model = joblib.load("models/audio_model.pkl")
    except:
        print("⚠️ Some models not found. Fusion will use text model only.")
        image_model = None
        audio_model = None

    mood, action = predict_mood(
        "dataset/audio/happy_01.wav",
        "dataset/images/happy_01.jpg",
        "I feel so great today!",
        image_model,
        audio_model
    )

    print("Predicted Mood:", mood)
    print("Recommended Action:", action)
