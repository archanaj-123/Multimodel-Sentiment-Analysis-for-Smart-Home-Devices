import os
import pandas as pd
import random

# --- Paths ---
BASE_DIR = r"C:\Users\Dell\Downloads\Multimodal sentiment analysis\Datasets"
TEXT_CSV = os.path.join(BASE_DIR, "text", "combined_emotion.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset.csv")

# --- Read the text CSV ---
text_df = pd.read_csv(TEXT_CSV)
text_df = text_df.dropna(subset=["sentence", "emotion"])  # remove NaNs
text_df["emotion"] = text_df["emotion"].str.strip().str.capitalize()

# --- Initialize list for combined dataset ---
dataset = []

# --- Labels from audio folder ---
labels = [f for f in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, f))]
labels = [lbl.capitalize() for lbl in labels]

print("Found labels:", labels)

# --- Combine matching audio, image, and text by emotion ---
for label in labels:
    audio_path = os.path.join(AUDIO_DIR, label)
    
    # Image paths can be in train or test subfolders
    image_train_path = os.path.join(IMAGE_DIR, "train", label)
    image_test_path = os.path.join(IMAGE_DIR, "test", label)
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Skipping {label} — audio folder missing")
        continue
    
    # Collect all images from train and test
    image_files = []
    for path in [image_train_path, image_test_path]:
        if os.path.exists(path):
            image_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    
    if not image_files:
        print(f"⚠️ No image files found for {label}, skipping.")
        continue

    audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.lower().endswith(".wav")]
    if not audio_files:
        print(f"⚠️ No audio files found for {label}, skipping.")
        continue

    min_len = min(len(audio_files), len(image_files))
    audio_files = random.sample(audio_files, min_len)
    image_files = random.sample(image_files, min_len)

    # Filter text for this label
    text_subset = text_df[text_df["emotion"].str.lower() == label.lower()]
    text_sentences = text_subset["sentence"].tolist()

    # Repeat text if not enough
    if len(text_sentences) < min_len and len(text_sentences) > 0:
        text_sentences = (text_sentences * ((min_len // len(text_sentences)) + 1))[:min_len]
    elif len(text_sentences) == 0:
        text_sentences = ["No text available"] * min_len
    else:
        text_sentences = random.sample(text_sentences, min_len)

    for a_file, i_file, t_sent in zip(audio_files, image_files, text_sentences):
        dataset.append({
            "audio": a_file,
            "image": i_file,
            "text": t_sent,
            "label": label
        })

# --- Save dataset CSV ---
df_dataset = pd.DataFrame(dataset)
df_dataset.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"✅ Dataset CSV created with {len(df_dataset)} rows at: {OUTPUT_CSV}")
