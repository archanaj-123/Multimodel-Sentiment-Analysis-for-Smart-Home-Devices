import streamlit as st
from multimodal_fusion import predict_mood
import os
import random
from datetime import datetime

st.set_page_config(
    page_title="Smart Home Mood Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar instructions ---
st.sidebar.header("📘 How to Use")
st.sidebar.write("""
1. Greet the assistant by typing how you feel today 💬  
2. Upload your **audio (.wav)** and **image (.jpg/.png)** files.  
3. Click **🎯 Predict Mood**.  
4. Watch your smart home react automatically with colors and music!
""")

# --- Greeting based on time ---
hour = datetime.now().hour
if hour < 12:
    greeting = "🌞 Good morning! How was your night?"
elif hour < 18:
    greeting = "🌼 Good afternoon! How’s your day going?"
else:
    greeting = "🌙 Good evening! How are you feeling today?"

st.title("🏠 Smart Home Mood Assistant")
st.subheader(greeting)

# --- User message ---
user_message = st.text_input("💬 Tell me how you feel today:", "")

# --- Upload section ---
st.markdown("### 📤 Upload your Inputs")
col1, col2 = st.columns([1, 1])
with col1:
    audio_file = st.file_uploader("🎵 Upload Audio (.wav)", type=["wav"])
with col2:
    image_file = st.file_uploader("🖼️ Upload Image (.jpg/.png)", type=["jpg", "png"])

# --- Mood colors, fonts, and icons ---
mood_themes = {
    "Happy": {"bg": "#FFF7D1", "text": "#3E3E1F", "icon": "assets/happy.png"},
    "Sad": {"bg": "#E8D6CB", "text": "#3B2E2E", "icon": "assets/sad.png"},
    "Neutral": {"bg": "#D6F0F0", "text": "#2E4444", "icon": "assets/neutral.png"},
    "Angry": {"bg": "#F5C2C2", "text": "#4A1212", "icon": "assets/angry.png"}
}

music_files = {
    "Happy": "assets/happy.mp3",
    "Sad": "assets/sad.mp3",
    "Neutral": "assets/neutral.mp3",
    "Angry": "assets/angry.mp3"
}

# --- Theme styling ---
def set_theme(mood):
    theme = mood_themes.get(mood, {"bg": "#FFFFFF", "text": "#000000"})
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
            transition: all 0.8s ease-in-out;
        }}
        h1, h2, h3, h4, h5, h6, p, div, span {{
            color: {theme['text']} !important;
            transition: color 0.8s ease-in-out;
            font-family: 'Poppins', sans-serif;
        }}
        .stButton>button {{
            background: linear-gradient(45deg, #B4E7B0, #9AD0EC);
            color: #2C2C2C;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            transition: 0.3s ease;
        }}
        .stButton>button:hover {{
            background: linear-gradient(45deg, #A2D2FF, #B5EAEA);
            transform: scale(1.03);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Predict button ---
if st.button("🎯 Predict Mood"):
    if audio_file is None or image_file is None:
        st.warning("⚠️ Please upload both audio and image files.")
    else:
        # Save temporary files
        audio_path = "temp_audio.wav"
        image_path = "temp_image.png"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        # Predict mood
        mood, action = predict_mood(audio_path, image_path, user_message)

        # Apply theme
        set_theme(mood)

        # Display results
        st.markdown(f"## 🧠 Detected Mood: **{mood}**")
        st.write(f"### ✨ Smart Action: {action}")

        # Display mood image
        icon_path = mood_themes[mood]["icon"]
        if os.path.exists(icon_path):
            st.image(icon_path, width=180)

        # Automatically play mood music
        music_path = music_files.get(mood)
        if music_path and os.path.exists(music_path):
            st.audio(music_path, format="audio/mp3", start_time=0)
        else:
            st.warning(f"🎵 No music available for mood '{mood}'")

# --- Interactive emotional response ---
responses = {
    "Happy": ["Yay! I'm so glad you're smiling! 😄", "Keep that sunshine energy going! 🌻", "Time to celebrate your good vibes! 🎉"],
    "Sad": ["It’s okay to feel low sometimes. 🌧️", "Sending you a virtual hug 💛", "Let’s play something soothing for you 🎵"],
    "Neutral": ["A calm state is powerful. 🌿", "Let’s keep things peaceful today. 🕊️", "Everything feels balanced — nice job! ⚖️"],
    "Angry": ["Deep breaths, you’ve got this 💨", "Let’s calm things down with gentle music 🎶", "Don’t let it stay long — breathe out 🔥"]
}

if user_message:
    random_mood = random.choice(list(mood_themes.keys()))
    st.info(random.choice(responses[random_mood]))
