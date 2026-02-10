import joblib
import streamlit as st
import pandas as pd

# ----------------------------
# Load model + artifacts
# ----------------------------
model = joblib.load("final_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")
rare_publishers = set(joblib.load("rare_publishers.pkl"))

# ----------------------------
# Load dataset ONLY for dropdown lists
# ----------------------------
df = pd.read_csv("vgsales.csv")

platform_list = sorted(df["Platform"].dropna().unique())
genre_list = sorted(df["Genre"].dropna().unique())
publisher_list = sorted(df["Publisher"].dropna().unique())

# ----------------------------
# Platform display mapping (UI-friendly)
# ----------------------------
platform_display_map = {
    "2600": "Atari 2600",
    "3DO": "3DO Interactive Multiplayer",
    "DC": "Sega Dreamcast",
    "GB": "Nintendo Game Boy",
    "GBA": "Nintendo Game Boy Advance",
    "GC": "Nintendo GameCube",
    "GEN": "Sega Genesis",
    "GG": "Sega Game Gear",
    "N64": "Nintendo 64",
    "NES": "Nintendo Entertainment System (NES)",
    "NG": "Neo Geo",
    "PCFX": "PC-FX",
    "PS": "PlayStation",
    "PS2": "PlayStation 2",
    "PS3": "PlayStation 3",
    "PS4": "PlayStation 4",
    "PSP": "PlayStation Portable (PSP)",
    "PSV": "PlayStation Vita",
    "SAT": "Sega Saturn",
    "SCD": "Sega CD",
    "SNES": "Super Nintendo (SNES)",
    "TG16": "TurboGrafx-16",
    "WS": "WonderSwan",
    "Wii": "Nintendo Wii",
    "WiiU": "Nintendo Wii U",
    "XB": "Xbox",
    "X360": "Xbox 360",
    "XOne": "Xbox One"
}

platform_labels = [platform_display_map.get(p, p) for p in platform_list]
label_to_code = {platform_display_map.get(p, p): p for p in platform_list}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎮 Video Game Sales Prediction")

st.write(
    "Predict whether a video game will achieve **High Sales (≥ 1 million units)** "
    "based on its release year, platform, genre, and publisher."
)

year = st.slider("Release Year", 1980, 2020, 2010)

platform_label = st.selectbox(
    "Platform",
    platform_labels,
    index=platform_labels.index("PlayStation 2") if "PlayStation 2" in platform_labels else 0
)
platform = label_to_code[platform_label]

genre = st.selectbox("Genre", genre_list)
publisher = st.selectbox("Publisher", publisher_list)

# ----------------------------
# Apply feature engineering (Publisher Grouping)
# ----------------------------
publisher_grouped = "Other" if publisher in rare_publishers else publisher

# ----------------------------
# Build input DataFrame
# ----------------------------
input_df = pd.DataFrame([{
    "Year": year,
    "Platform": platform,
    "Genre": genre,
    "Publisher_Grouped": publisher_grouped
}])

# One-hot encode like training
input_df = pd.get_dummies(
    input_df,
    columns=["Platform", "Genre", "Publisher_Grouped"]
)

# Align columns to training data
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict High Sales"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f" Prediction: HIGH SALES (≥ 1M units)\n\nConfidence: {probability:.2%}")
    else:
        st.info(f" Prediction: LOW SALES (< 1M units)\n\nConfidence: {probability:.2%}")
