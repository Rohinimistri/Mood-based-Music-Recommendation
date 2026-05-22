"""
🎧 Mood-Based Music Recommendation System
==========================================
A Streamlit app that recommends songs based on user mood using
content-based filtering with cosine similarity.

Tech Stack: Python, Pandas, Scikit-learn, Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import time
import urllib.parse

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mood Music Recommender 🎧",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — warm vinyl / record-shop aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --cream:   #F5EFE0;
    --warm:    #E8D5B0;
    --amber:   #C8873A;
    --burnt:   #A0522D;
    --charcoal:#1A1410;
    --carbon:  #2C2418;
    --groove:  #3D2F1E;
    --text:    #1A1410;
    --muted:   #6B5A42;
    --white:   #FDFAF4;
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--carbon) !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1100px; }

/* ── Hero Header ── */
.hero-wrap {
    background: var(--charcoal);
    border-radius: 24px;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    border: 3px solid rgba(200,135,58,0.15);
    box-shadow: 0 0 0 60px rgba(200,135,58,0.04),
                0 0 0 120px rgba(200,135,58,0.02);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 900;
    color: var(--cream);
    line-height: 1.05;
    margin: 0 0 0.6rem;
    letter-spacing: -1px;
}
.hero-title span { color: var(--amber); }
.hero-sub {
    font-size: 1.05rem;
    color: var(--warm);
    font-weight: 300;
    letter-spacing: 0.5px;
    margin: 0;
}
.vinyl-badge {
    position: absolute;
    top: 1.8rem; right: 2.2rem;
    font-size: 5rem;
    opacity: 0.18;
    transform: rotate(-15deg);
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.5rem;
}

/* ── Mood Card Grid ── */
.mood-grid { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.5rem 0 1.5rem; }
.mood-pill {
    padding: 0.55rem 1.1rem;
    border-radius: 50px;
    border: 2px solid var(--warm);
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--groove);
    font-weight: 500;
}
.mood-pill:hover, .mood-pill.active {
    background: var(--charcoal);
    border-color: var(--amber);
    color: var(--cream);
}

/* ── Control Panel ── */
.control-panel {
    background: var(--white);
    border: 2px solid var(--warm);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 2rem;
}

/* ── Streamlit widget overrides ── */
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border: 2px solid var(--warm) !important;
    background: var(--white) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
div[data-baseweb="select"] > div:hover { border-color: var(--amber) !important; }

[data-testid="stSlider"] > div > div > div {
    background: var(--amber) !important;
}
.stSlider [data-testid="stTickBar"] { color: var(--muted); }

/* ── Recommend button ── */
.stButton > button {
    background: var(--charcoal) !important;
    color: var(--cream) !important;
    border: 2px solid var(--amber) !important;
    border-radius: 14px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--amber) !important;
    border-color: var(--amber) !important;
    color: var(--charcoal) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(200,135,58,0.3) !important;
}

/* ── Song Card ── */
.song-card {
    background: var(--black);
    border: 2px solid var(--warm);
    border-radius: 18px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.song-card:hover {
    border-color: var(--amber);
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(160,82,45,0.1);
}
.song-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--amber);
    border-radius: 4px 0 0 4px;
    opacity: 0;
    transition: opacity 0.2s;
}
.song-card:hover::before { opacity: 1; }

.track-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--warm);
    min-width: 2.2rem;
    text-align: center;
}
.track-info { flex: 1; }
.track-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.02rem;
    font-weight: 500;
    color: var(--charcoal);
    margin: 0 0 0.15rem;
}
.track-artist {
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0;
}
.score-badge {
    background: var(--charcoal);
    color: var(--amber);
    border-radius: 8px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 500;
    white-space: nowrap;
}
.link-btn {
    text-decoration: none !important;
    background: var(--warm);
    color: var(--groove) !important;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    transition: all 0.18s;
    white-space: nowrap;
}
.link-btn:hover {
    background: var(--amber);
    color: var(--white) !important;
}

/* ── Feature bars ── */
.feat-row { margin-bottom: 0.4rem; }
.feat-label { font-size: 0.75rem; color: var(--muted); margin-bottom: 2px; }
.feat-bar-bg {
    background: var(--warm);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--burnt), var(--amber));
}

/* ── Mood profile card ── */
.mood-profile {
    background: var(--charcoal);
    border-radius: 18px;
    padding: 1.8rem;
    color: var(--cream);
    height: 100%;
}
.mood-profile h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    margin: 0 0 1rem;
    color: var(--amber);
}

/* ── Divider ── */
.groove-divider {
    border: none;
    border-top: 2px solid var(--warm);
    margin: 2rem 0;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0 0;
}
.stat-box {
    background: var(--white);
    border: 2px solid var(--warm);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    flex: 1;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--amber);
}
.stat-desc { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* ── Keyword chip ── */
.kw-chip {
    display: inline-block;
    background: var(--warm);
    color: var(--groove);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.15rem;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: var(--amber) !important; }

/* text area / input */
/* ✅ FIX placeholder + typing visibility */
div[data-baseweb="input"] input {
    color: #1A1410 !important;
    -webkit-text-fill-color: #1A1410 !important;
    caret-color: #1A1410 !important;
}

div[data-baseweb="input"] input::placeholder {
    color: #6B5A42 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 1. DATASET — generate rich dummy dataset
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Generate a rich dummy Spotify-like dataset with realistic
    audio features. Falls back to this if a real CSV isn't found.
    """
    np.random.seed(42)

    # ── Curated song pool (track_name, artist, energy, danceability, valence, tempo)
    songs = [
        # Happy / Uplifting
        ("Happy",                    "Pharrell Williams",    0.85, 0.80, 0.96, 160.0),
        ("Can't Stop the Feeling!",  "Justin Timberlake",    0.88, 0.91, 0.94, 113.0),
        ("Shake It Off",             "Taylor Swift",         0.80, 0.90, 0.93, 160.0),
        ("Uptown Funk",              "Mark Ronson",          0.91, 0.89, 0.88, 115.0),
        ("Dancing Queen",            "ABBA",                 0.79, 0.86, 0.90, 101.0),
        ("Walking on Sunshine",      "Katrina & the Waves",  0.87, 0.83, 0.92, 145.0),
        ("I Gotta Feeling",          "Black Eyed Peas",      0.84, 0.87, 0.91, 128.0),
        ("Good as Hell",             "Lizzo",                0.76, 0.85, 0.89, 96.0),
        ("Levitating",               "Dua Lipa",             0.82, 0.84, 0.86, 103.0),
        ("Dynamite",                 "BTS",                  0.76, 0.87, 0.90, 114.0),

        # Sad / Melancholic
        ("Someone Like You",         "Adele",                0.22, 0.30, 0.08, 68.0),
        ("The Night We Met",         "Lord Huron",           0.31, 0.33, 0.07, 89.0),
        ("Skinny Love",              "Bon Iver",             0.25, 0.27, 0.10, 94.0),
        ("Fix You",                  "Coldplay",             0.40, 0.31, 0.15, 138.0),
        ("All I Want",               "Kodaline",             0.21, 0.29, 0.09, 70.0),
        ("Hurt",                     "Nine Inch Nails",      0.38, 0.23, 0.06, 54.0),
        ("The Sound of Silence",     "Simon & Garfunkel",    0.20, 0.26, 0.11, 104.0),
        ("Falling",                  "Harry Styles",         0.28, 0.35, 0.13, 94.0),
        ("Let Her Go",               "Passenger",            0.32, 0.37, 0.12, 76.0),
        ("Yesterday",                "The Beatles",          0.24, 0.28, 0.14, 97.0),

        # Energetic / Workout
        ("Eye of the Tiger",         "Survivor",             0.95, 0.71, 0.72, 109.0),
        ("Thunderstruck",            "AC/DC",                0.98, 0.60, 0.65, 133.0),
        ("Till I Collapse",          "Eminem",               0.96, 0.72, 0.58, 171.0),
        ("Lose Yourself",            "Eminem",               0.93, 0.69, 0.55, 171.0),
        ("Jump Around",              "House of Pain",        0.92, 0.81, 0.70, 165.0),
        ("Run This Town",            "Jay-Z",                0.90, 0.77, 0.62, 130.0),
        ("Blinding Lights",          "The Weeknd",           0.73, 0.51, 0.33, 171.0),
        ("Stronger",                 "Kanye West",           0.87, 0.79, 0.67, 104.0),
        ("Power",                    "Kanye West",           0.86, 0.68, 0.48, 90.0),
        ("Rage Against the Machine", "Killing in the Name",  0.97, 0.62, 0.50, 126.0),

        # Chill / Relaxed
        ("Chill Bill",               "Rob $tone",            0.42, 0.72, 0.60, 75.0),
        ("Sunset Lover",             "Petit Biscuit",        0.45, 0.68, 0.65, 100.0),
        ("Redbone",                  "Childish Gambino",     0.48, 0.64, 0.55, 80.0),
        ("Electric Feel",            "MGMT",                 0.55, 0.73, 0.62, 107.0),
        ("Lo-Fi Hip Hop",            "ChilledCow",           0.35, 0.62, 0.58, 85.0),
        ("Sunday Morning",           "Maroon 5",             0.44, 0.67, 0.70, 91.0),
        ("Dreams",                   "Fleetwood Mac",        0.50, 0.66, 0.74, 120.0),
        ("Banana Pancakes",          "Jack Johnson",         0.38, 0.60, 0.80, 120.0),
        ("Better Together",          "Jack Johnson",         0.41, 0.63, 0.77, 107.0),
        ("Harvest Moon",             "Neil Young",           0.36, 0.58, 0.75, 96.0),

        # Romantic / Love
        ("Perfect",                  "Ed Sheeran",           0.45, 0.50, 0.75, 95.0),
        ("Thinking Out Loud",        "Ed Sheeran",           0.47, 0.55, 0.78, 79.0),
        ("Make You Feel My Love",    "Adele",                0.30, 0.32, 0.60, 70.0),
        ("At Last",                  "Etta James",           0.35, 0.40, 0.72, 65.0),
        ("Your Song",                "Elton John",           0.38, 0.45, 0.80, 138.0),
        ("Unchained Melody",         "The Righteous Brothers",0.28, 0.28, 0.62, 66.0),
        ("A Thousand Years",         "Christina Perri",      0.29, 0.37, 0.55, 95.0),
        ("All of Me",                "John Legend",          0.40, 0.48, 0.70, 120.0),
        ("Can't Help Falling in Love","Elvis Presley",       0.32, 0.35, 0.68, 57.0),
        ("La Vie en Rose",           "Édith Piaf",           0.26, 0.31, 0.65, 61.0),

        # Focus / Study
        ("Intro",                    "The xx",               0.30, 0.45, 0.40, 77.0),
        ("Comptine d'un autre été",  "Yann Tiersen",         0.18, 0.38, 0.48, 80.0),
        ("Experience",               "Ludovico Einaudi",     0.20, 0.35, 0.50, 52.0),
        ("Nuvole Bianche",           "Ludovico Einaudi",     0.17, 0.33, 0.46, 58.0),
        ("Clair de Lune",            "Debussy",              0.14, 0.30, 0.52, 75.0),
        ("River Flows in You",       "Yiruma",               0.15, 0.32, 0.54, 72.0),
        ("Weightless",               "Marconi Union",        0.22, 0.36, 0.45, 60.0),
        ("Brain Power",              "Study Music",          0.28, 0.42, 0.50, 90.0),
        ("Focus Flow",               "Lofi Beats",           0.33, 0.48, 0.55, 88.0),
        ("Deep Work",                "Ambient Sounds",       0.25, 0.40, 0.48, 82.0),

        # Party / Dance
        ("Blurred Lines",            "Robin Thicke",         0.83, 0.92, 0.78, 120.0),
        ("Get Lucky",                "Daft Punk",            0.71, 0.86, 0.82, 116.0),
        ("One Dance",                "Drake",                0.63, 0.79, 0.71, 104.0),
        ("Yeah!",                    "Usher",                0.88, 0.93, 0.75, 128.0),
        ("Party Rock Anthem",        "LMFAO",                0.87, 0.95, 0.77, 130.0),
        ("Turn Down for What",       "DJ Snake",             0.94, 0.88, 0.68, 100.0),
        ("Nights",                   "Frank Ocean",          0.65, 0.74, 0.60, 109.0),
        ("Sicko Mode",               "Travis Scott",         0.82, 0.80, 0.55, 155.0),
        ("Hotline Bling",            "Drake",                0.58, 0.76, 0.64, 135.0),
        ("Work",                     "Rihanna",              0.62, 0.88, 0.70, 92.0),

        # Anger / Dark
        ("Break Stuff",              "Limp Bizkit",          0.97, 0.64, 0.30, 167.0),
        ("Given Up",                 "Linkin Park",          0.96, 0.58, 0.28, 186.0),
        ("In the End",               "Linkin Park",          0.82, 0.62, 0.38, 105.0),
        ("Numb",                     "Linkin Park",          0.78, 0.60, 0.35, 108.0),
        ("Bodies",                   "Drowning Pool",        0.98, 0.55, 0.25, 148.0),
        ("Welcome to the Black Parade","My Chemical Romance",0.88, 0.56, 0.32, 96.0),
        ("Headstrong",               "Trapt",                0.91, 0.62, 0.35, 162.0),
        ("Killing Me Softly",        "Fugees",               0.60, 0.72, 0.52, 101.0),
        ("99 Problems",              "Jay-Z",                0.89, 0.73, 0.42, 140.0),
        ("Killing in the Name",      "RATM",                 0.96, 0.65, 0.38, 126.0),
    ]

    records = []
    for track, artist, energy, dance, valence, tempo in songs:
        # add tiny noise for variety
        records.append({
            "track_name":    track,
            "artist":        artist,
            "energy":        float(np.clip(energy   + np.random.normal(0, 0.02), 0, 1)),
            "danceability":  float(np.clip(dance    + np.random.normal(0, 0.02), 0, 1)),
            "valence":       float(np.clip(valence  + np.random.normal(0, 0.02), 0, 1)),
            "tempo":         float(np.clip(tempo    + np.random.normal(0, 2),   40, 220)),
        })

    df = pd.DataFrame(records)
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────
FEATURE_COLS = ["energy", "danceability", "valence", "tempo"]

@st.cache_data
def preprocess(df: pd.DataFrame):
    """Handle missing values and normalise feature columns."""
    df = df.copy()

    # Drop rows missing any critical field
    df.dropna(subset=["track_name", "artist"], inplace=True)

    # Fill missing numerics with column median
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Normalise to [0, 1]
    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    return df, scaler


# ──────────────────────────────────────────────
# 3. MOOD MAPPING
# ──────────────────────────────────────────────
MOOD_CONFIG = {
    "😊 Happy": {
        "vector":   {"energy": 0.80, "danceability": 0.85, "valence": 0.95, "tempo": 0.65},
        "keywords": ["happy", "joyful", "excited", "cheerful", "elated", "great", "amazing"],
        "color":    "#F4B942",
        "desc":     "Upbeat, joyful tracks to keep your spirits high.",
        "emoji":    "😊",
    },
    "😢 Sad": {
        "vector":   {"energy": 0.20, "danceability": 0.25, "valence": 0.05, "tempo": 0.25},
        "keywords": ["sad", "depressed", "lonely", "heartbroken", "down", "gloomy", "blue"],
        "color":    "#6B8CBE",
        "desc":     "Gentle, melancholic songs for quiet moments.",
        "emoji":    "😢",
    },
    "⚡ Energetic": {
        "vector":   {"energy": 0.97, "danceability": 0.75, "valence": 0.70, "tempo": 0.90},
        "keywords": ["energetic", "pumped", "workout", "gym", "running", "fired up", "hyped"],
        "color":    "#E05C3A",
        "desc":     "High-intensity bangers to fuel your grind.",
        "emoji":    "⚡",
    },
    "😌 Chill": {
        "vector":   {"energy": 0.35, "danceability": 0.60, "valence": 0.65, "tempo": 0.35},
        "keywords": ["chill", "relax", "calm", "mellow", "laid back", "peaceful", "cozy"],
        "color":    "#6BBE8C",
        "desc":     "Smooth, soothing vibes to unwind.",
        "emoji":    "😌",
    },
    "💕 Romantic": {
        "vector":   {"energy": 0.35, "danceability": 0.45, "valence": 0.78, "tempo": 0.35},
        "keywords": ["romantic", "love", "date", "crush", "intimate", "tender", "sweetheart"],
        "color":    "#E07BA0",
        "desc":     "Soft love songs and slow-dance favourites.",
        "emoji":    "💕",
    },
    "🎓 Focus": {
        "vector":   {"energy": 0.20, "danceability": 0.38, "valence": 0.48, "tempo": 0.30},
        "keywords": ["focus", "study", "work", "concentrate", "productivity", "deep work"],
        "color":    "#8C7BE0",
        "desc":     "Minimal, ambient tracks for deep concentration.",
        "emoji":    "🎓",
    },
    "🎉 Party": {
        "vector":   {"energy": 0.88, "danceability": 0.93, "valence": 0.78, "tempo": 0.72},
        "keywords": ["party", "dance", "club", "celebrate", "night out", "groove", "bop"],
        "color":    "#E0B03A",
        "desc":     "Floor-filling anthems for when the night is young.",
        "emoji":    "🎉",
    },
    "😤 Angry": {
        "vector":   {"energy": 0.96, "danceability": 0.60, "valence": 0.20, "tempo": 0.85},
        "keywords": ["angry", "mad", "furious", "rage", "frustrated", "annoyed", "pissed"],
        "color":    "#BE3A3A",
        "desc":     "Raw, intense tracks to let it all out.",
        "emoji":    "😤",
    },
}

MOOD_LABELS = list(MOOD_CONFIG.keys())


# ──────────────────────────────────────────────
# 4. RECOMMENDATION ENGINE
# ──────────────────────────────────────────────
def build_mood_vector(mood_label: str, scaler: MinMaxScaler) -> np.ndarray:
    """
    Convert a mood's raw feature dict into the same normalised space
    the songs live in, using the fitted scaler.
    """
    raw = MOOD_CONFIG[mood_label]["vector"]
    raw_df = pd.DataFrame([[raw[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

    # tempo is between 40–220 in raw; we stored normalised vectors (0-1)
    # so we need to convert tempo back to raw range before scaling
    raw_df["tempo"] = raw_df["tempo"] * (220 - 40) + 40

    scaled = scaler.transform(raw_df)
    return scaled


def recommend(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    moods: list[str],
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Content-based filtering using cosine similarity between
    a blended mood vector and all song feature vectors.
    """
    # Blend multiple mood vectors by averaging
    vectors = [build_mood_vector(m, scaler) for m in moods]
    mood_vec = np.mean(vectors, axis=0)               # shape (1, 4)

    song_matrix = df[FEATURE_COLS].values              # shape (N, 4)
    sims = cosine_similarity(mood_vec, song_matrix)[0] # shape (N,)

    df = df.copy()
    df["similarity"] = sims
    df_sorted = df.sort_values("similarity", ascending=False).head(top_n)
    return df_sorted.reset_index(drop=True)


# ──────────────────────────────────────────────
# 5. KEYWORD MOOD DETECTION
# ──────────────────────────────────────────────
def detect_mood_from_text(text: str) -> str | None:
    """Basic keyword scan — returns best-matching mood or None."""
    text_lower = text.lower()
    scores = {}
    for mood, cfg in MOOD_CONFIG.items():
        hit = sum(1 for kw in cfg["keywords"] if kw in text_lower)
        if hit:
            scores[mood] = hit
    if not scores:
        return None
    return max(scores, key=scores.get)


# ──────────────────────────────────────────────
# 6. HELPER — build song cards HTML
# ──────────────────────────────────────────────
def song_card_html(rank: int, row: pd.Series) -> str:
    name    = row["track_name"]
    artist  = row["artist"]
    sim     = row["similarity"]

    yt_query  = urllib.parse.quote_plus(f"{name} {artist}")
    sp_query  = urllib.parse.quote_plus(f"{name} {artist}")
    yt_url    = f"https://www.youtube.com/results?search_query={yt_query}"
    sp_url    = f"https://open.spotify.com/search/{sp_query}"

    # Feature bars (de-normalised back to readable %)
    feats = {
        "Energy":       row["energy"],
        "Danceability": row["danceability"],
        "Valence":      row["valence"],
    }
    bars = ""
    for label, val in feats.items():
        pct = int(val * 100)
        bars += f"""
        <div class='feat-row'>
          <div class='feat-label'>{label} {pct}%</div>
          <div class='feat-bar-bg'><div class='feat-bar-fill' style='width:{pct}%'></div></div>
        </div>"""

    return f"""
    <div class='song-card'>
      <div class='track-num'>{rank}</div>
      <div class='track-info'>
        <div class='track-name'>{name}</div>
        <div class='track-artist'>🎤 {artist}</div>
        <div style='margin-top:0.6rem'>{bars}</div>
      </div>
      <div style='display:flex;flex-direction:column;gap:0.5rem;align-items:flex-end'>
        <span class='score-badge'>Match {sim:.0%}</span>
        <a href='{yt_url}' target='_blank' class='link-btn'>▶ YouTube</a>
        <a href='{sp_url}' target='_blank' class='link-btn'>♫ Spotify</a>
      </div>
    </div>"""


# ──────────────────────────────────────────────
# 7. MAIN APP
# ──────────────────────────────────────────────
def main():
    # ── Load & preprocess data
    raw_df             = load_data()
    df, scaler         = preprocess(raw_df)

    # ── Hero header
    st.markdown("""
    <div class='hero-wrap'>
      <div class='vinyl-badge'>🎵</div>
      <div class='hero-title'>Mood-Based<br><span>Music Recommender</span> 🎧</div>
      <p class='hero-sub'>Tell us how you feel — we'll find the perfect soundtrack.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats strip
    st.markdown(f"""
    <div class='stats-strip'>
      <div class='stat-box'><div class='stat-num'>{len(df)}</div><div class='stat-desc'>Songs in library</div></div>
      <div class='stat-box'><div class='stat-num'>{len(MOOD_CONFIG)}</div><div class='stat-desc'>Mood profiles</div></div>
      <div class='stat-box'><div class='stat-num'>4</div><div class='stat-desc'>Audio features</div></div>
      <div class='stat-box'><div class='stat-num'>∞</div><div class='stat-desc'>Good vibes</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='groove-divider'>", unsafe_allow_html=True)

    # ── Control panel
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-label'>Select your mood(s)</div>", unsafe_allow_html=True)
        selected_moods = st.multiselect(
            label="Mood selector",
            options=MOOD_LABELS,
            default=[MOOD_LABELS[0]],
            label_visibility="collapsed",
            help="Select one or more moods to blend a recommendation!",
        )

        st.markdown("<div class='section-label' style='margin-top:1.2rem'>Or describe how you feel</div>",
                    unsafe_allow_html=True)
        text_input = st.text_input(
            label="Text input",
            placeholder="e.g. I feel pumped and ready to hit the gym 🏋️",
            label_visibility="collapsed",
        )
        if text_input:
            detected = detect_mood_from_text(text_input)
            if detected:
                st.success(f"Detected mood: **{detected}**  ← added to selection")
                if detected not in selected_moods:
                    selected_moods = list(set(selected_moods + [detected]))
            else:
                st.info("Couldn't detect a specific mood — using your dropdown selection.")

        top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=8, step=1)

    with col_right:
        if selected_moods:
            st.markdown("<div class='section-label'>Mood profile</div>", unsafe_allow_html=True)
            # Show profile card for first selected mood
            m = selected_moods[0]
            cfg = MOOD_CONFIG[m]
            vec = cfg["vector"]
            kw_chips = " ".join(f"<span class='kw-chip'>{k}</span>" for k in cfg["keywords"][:5])
            st.markdown(f"""
            <div class='mood-profile'>
              <h4>{m}</h4>
              <p style='font-size:0.85rem;color:#E8D5B0;margin-bottom:1rem'>{cfg['desc']}</p>
              <div style='margin-bottom:0.8rem'>
                <div class='feat-label' style='color:#C8873A'>Energy</div>
                <div class='feat-bar-bg'><div class='feat-bar-fill' style='width:{int(vec["energy"]*100)}%'></div></div>
              </div>
              <div style='margin-bottom:0.8rem'>
                <div class='feat-label' style='color:#C8873A'>Danceability</div>
                <div class='feat-bar-bg'><div class='feat-bar-fill' style='width:{int(vec["danceability"]*100)}%'></div></div>
              </div>
              <div style='margin-bottom:1rem'>
                <div class='feat-label' style='color:#C8873A'>Valence (positivity)</div>
                <div class='feat-bar-bg'><div class='feat-bar-fill' style='width:{int(vec["valence"]*100)}%'></div></div>
              </div>
              <div>{kw_chips}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Select at least one mood to see its profile.")

    st.markdown("</div>", unsafe_allow_html=True)  # close control-panel

    # ── Recommend button
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        recommend_clicked = st.button("🎵 Recommend Songs", use_container_width=True)

    # ──────────────────────────────────────────
    # RESULTS
    # ──────────────────────────────────────────
    if recommend_clicked:
        if not selected_moods:
            st.warning("Please select at least one mood first!")
            return

        with st.spinner("Finding your perfect tracks…"):
            time.sleep(0.8)   # small UX pause so spinner is visible
            results = recommend(df, scaler, selected_moods, top_n=top_n)

        st.markdown("<hr class='groove-divider'>", unsafe_allow_html=True)

        mood_str = " + ".join(selected_moods)
        st.markdown(f"""
        <div style='margin-bottom:1.5rem'>
          <div class='section-label'>Your Recommendations</div>
          <h2 style='font-family:"Playfair Display",serif;margin:0;font-size:1.8rem'>
            Top {top_n} tracks for <span style='color:var(--amber)'>{mood_str}</span>
          </h2>
        </div>
        """, unsafe_allow_html=True)

        # ── Song cards
        for i, (_, row) in enumerate(results.iterrows(), 1):
            st.markdown(song_card_html(i, row), unsafe_allow_html=True)

        # ── Summary table (toggle)
        with st.expander("📊 View data table"):
            show_df = results[["track_name", "artist", "energy", "danceability", "valence", "tempo", "similarity"]].copy()
            show_df.columns = ["Track", "Artist", "Energy", "Danceability", "Valence", "Tempo", "Match Score"]
            show_df["Match Score"] = show_df["Match Score"].map("{:.2%}".format)
            cols = ["Energy", "Danceability", "Valence"]
            show_df[cols] = show_df[cols].apply(lambda col: col.map(lambda x: ...))
            show_df["Tempo"] = show_df["Tempo"].map("{:.1f}".format)
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        # ── Footer note
        st.markdown("""
        <p style='text-align:center;color:var(--muted);font-size:0.8rem;margin-top:2rem'>
          Click ▶ YouTube or ♫ Spotify to listen · Match scores via cosine similarity
        </p>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
