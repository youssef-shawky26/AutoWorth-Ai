import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AutoWorth AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer, header {#visibility: hidden;}

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(255,45,74,0.10) 0%, transparent 45%),
            radial-gradient(circle at 100% 100%, rgba(255,45,74,0.06) 0%, transparent 40%),
            repeating-linear-gradient(135deg, #0b0b0d 0px, #0b0b0d 2px, #111113 2px, #111113 4px);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 1.2rem 0 2rem 0;
        border-bottom: 2px solid #ff2b4a;
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        color: #f4f4f5;
        text-shadow: 0 0 18px rgba(255, 43, 74, 0.45);
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }
    .hero p {
        color: #9a9aa2;
        font-size: 1rem;
        letter-spacing: 0.03em;
        margin: 0;
    }

    /* Section cards */
    .section-card {
        background: linear-gradient(180deg, #17171a 0%, #131315 100%);
        border: 1px solid #29292e;
        border-left: 3px solid #ff2b4a;
        border-radius: 10px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.3rem;
    }
    .section-title {
        color: #f4f4f5;
        font-size: 1.0rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
        background-color: #0c0c0e !important;
        border-radius: 6px !important;
        border: 1px solid #2e2e33 !important;
        color: #f0f0f2 !important;
    }
    label, .stNumberInput label, .stSelectbox label, .stTextInput label {
        color: #9a9aa2 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0c0c0e;
        border-right: 1px solid #29292e;
    }
    section[data-testid="stSidebar"] * { color: #d6d6da !important; }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff2b4a, #ff6a3d);
        color: #0b0b0d;
        font-weight: 800;
        font-size: 1.0rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.8rem 0;
        border-radius: 6px;
        border: none;
        box-shadow: 0 0 22px rgba(255, 43, 74, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(255, 43, 74, 0.55);
    }

    /* Result cards */
    .result-card {
        border-radius: 10px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        border: 1px solid #29292e;
    }
    .price-card {
        background: linear-gradient(180deg, #17171a 0%, #131315 100%);
        border-top: 3px solid #ff2b4a;
    }
    .price-card .label { color: #9a9aa2; font-size: 0.9rem; letter-spacing: 0.03em; text-transform: uppercase; }
    .price-card .value {
        font-size: 2.7rem;
        font-weight: 900;
        color: #ff2b4a;
        text-shadow: 0 0 18px rgba(255, 43, 74, 0.4);
        margin: 0.2rem 0;
    }

    .verdict-good { background: linear-gradient(180deg, #10241a 0%, #0d1c15 100%); border-top: 3px solid #2fbf6e; }
    .verdict-fair { background: linear-gradient(180deg, #2a2210 0%, #211b0c 100%); border-top: 3px solid #e0a72e; }
    .verdict-bad  { background: linear-gradient(180deg, #2a1113 0%, #210d0e 100%); border-top: 3px solid #ff2b4a; }

    .verdict-title { font-size: 1.5rem; font-weight: 900; letter-spacing: 0.03em; text-transform: uppercase; margin-bottom: 0.2rem; color: #f4f4f5; }
    .verdict-sub   { color: #9a9aa2; font-size: 0.92rem; }

    .footnote { color: #55555c; font-size: 0.78rem; text-align: center; margin-top: 2rem; letter-spacing: 0.02em; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Reference data (Make -> available Models), mirrors the training data
# ------------------------------------------------------------------
MAKE_MODELS = {
    "audi": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "Q2", "Q3", "Q5", "Q7", "Q8",
             "R8", "RS3", "RS4", "RS5", "RS6", "RS7", "S3", "S4", "S5", "S8", "SQ5", "SQ7", "TT"],
    "bmw": ["1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "6 Series", "7 Series",
            "8 Series", "M2", "M3", "M4", "M5", "M6", "X1", "X2", "X3", "X4", "X5", "X6", "X7",
            "Z3", "Z4", "i3", "i8"],
    "cclass": ["C Class"],
    "focus": ["Focus"],
    "ford": ["B-MAX", "C-MAX", "EcoSport", "Edge", "Escort", "Fiesta", "Focus", "Fusion", "Galaxy",
             "Grand C-MAX", "Grand Tourneo Connect", "KA", "Ka+", "Kuga", "Mondeo", "Mustang", "Puma",
             "Ranger", "S-MAX", "Streetka", "Tourneo Connect", "Tourneo Custom", "Transit Tourneo"],
    "hyundi": ["Accent", "Amica", "Getz", "I10", "I20", "I30", "I40", "I800", "IX20", "IX35",
               "Ioniq", "Kona", "Santa Fe", "Terracan", "Tucson", "Veloster"],
    "merc": ["180", "200", "220", "230", "A Class", "B Class", "C Class", "CL Class", "CLA Class",
             "CLC Class", "CLK", "CLS Class", "E Class", "G Class", "GL Class", "GLA Class",
             "GLB Class", "GLC Class", "GLE Class", "GLS Class", "M Class", "R Class", "S Class",
             "SL CLASS", "SLK", "V Class", "X-CLASS"],
    "skoda": ["Citigo", "Fabia", "Kamiq", "Karoq", "Kodiaq", "Octavia", "Rapid", "Roomster",
              "Scala", "Superb", "Yeti", "Yeti Outdoor"],
    "toyota": ["Auris", "Avensis", "Aygo", "C-HR", "Camry", "Corolla", "GT86", "Hilux", "IQ",
               "Land Cruiser", "PROACE VERSO", "Prius", "RAV4", "Supra", "Urban Cruiser", "Verso",
               "Verso-S", "Yaris"],
    "vauxhall": ["Adam", "Agila", "Ampera", "Antara", "Astra", "Cascada", "Combo Life", "Corsa",
                 "Crossland X", "GTC", "Grandland X", "Insignia", "Kadjar", "Meriva", "Mokka",
                 "Mokka X", "Tigra", "Vectra", "Viva", "Vivaro", "Zafira", "Zafira Tourer"],
    "vw": ["Amarok", "Arteon", "Beetle", "CC", "Caddy", "Caddy Life", "Caddy Maxi",
           "Caddy Maxi Life", "California", "Caravelle", "Eos", "Fox", "Golf", "Golf SV", "Jetta",
           "Passat", "Polo", "Scirocco", "Sharan", "Shuttle", "T-Cross", "T-Roc", "Tiguan",
           "Tiguan Allspace", "Touareg", "Touran", "Up"],
}

TRANSMISSIONS = ["Manual", "Automatic", "Semi-Auto", "Other"]
FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    load_error = str(e)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🚗 AutoWorth AI</h1>
    <p>Used Car Price &amp; Deal Advisor — powered by a Random Forest model</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        f"⚠️ Could not load `model.pkl`. Make sure it sits in the same folder as this app.\n\n{load_error}"
    )
    st.stop()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this app")
    st.markdown(
        "AutoWorth AI estimates the **fair market price** of a used car from its "
        "specs, then compares it against a seller's asking price to flag whether "
        "the deal looks good, fair, or overpriced."
    )
    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown(
        "- **Algorithm:** Random Forest Regressor\n"
        "- **Encoding:** Target encoding (model) + One-Hot (make / fuel / transmission)\n"
        "- **Scaling:** Robust Scaler on numeric features\n"
        "- **Test R²:** ≈ 0.96"
    )
    st.markdown("---")
    st.caption("Machine Learning Project by Youssef Mohamed")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
st.markdown('<div class="section-card"><div class="section-title">🏷️ Identity</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    make = st.selectbox("Brand", sorted(MAKE_MODELS.keys()), format_func=lambda x: x.upper())
with c2:
    model_name = st.selectbox("Model", MAKE_MODELS[make])
with c3:
    year = st.number_input("Year", min_value=1990, max_value=2026, value=2019, step=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">⚙️ Specifications</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    transmission = st.selectbox("Transmission", TRANSMISSIONS)
with c2:
    fuelType = st.selectbox("Fuel Type", FUEL_TYPES)
with c3:
    engineSize = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.6, value=1.6, step=0.1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">📊 Usage &amp; Running Costs</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    mileage = st.number_input("Mileage (miles)", min_value=0, max_value=400000, value=25000, step=500)
with c2:
    mpg = st.number_input("MPG", min_value=0.0, max_value=470.0, value=50.0, step=0.5)
with c3:
    tax = st.number_input("Annual Tax (£)", min_value=0.0, max_value=600.0, value=145.0, step=5.0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">💷 Deal Check</div>', unsafe_allow_html=True)
asking_price = st.number_input(
    "Seller's Asking Price (£)", min_value=0.0, value=15000.0, step=100.0,
    help="Enter what the seller is asking for — we'll compare it to the model's fair-price estimate."
)
st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("🔮 Predict Market Price")

# ------------------------------------------------------------------
# Prediction & results
# ------------------------------------------------------------------
if predict_clicked:
    # Column names/order must match what the training pipeline expects
    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize],
        "Make": [make],
    })

    with st.spinner("Crunching the numbers..."):
        predicted_price = float(model.predict(input_data)[0])

    difference = asking_price - predicted_price
    pct_diff = (difference / predicted_price * 100) if predicted_price else 0

    st.markdown("---")

    r1, r2 = st.columns([1, 1.2])

    with r1:
        st.markdown(f"""
        <div class="result-card price-card">
            <div class="label">Estimated Market Price</div>
            <div class="value">£{predicted_price:,.0f}</div>
            <div class="label">vs. asking price of £{asking_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        if difference < 0:
            verdict_class, title, sub = "verdict-good", "🟢 Good Deal", \
                f"Priced £{abs(difference):,.0f} ({abs(pct_diff):.1f}%) below Market value"
        elif difference < predicted_price * 0.10:
            verdict_class, title, sub = "verdict-fair", "🟡 Fair Deal", \
                f"Priced £{difference:,.0f} ({pct_diff:.1f}%) above Market value — within a reasonable range"
        else:
            verdict_class, title, sub = "verdict-bad", "🔴 Overpriced", \
                f"Priced £{difference:,.0f} ({pct_diff:.1f}%) above Market value"

        st.markdown(f"""
        <div class="result-card {verdict_class}">
            <div class="verdict-title">{title}</div>
            <div class="verdict-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Comparison bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Asking Price", "Market Price (AI)"],
        x=[asking_price, predicted_price],
        orientation="h",
        marker=dict(color=["#6b6b73", "#ff2b4a"]),
        text=[f"£{asking_price:,.0f}", f"£{predicted_price:,.0f}"],
        textposition="outside",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=40, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d6d6da"),
        xaxis=dict(showgrid=False, title="£"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="footnote">Estimates are generated by a machine-learning model trained on '
        'historical UK used-car listings and are provided for guidance only — always verify with '
        'an independent inspection.</div>',
        unsafe_allow_html=True,
    )
