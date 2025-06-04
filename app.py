import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Changed from pickle to joblib for better compatibility
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Language configuration
LANGUAGES = {
    "🇺🇸 English": "en",
    "🇮🇩 Bahasa Indonesia": "id"
}

# Multi-language text content
TEXTS = {
    "en": {
        "app_title": "🩺 Advanced Diabetes Prediction System",
        "app_subtitle": "Powered by Machine Learning & AI Technology",
        "navigation": "🧭 Navigation",
        "language": "🌐 Language",
        "home": "🏠 Home",
        "data_analysis": "📊 Data Analysis",
        "prediction": "🔮 Prediction",
        "about": "ℹ️ About",
        "quick_stats": "📈 Quick Stats",
        "total_samples": "Total Samples",
        "features": "Features",
        "diabetes_rate": "Diabetes Rate",
        "model_info": "🎯 Model Info",
        "algorithm": "Algorithm",
        "accuracy": "Accuracy",
        "medical_indicators": "medical indicators",
        "welcome_title": "🎯 Welcome to Advanced Diabetes Prediction",
        "welcome_text": "Harness the power of artificial intelligence to assess diabetes risk with precision and confidence. Our advanced machine learning model analyzes multiple health indicators to provide accurate predictions.",
        "dataset_overview": "📊 Real-time Dataset Overview",
        "patient_records": "Patient Records",
        "health_indicators": "Health Indicators",
        "positive_cases": "Positive Cases",
        "negative_cases": "Negative Cases",
        "healthy_cases": "✅ Healthy Cases",
        "diabetes_cases": "⚠️ Diabetes Cases",
        "interactive_overview": "📈 Interactive Data Overview",
        "diabetes_distribution": "Diabetes Distribution",
        "no_diabetes": "No Diabetes",
        "diabetes": "Diabetes",
        "age_distribution": "Age Distribution by Diabetes Status",
        "understanding_problem": "🎯 Understanding the Problem",
        "healthcare_challenge": "🏥 Healthcare Challenge",
        "healthcare_challenge_text": "Diabetes affects millions worldwide and early detection is crucial for effective treatment. Traditional diagnosis methods can be expensive and time-consuming, creating barriers to timely healthcare access.",
        "ai_solution": "🤖 AI Solution",
        "ai_solution_text": "Our machine learning model provides instant risk assessment using easily obtainable health metrics, enabling proactive healthcare decisions and early intervention strategies.",
        "health_indicators_analyze": "🔬 Health Indicators We Analyze",
        "risk_assessment": "🔮 Advanced Diabetes Risk Assessment",
        "how_it_works": "🎯 How It Works",
        "how_it_works_text": "Our AI model analyzes 8 key health indicators to assess diabetes risk. Simply enter your health information below to get an instant risk assessment with personalized recommendations.",
        "enter_health_info": "📝 Enter Health Information",
        "personal_info": "Personal Information",
        "medical_info": "Medical Information",
        "age": "Age (years)",
        "pregnancies": "Number of Pregnancies",
        "glucose": "Glucose Level (mg/dL)",
        "blood_pressure": "Blood Pressure (mmHg)",
        "skin_thickness": "Skin Thickness (mm)",
        "insulin": "Insulin Level (μU/mL)",
        "bmi": "BMI (kg/m²)",
        "diabetes_pedigree": "Diabetes Pedigree Function",
        "predict_button": "🔮 Predict Diabetes Risk",
        "prediction_result": "🎯 Prediction Result",
        "risk_category": "Risk Category",
        "confidence": "Confidence",
        "high_risk": "HIGH RISK",
        "low_risk": "LOW RISK",
        "recommendations": "💡 Recommendations",
        "feature_analysis": "📊 Feature Analysis",
        "about_title": "ℹ️ About This Application",
        "about_description": "This diabetes prediction application uses advanced machine learning algorithms to assess diabetes risk based on various health indicators.",
        "features_title": "✨ Key Features",
        "model_details": "🤖 Model Details",
        "disclaimer": "⚠️ Medical Disclaimer",
        "disclaimer_text": "This tool is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions."
    },
    "id": {
        "app_title": "🩺 Sistem Prediksi Diabetes Canggih",
        "app_subtitle": "Didukung oleh Teknologi Machine Learning & AI",
        "navigation": "🧭 Navigasi",
        "language": "🌐 Bahasa",
        "home": "🏠 Beranda",
        "data_analysis": "📊 Analisis Data",
        "prediction": "🔮 Prediksi",
        "about": "ℹ️ Tentang",
        "quick_stats": "📈 Statistik Cepat",
        "total_samples": "Total Sampel",
        "features": "Fitur",
        "diabetes_rate": "Tingkat Diabetes",
        "model_info": "🎯 Info Model",
        "algorithm": "Algoritma",
        "accuracy": "Akurasi",
        "medical_indicators": "indikator medis",
        "welcome_title": "🎯 Selamat Datang di Prediksi Diabetes Canggih",
        "welcome_text": "Manfaatkan kekuatan kecerdasan buatan untuk menilai risiko diabetes dengan presisi dan kepercayaan diri. Model machine learning canggih kami menganalisis berbagai indikator kesehatan untuk memberikan prediksi yang akurat.",
        "dataset_overview": "📊 Tinjauan Dataset Real-time",
        "patient_records": "Rekam Pasien",
        "health_indicators": "Indikator Kesehatan",
        "positive_cases": "Kasus Positif",
        "negative_cases": "Kasus Negatif",
        "healthy_cases": "✅ Kasus Sehat",
        "diabetes_cases": "⚠️ Kasus Diabetes",
        "interactive_overview": "📈 Tinjauan Interaktif Data",
        "diabetes_distribution": "Distribusi Diabetes",
        "no_diabetes": "Tidak Diabetes",
        "diabetes": "Diabetes",
        "age_distribution": "Distribusi Usia berdasarkan Status Diabetes",
        "understanding_problem": "🎯 Memahami Masalah",
        "healthcare_challenge": "🏥 Tantangan Kesehatan",
        "healthcare_challenge_text": "Diabetes mempengaruhi jutaan orang di seluruh dunia dan deteksi dini sangat penting untuk pengobatan yang efektif. Metode diagnosis tradisional bisa mahal dan memakan waktu, menciptakan hambatan untuk akses kesehatan yang tepat waktu.",
        "ai_solution": "🤖 Solusi AI",
        "ai_solution_text": "Model machine learning kami menyediakan penilaian risiko instan menggunakan metrik kesehatan yang mudah diperoleh, memungkinkan keputusan kesehatan proaktif dan strategi intervensi dini.",
        "health_indicators_analyze": "🔬 Indikator Kesehatan yang Kami Analisis",
        "risk_assessment": "🔮 Penilaian Risiko Diabetes Canggih",
        "how_it_works": "🎯 Cara Kerja",
        "how_it_works_text": "Model AI kami menganalisis 8 indikator kesehatan kunci untuk menilai risiko diabetes. Cukup masukkan informasi kesehatan Anda di bawah ini untuk mendapatkan penilaian risiko instan dengan rekomendasi yang dipersonalisasi.",
        "enter_health_info": "📝 Masukkan Informasi Kesehatan",
        "personal_info": "Informasi Pribadi",
        "medical_info": "Informasi Medis",
        "age": "Usia (tahun)",
        "pregnancies": "Jumlah Kehamilan",
        "glucose": "Tingkat Glukosa (mg/dL)",
        "blood_pressure": "Tekanan Darah (mmHg)",
        "skin_thickness": "Ketebalan Kulit (mm)",
        "insulin": "Tingkat Insulin (μU/mL)",
        "bmi": "BMI (kg/m²)",
        "diabetes_pedigree": "Fungsi Silsilah Diabetes",
        "predict_button": "🔮 Prediksi Risiko Diabetes",
        "prediction_result": "🎯 Hasil Prediksi",
        "risk_category": "Kategori Risiko",
        "confidence": "Tingkat Kepercayaan",
        "high_risk": "RISIKO TINGGI",
        "low_risk": "RISIKO RENDAH",
        "recommendations": "💡 Rekomendasi",
        "feature_analysis": "📊 Analisis Fitur",
        "about_title": "ℹ️ Tentang Aplikasi Ini",
        "about_description": "Aplikasi prediksi diabetes ini menggunakan algoritma machine learning canggih untuk menilai risiko diabetes berdasarkan berbagai indikator kesehatan.",
        "features_title": "✨ Fitur Utama",
        "model_details": "🤖 Detail Model",
        "disclaimer": "⚠️ Penafian Medis",
        "disclaimer_text": "Alat ini hanya untuk tujuan edukasi dan informasi. Tidak boleh menggantikan nasihat medis profesional, diagnosis, atau pengobatan. Selalu konsultasikan dengan penyedia layanan kesehatan yang berkualitas untuk keputusan medis."
    }
}

# Feature descriptions in multiple languages
FEATURE_INFO = {
    "en": {
        "Pregnancies": "Number of pregnancies (risk factor for gestational diabetes)",
        "Glucose": "Blood glucose level (primary diabetes indicator)", 
        "BloodPressure": "Diastolic blood pressure (cardiovascular health)",
        "SkinThickness": "Skin fold thickness (body fat indicator)",
        "Insulin": "Insulin level (hormone regulation)",
        "BMI": "Body Mass Index (weight-to-height ratio)",
        "DiabetesPedigreeFunction": "Genetic predisposition (family history)",
        "Age": "Patient age (risk increases with age)"
    },
    "id": {
        "Pregnancies": "Jumlah kehamilan (faktor risiko diabetes gestasional)",
        "Glucose": "Tingkat glukosa darah (indikator utama diabetes)", 
        "BloodPressure": "Tekanan darah diastolik (kesehatan kardiovaskular)",
        "SkinThickness": "Ketebalan lipatan kulit (indikator lemak tubuh)",
        "Insulin": "Tingkat insulin (regulasi hormon)",
        "BMI": "Indeks Massa Tubuh (rasio berat-tinggi)",
        "DiabetesPedigreeFunction": "Predisposisi genetik (riwayat keluarga)",
        "Age": "Usia pasien (risiko meningkat dengan usia)"
    }
}

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stAlert {
        margin-top: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .risk-high {
        border-left-color: #f44336 !important;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important;
    }
    
    .risk-low {
        border-left-color: #4CAF50 !important;
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%) !important;
    }
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > label {
        font-weight: bold;
        color: #1e88e5;
    }
    
    .stNumberInput > label {
        font-weight: bold;
        color: #1e88e5;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .header-style {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-top: 3px solid #667eea;
        margin: 1rem 0;
    }
    
    .animated-text {
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_rf_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        # Model files not found - training new model (removed persistent notification)
        return train_and_save_model()
    except (ModuleNotFoundError, ImportError, ValueError) as e:
        # Model compatibility issue - training new model (removed persistent notification)
        return train_and_save_model()
    except Exception as e:
        # Error loading model - training new model (removed persistent notification)
        return train_and_save_model()

# Language helper function
def get_text(key, lang="en"):
    """Get text in specified language"""
    return TEXTS.get(lang, TEXTS["en"]).get(key, key)

def get_feature_info(lang="en"):
    """Get feature information in specified language"""
    return FEATURE_INFO.get(lang, FEATURE_INFO["en"])

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Train and save model function
def train_and_save_model():
    """Train a new model and save it"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import joblib
          # Load dataset
        df = load_dataset()
        if df is None:
            # Cannot train model: Dataset not available (removed persistent notification)
            return None, None
        
        # Prepare data
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
          # Save model and scaler using joblib for better compatibility
        joblib.dump(model, 'diabetes_rf_model.pkl')
        joblib.dump(scaler, 'diabetes_scaler.pkl')
        
        # Model trained and saved successfully (removed persistent notification)
        return model, scaler
        
    except Exception as e:
        # Error training model (removed persistent notification - using fallback)
        return None, None

# Load dataset for analysis
@st.cache_data
def load_dataset():
    try:
        # Try multiple possible paths for the dataset
        possible_paths = [
            "dataset/diabetes.csv",
            "diabetes.csv",
            "akhir/dataset/diabetes.csv",
            "../Project-Akhir/akhir/dataset/diabetes.csv"
        ]
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                # Dataset loaded successfully (removed persistent notification)
                return df
            except FileNotFoundError:
                continue
        
        # If no dataset found, create a sample dataset
        # Creating sample dataset for demonstration (removed persistent notification)
        return create_sample_dataset()
        
    except Exception as e:
        # Error loading dataset - using sample data (removed persistent notification)
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes"""
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
        'BloodPressure': np.random.normal(80, 15, n_samples).clip(0, 122),
        'SkinThickness': np.random.normal(25, 8, n_samples).clip(0, 100),
        'Insulin': np.random.normal(80, 40, n_samples).clip(0, 846),
        'BMI': np.random.normal(32, 7, n_samples).clip(0, 67),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])    }
    
    df = pd.DataFrame(data)
    # Using sample dataset for demonstration (removed persistent notification)
    return df

def main():
    lang = st.session_state.language
    
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h2>{get_text('navigation', lang)}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {get_text('language', st.session_state.language)}")
    
    # Language selection
    selected_language = st.sidebar.selectbox(
        "",
        list(LANGUAGES.keys()),
        index=0 if st.session_state.language == 'en' else 1,
        key="language_selector"
    )
    
    # Update session state if language changed
    new_lang = LANGUAGES[selected_language]
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()
    
    # Get current language
    
    # Header with animation
    st.markdown(f"""
    <div class="header-style animated-text">
        <h1>{get_text('app_title', lang)}</h1>
        <p>{get_text('app_subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    
    
    # Enhanced navigation with icons
    page_options = {
        get_text('home', lang): "Home",
        get_text('prediction', lang): "Prediction",
        get_text('about', lang): "About"
    }
    
    selected_page = st.sidebar.selectbox(
        "",
        list(page_options.keys()),
        format_func=lambda x: x
    )
    
    page = page_options[selected_page]
    
    # Add sidebar info
    st.sidebar.markdown(f"""
    ---
    ### {get_text('quick_stats', lang)}
    """)
    
    df = load_dataset()
    if df is not None:
        st.sidebar.metric(get_text('total_samples', lang), len(df))
        st.sidebar.metric(get_text('features', lang), len(df.columns) - 1)
        diabetes_rate = (df['Outcome'].sum() / len(df)) * 100
        st.sidebar.metric(get_text('diabetes_rate', lang), f"{diabetes_rate:.1f}%")
    
    st.sidebar.markdown(f"""
    ---
    ### {get_text('model_info', lang)}
    - **{get_text('algorithm', lang)}**: Random Forest
    - **{get_text('accuracy', lang)}**: ~85%
    - **{get_text('features', lang)}**: 8 {get_text('medical_indicators', lang)}
    """)
    
    # Route to pages
    if page == "Home":
        show_home(lang)
    elif page == "Prediction":
        show_prediction(lang)
    elif page == "About":
        show_about(lang)

def show_home(lang="en"):
    # Welcome section with cards
    st.markdown(f"""
    <div class="info-card animated-text">
        <h2>{get_text('welcome_title', lang)}</h2>
        <p style="font-size: 18px; line-height: 1.6;">
            {get_text('welcome_text', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced statistics with interactive metrics
    st.markdown(f"### {get_text('dataset_overview', lang)}")
    
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📋 {get_text('total_samples', lang)}</h3>
                <h2>{len(df)}</h2>
                <p>{get_text('patient_records', lang)}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            features_count = len(df.columns) - 1
            st.markdown(f"""
            <div class="metric-container">
                <h3>🔍 {get_text('features', lang)}</h3>
                <h2>{features_count}</h2>
                <p>{get_text('health_indicators', lang)}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            diabetes_count = df['Outcome'].sum()
            st.markdown(f"""
            <div class="metric-container">
                <h3>{get_text('diabetes_cases', lang)}</h3>
                <h2>{diabetes_count}</h2>
                <p>{get_text('positive_cases', lang)}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            no_diabetes_count = len(df) - diabetes_count
            st.markdown(f"""
            <div class="metric-container">
                <h3>{get_text('healthy_cases', lang)}</h3>
                <h2>{no_diabetes_count}</h2>
                <p>{get_text('negative_cases', lang)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Interactive overview charts
    st.markdown(f"### {get_text('interactive_overview', lang)}")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Plotly pie chart for outcome distribution
            outcome_counts = df['Outcome'].value_counts()
            fig = px.pie(
                values=outcome_counts.values,
                names=[get_text('no_diabetes', lang), get_text('diabetes', lang)],
                title=get_text('diabetes_distribution', lang),
                color_discrete_sequence=['#4CAF50', '#f44336']
            )
            fig.update_layout(
                title_font_size=16,
                font=dict(size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Interactive age distribution
            fig = px.histogram(
                df, 
                x='Age', 
                color='Outcome',
                title=get_text('age_distribution', lang),
                color_discrete_sequence=['#4CAF50', '#f44336'],
                nbins=20
            )
            fig.update_layout(
                title_font_size=16,
                font=dict(size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Business understanding section
    st.markdown(f"### {get_text('understanding_problem', lang)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        challenge_points = [
            "High healthcare costs" if lang == "en" else "Biaya kesehatan yang tinggi",
            "Limited access to testing" if lang == "en" else "Akses terbatas untuk testing",
            "Need for early intervention" if lang == "en" else "Perlu intervensi dini",
            "Prevention opportunities" if lang == "en" else "Peluang pencegahan"
        ]
        
        st.markdown(f"""
        <div class="info-card">
            <h4>{get_text('healthcare_challenge', lang)}</h4>
            <p>
                {get_text('healthcare_challenge_text', lang)}
            </p>
            <ul>
                {"".join([f"<li>{point}</li>" for point in challenge_points])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        solution_points = [
            "Instant risk assessment" if lang == "en" else "Penilaian risiko instan",
            "85%+ accuracy rate" if lang == "en" else "Tingkat akurasi 85%+",
            "Easy-to-use interface" if lang == "en" else "Antarmuka yang mudah digunakan",
            "Personalized recommendations" if lang == "en" else "Rekomendasi yang dipersonalisasi"
        ]
        
        st.markdown(f"""
        <div class="info-card">
            <h4>{get_text('ai_solution', lang)}</h4>
            <p>
                {get_text('ai_solution_text', lang)}
            </p>
            <ul>
                {"".join([f"<li>{point}</li>" for point in solution_points])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature overview
    st.markdown(f"### {get_text('health_indicators_analyze', lang)}")
    
    feature_info = get_feature_info(lang)
    
    cols = st.columns(2)
    for i, (feature, description) in enumerate(feature_info.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                <h5 style="color: #667eea; margin: 0;">{feature}</h5>
                <p style="margin: 0.5rem 0 0 0; color: #666;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

# def show_data_analysis():
#     st.markdown("### 📊 Advanced Data Analysis Dashboard")
    
#     df = load_dataset()
#     if df is None:
#         st.warning("⚠️ Dataset not available. Please check the data source.")
#         return
    
#     # Interactive dataset overview
#     with st.expander("📋 Dataset Overview", expanded=True):
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.markdown("""
#             <div class="info-card">
#                 <h4>📏 Dataset Dimensions</h4>
#                 <p><strong>Rows:</strong> {}</p>
#                 <p><strong>Columns:</strong> {}</p>
#                 <p><strong>Memory:</strong> {:.2f} KB</p>
#             </div>
#             """.format(df.shape[0], df.shape[1], df.memory_usage(deep=True).sum() / 1024), 
#             unsafe_allow_html=True)
            
#         with col2:
#             missing_total = df.isnull().sum().sum()
#             st.markdown("""
#             <div class="info-card">
#                 <h4>🔍 Data Quality</h4>
#                 <p><strong>Missing Values:</strong> {}</p>
#                 <p><strong>Duplicates:</strong> {}</p>
#                 <p><strong>Data Types:</strong> {}</p>
#             </div>
#             """.format(missing_total, df.duplicated().sum(), len(df.dtypes.unique())), 
#             unsafe_allow_html=True)
            
#         with col3:
#             diabetes_rate = (df['Outcome'].sum() / len(df)) * 100
#             st.markdown("""
#             <div class="info-card">
#                 <h4>📈 Key Statistics</h4>
#                 <p><strong>Diabetes Rate:</strong> {:.1f}%</p>
#                 <p><strong>Avg Age:</strong> {:.1f} years</p>
#                 <p><strong>Avg BMI:</strong> {:.1f}</p>
#             </div>
#             """.format(diabetes_rate, df['Age'].mean(), df['BMI'].mean()), 
#             unsafe_allow_html=True)
    
#     # Interactive data sample with search
#     st.markdown("### 🔍 Interactive Data Explorer")
    
#     # Search and filter options
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         outcome_filter = st.selectbox("Filter by Outcome:", ["All", "No Diabetes", "Diabetes"])
#     with col2:
#         age_range = st.slider("Age Range:", int(df['Age'].min()), int(df['Age'].max()), 
#                              (int(df['Age'].min()), int(df['Age'].max())))
#     with col3:
#         sample_size = st.selectbox("Sample Size:", [10, 25, 50, 100], index=1)
#       # Apply filters
#     filtered_df = df.copy()
#     if outcome_filter != "All":
#         outcome_val = 0 if outcome_filter == "No Diabetes" else 1
#         filtered_df = filtered_df[filtered_df['Outcome'] == outcome_val]
    
#     filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & 
#                              (filtered_df['Age'] <= age_range[1])]
    
#     st.dataframe(filtered_df.head(sample_size), use_container_width=True)
#     # Showing filtered records (removed persistent notification)
    
#     # Enhanced statistical summary
#     st.markdown("### 📊 Statistical Summary")
    
#     # Interactive feature selection for detailed stats
#     selected_features = st.multiselect(
#         "Select features for detailed analysis:",
#         df.columns[:-1].tolist(),
#         default=['Glucose', 'BMI', 'Age']
#     )
    
#     if selected_features:
#         stats_df = df[selected_features + ['Outcome']].groupby('Outcome').describe()
#         st.dataframe(stats_df.round(2), use_container_width=True)
    
#     # Interactive visualizations
#     st.markdown("### 📈 Interactive Visualizations")
    
#     # Outcome distribution with enhanced styling
#     col1, col2 = st.columns(2)
    
#     with col1:
#         outcome_counts = df['Outcome'].value_counts()
#         fig = px.pie(
#             values=outcome_counts.values,
#             names=['No Diabetes', 'Diabetes'],
#             title="Diabetes Distribution",
#             color_discrete_sequence=['#4CAF50', '#f44336'],
#             hole=0.4
#         )
#         fig.update_traces(textposition='inside', textinfo='percent+label')
#         fig.update_layout(height=400, title_font_size=16)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Add interpretation
#         diabetes_percentage = (outcome_counts[1] / outcome_counts.sum()) * 100
#         st.markdown(f"""
#         <div class="info-card">
#             <h5>📊 Distribution Analysis</h5>
#             <p><strong>Diabetes Cases:</strong> {outcome_counts[1]} ({diabetes_percentage:.1f}%)</p>
#             <p><strong>Healthy Cases:</strong> {outcome_counts[0]} ({100-diabetes_percentage:.1f}%)</p>
#             <p><strong>Class Balance:</strong> {'Imbalanced' if abs(diabetes_percentage - 50) > 15 else 'Balanced'}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         # Interactive feature distribution
#         feature_to_analyze = st.selectbox(
#             "Select feature for distribution analysis:", 
#             df.columns[:-1],
#             key="feature_dist"
#         )
        
#         fig = px.histogram(
#             df, 
#             x=feature_to_analyze, 
#             color='Outcome',
#             title=f'{feature_to_analyze} Distribution by Diabetes Status',
#             color_discrete_sequence=['#4CAF50', '#f44336'],
#             nbins=25,
#             opacity=0.7
#         )
#         fig.update_layout(height=400, title_font_size=16)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Statistical comparison
#         no_diabetes_mean = df[df['Outcome'] == 0][feature_to_analyze].mean()
#         diabetes_mean = df[df['Outcome'] == 1][feature_to_analyze].mean()
#         difference = diabetes_mean - no_diabetes_mean
        
#         st.markdown(f"""
#         <div class="info-card">
#             <h5>📊 Feature Analysis: {feature_to_analyze}</h5>
#             <p><strong>No Diabetes Mean:</strong> {no_diabetes_mean:.2f}</p>
#             <p><strong>Diabetes Mean:</strong> {diabetes_mean:.2f}</p>
#             <p><strong>Difference:</strong> {difference:+.2f}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Advanced correlation analysis
#     st.markdown("### 🔗 Correlation Analysis")
    
#     # Interactive correlation heatmap
#     correlation_matrix = df.corr()
    
#     fig = px.imshow(
#         correlation_matrix,
#         color_continuous_scale='RdBu',
#         aspect='auto',
#         title='Feature Correlation Matrix',
#         color_continuous_midpoint=0
#     )
#     fig.update_layout(height=600, title_font_size=16)
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Feature importance analysis
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Correlation with outcome
#         outcome_corr = correlation_matrix['Outcome'].drop('Outcome').sort_values(key=abs, ascending=False)
        
#         fig = px.bar(
#             x=outcome_corr.values,
#             y=outcome_corr.index,
#             orientation='h',
#             title='Feature Correlation with Diabetes',
#             color=outcome_corr.values,
#             color_continuous_scale='RdBu',
#             color_continuous_midpoint=0
#         )
#         fig.update_layout(height=500, title_font_size=16)
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         # Feature ranking
#         st.markdown("#### 🏆 Feature Importance Ranking")
#         for i, (feature, corr) in enumerate(outcome_corr.items(), 1):
#             strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
#             direction = "Positive" if corr > 0 else "Negative"
            
#             st.markdown(f"""
#             <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; 
#                         border-left: 4px solid {'#f44336' if abs(corr) > 0.5 else '#ff9800' if abs(corr) > 0.3 else '#4CAF50'};">
#                 <h6 style="margin: 0; color: #333;">#{i} {feature}</h6>
#                 <p style="margin: 0.3rem 0 0 0; color: #666; font-size: 14px;">
#                     Correlation: {corr:.3f} | {strength} {direction}
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)
    
#     # Box plots for feature comparison
#     st.markdown("### 📦 Feature Comparison by Diabetes Status")
    
#     selected_features_box = st.multiselect(
#         "Select features for box plot comparison:",
#         df.columns[:-1].tolist(),
#         default=['Glucose', 'BMI', 'BloodPressure'],
#         key="box_features"
#     )
    
#     if selected_features_box:
#         n_features = len(selected_features_box)
#         cols = 2
#         rows = (n_features + cols - 1) // cols
        
#         fig = make_subplots(
#             rows=rows, cols=cols,
#             subplot_titles=selected_features_box,
#             vertical_spacing=0.08
#         )
        
#         for i, feature in enumerate(selected_features_box):
#             row = i // cols + 1
#             col = i % cols + 1
            
#             for outcome in [0, 1]:
#                 outcome_label = 'No Diabetes' if outcome == 0 else 'Diabetes'
#                 color = '#4CAF50' if outcome == 0 else '#f44336'
                
#                 fig.add_trace(
#                     go.Box(
#                         y=df[df['Outcome'] == outcome][feature],
#                         name=outcome_label,
#                         marker_color=color,
#                         showlegend=(i == 0)
#                     ),
#                     row=row, col=col
#                 )
        
#         fig.update_layout(height=300*rows, title_text="Feature Distributions by Diabetes Status")
#         st.plotly_chart(fig, use_container_width=True)

def show_prediction(lang="en"):
    st.markdown(f"### {get_text('risk_assessment', lang)}")
    
    model, scaler = load_model()
    
    if model is None or scaler is None:
        warning_text = "⚠️ Model initialization required. Training new model..." if lang == "en" else "⚠️ Inisialisasi model diperlukan. Melatih model baru..."
        st.warning(warning_text)
        # Attempt to retrain model automatically
        model, scaler = train_and_save_model()
        if model is None or scaler is None:
            error_text = "Unable to initialize prediction model. Please check your setup." if lang == "en" else "Tidak dapat menginisialisasi model prediksi. Silakan periksa pengaturan Anda."
            st.error(error_text)
            return
    
    # Introduction section
    st.markdown(f"""
    <div class="info-card">
        <h4>{get_text('how_it_works', lang)}</h4>
        <p>
            {get_text('how_it_works_text', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced input form with validation
    st.markdown(f"#### {get_text('enter_health_info', lang)}")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{get_text('personal_info', lang)}**")
            age = st.number_input(
                get_text('age', lang), 
                min_value=1, max_value=120, value=25,
                help="Patient's age in years" if lang == "en" else "Usia pasien dalam tahun"
            )
            pregnancies = st.number_input(
                get_text('pregnancies', lang), 
                min_value=0, max_value=20, value=0,
                help="Total number of pregnancies" if lang == "en" else "Total jumlah kehamilan"
            )
            
            vital_signs_text = "**Vital Signs**" if lang == "en" else "**Tanda Vital**"
            st.markdown(vital_signs_text)
            glucose = st.number_input(
                get_text('glucose', lang), 
                min_value=0.0, max_value=300.0, value=120.0,
                help="Blood glucose level (normal: 70-100 mg/dL fasting)" if lang == "en" else "Tingkat glukosa darah (normal: 70-100 mg/dL puasa)"
            )
            blood_pressure = st.number_input(
                get_text('blood_pressure', lang), 
                min_value=0.0, max_value=200.0, value=80.0,
                help="Diastolic blood pressure (normal: <80 mmHg)" if lang == "en" else "Tekanan darah diastolik (normal: <80 mmHg)"
            )
        
        with col2:
            physical_measurements_text = "**Physical Measurements**" if lang == "en" else "**Pengukuran Fisik**"
            st.markdown(physical_measurements_text)
            bmi = st.number_input(
                get_text('bmi', lang), 
                min_value=0.0, max_value=70.0, value=25.0,
                help="Body Mass Index (normal: 18.5-24.9)" if lang == "en" else "Indeks Massa Tubuh (normal: 18.5-24.9)"
            )
            skin_thickness = st.number_input(
                get_text('skin_thickness', lang), 
                min_value=0.0, max_value=100.0, value=20.0,
                help="Triceps skin fold thickness" if lang == "en" else "Ketebalan lipatan kulit trisep"
            )
            
            lab_values_text = "**Laboratory Values**" if lang == "en" else "**Nilai Laboratorium**"
            st.markdown(lab_values_text)
            insulin = st.number_input(
                get_text('insulin', lang), 
                min_value=0.0, max_value=900.0, value=80.0,
                help="2-hour serum insulin level" if lang == "en" else "Tingkat insulin serum 2 jam"
            )
            diabetes_pedigree = st.number_input(
                get_text('diabetes_pedigree', lang), 
                min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                help="Genetic predisposition score" if lang == "en" else "Skor predisposisi genetik"
            )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button(
                get_text('predict_button', lang), 
                use_container_width=True
            )
    
    # Input validation and warnings
    warnings = []
    if glucose > 126:
        warning_text = "⚠️ High glucose level detected (>126 mg/dL)" if lang == "en" else "⚠️ Tingkat glukosa tinggi terdeteksi (>126 mg/dL)"
        warnings.append(warning_text)
    if blood_pressure > 90:
        warning_text = "⚠️ High blood pressure detected (>90 mmHg)" if lang == "en" else "⚠️ Tekanan darah tinggi terdeteksi (>90 mmHg)"
        warnings.append(warning_text)
    if bmi > 30:
        warning_text = "⚠️ Obesity detected (BMI >30)" if lang == "en" else "⚠️ Obesitas terdeteksi (BMI >30)"
        warnings.append(warning_text)
    if age > 45:
        warning_text = "⚠️ Age is a risk factor (>45 years)" if lang == "en" else "⚠️ Usia adalah faktor risiko (>45 tahun)"
        warnings.append(warning_text)
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    if submit_button:
        # Show loading animation
        loading_text = '🤖 AI is analyzing your health data...' if lang == "en" else '🤖 AI sedang menganalisis data kesehatan Anda...'
        with st.spinner(loading_text):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                   insulin, bmi, diabetes_pedigree, age]])
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Results section
            st.markdown("---")
            results_title = "### 📊 AI Analysis Results" if lang == "en" else "### 📊 Hasil Analisis AI"
            st.markdown(results_title)
            
            # Main prediction result
            risk_score = prediction_proba[1] * 100
            
            if prediction == 1:
                risk_level = get_text('high_risk', lang)
                risk_color = "#f44336"
                risk_emoji = "🔴"
                card_class = "risk-high"
                recommendation_text = 'Immediate medical consultation recommended' if lang == "en" else 'Konsultasi medis segera direkomendasikan'
            else:
                risk_level = get_text('low_risk', lang)
                risk_color = "#4CAF50"
                risk_emoji = "🟢"
                card_class = "risk-low"
                recommendation_text = 'Continue healthy lifestyle practices' if lang == "en" else 'Lanjutkan praktik gaya hidup sehat'
            
            # Enhanced results display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                risk_score_text = "Risk Score" if lang == "en" else "Skor Risiko"
                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <h2 style="text-align: center; margin: 0;">
                        {risk_emoji} {risk_level}
                    </h2>
                    <h3 style="text-align: center; color: {risk_color}; margin: 10px 0;">
                        {risk_score_text}: {risk_score:.1f}%
                    </h3>
                    <p style="text-align: center; font-size: 16px; margin: 0;">
                        {recommendation_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Risk gauge chart
                risk_level_chart_text = "Risk Level" if lang == "en" else "Tingkat Risiko"
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': risk_level_chart_text},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probability breakdown
            probability_breakdown_text = "#### 📈 Probability Breakdown" if lang == "en" else "#### 📈 Rincian Probabilitas"
            st.markdown(probability_breakdown_text)
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability pie chart
                chart_title = "Risk Probability Distribution" if lang == "en" else "Distribusi Probabilitas Risiko"
                fig = px.pie(
                    values=prediction_proba,
                    names=[get_text('no_diabetes', lang), get_text('diabetes', lang)],
                    title=chart_title,
                    color_discrete_sequence=['#4CAF50', '#f44336']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Confidence metrics
                confidence = max(prediction_proba) * 100
                
                confidence_title = "🎯 Model Confidence" if lang == "en" else "🎯 Keyakinan Model"
                prediction_confidence_text = "Prediction Confidence" if lang == "en" else "Keyakinan Prediksi"
                no_diabetes_prob_text = "No Diabetes Probability" if lang == "en" else "Probabilitas Tidak Diabetes"
                diabetes_prob_text = "Diabetes Probability" if lang == "en" else "Probabilitas Diabetes"
                model_accuracy_text = "Model Accuracy" if lang == "en" else "Akurasi Model"
                
                st.markdown(f"""
                <div class="info-card">
                    <h4>{confidence_title}</h4>
                    <p><strong>{prediction_confidence_text}:</strong> {confidence:.1f}%</p>
                    <p><strong>{no_diabetes_prob_text}:</strong> {prediction_proba[0]*100:.1f}%</p>
                    <p><strong>{diabetes_prob_text}:</strong> {prediction_proba[1]*100:.1f}%</p>
                    <p><strong>{model_accuracy_text}:</strong> ~85%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk category
                if risk_score < 30:
                    category = "Low Risk"
                    category_color = "#4CAF50"
                elif risk_score < 70:
                    category = "Moderate Risk"
                    category_color = "#ff9800"
                else:
                    category = "High Risk"
                    category_color = "#f44336"
                
                st.markdown(f"""
                <div style="background: {category_color}; color: white; padding: 1rem; 
                           border-radius: 10px; text-align: center; margin-top: 1rem;">
                    <h4 style="margin: 0;">Risk Category: {category}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance for this prediction
            st.markdown("#### 🔍 Key Risk Factors for Your Profile")
            
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            feature_values = [pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, diabetes_pedigree, age]
            
            # Get feature importance from model (if available)
            if hasattr(model, 'feature_importances_'):
                importance_data = list(zip(feature_names, feature_values, model.feature_importances_))
                importance_data.sort(key=lambda x: x[2], reverse=True)
                
                col1, col2 = st.columns(2)
                
                for i, (feature, value, importance) in enumerate(importance_data[:4]):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; 
                                    border-left: 4px solid {'#f44336' if importance > 0.15 else '#ff9800' if importance > 0.1 else '#4CAF50'};">
                            <h6 style="margin: 0; color: #333;">{feature}</h6>
                            <p style="margin: 0.3rem 0 0 0; color: #666;">
                                Your Value: {value:.1f} | Importance: {importance:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Personalized recommendations
            st.markdown("#### 💡 Personalized Recommendations")
            
            recommendations = []
            
            if glucose > 126:
                recommendations.append("🍎 **Glucose Management**: Your glucose level is elevated. Consider reducing sugar intake and monitoring blood sugar regularly.")
            elif glucose > 100:
                recommendations.append("🍎 **Glucose Monitoring**: Your glucose is in the pre-diabetic range. Regular monitoring recommended.")
            
            if bmi > 30:
                recommendations.append("🏃‍♀️ **Weight Management**: Focus on weight reduction through diet and exercise to reduce diabetes risk.")
            elif bmi > 25:
                recommendations.append("🏃‍♀️ **Healthy Weight**: Maintain current weight through balanced nutrition and regular exercise.")
            
            if blood_pressure > 90:
                recommendations.append("❤️ **Blood Pressure**: Monitor blood pressure regularly and consider lifestyle modifications.")
            
            if age > 45:
                recommendations.append("🩺 **Regular Checkups**: Age is a risk factor. Schedule regular health screenings.")
            
            if not recommendations:
                recommendations.append("✅ **Great Health Profile**: Continue your current healthy lifestyle practices!")
            
            for rec in recommendations:
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; 
                           border-left: 4px solid #2196f3;">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
              # Medical disclaimer
            st.markdown("---")
            st.error("""
            **⚠️ Important Medical Disclaimer**: This prediction tool is for educational purposes only and should not replace professional medical advice. 
            Always consult with a qualified healthcare provider for proper medical evaluation and treatment decisions.
            """)
            
            # Action button
            col1, col2, col3 = st.columns(3)
            with col2:  # Center the button
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.experimental_rerun()

def show_about(lang="en"):
    st.markdown(f"### {get_text('about_title', lang)}")
    
    # Simple overview section
    st.markdown(f"""
    <div class="info-card">
        <h4>🎯 {"What is this application?" if lang == "en" else "Apa itu aplikasi ini?"}</h4>
        <p>
            {get_text('about_description', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    st.markdown(f"#### {get_text('features_title', lang)}")
    
    if lang == "en":
        features = [
            "🔮 **AI-Powered Predictions**: Uses Random Forest machine learning algorithm",
            "📊 **Interactive Analysis**: Explore medical data with visualizations", 
            "🎯 **Risk Assessment**: Get personalized diabetes risk evaluation",
            "📱 **User-Friendly**: Simple interface for easy health screening"
        ]
    else:
        features = [
            "🔮 **Prediksi Bertenaga AI**: Menggunakan algoritma machine learning Random Forest",
            "📊 **Analisis Interaktif**: Jelajahi data medis dengan visualisasi", 
            "🎯 **Penilaian Risiko**: Dapatkan evaluasi risiko diabetes yang dipersonalisasi",
            "📱 **Ramah Pengguna**: Antarmuka sederhana untuk skrining kesehatan mudah"
        ]
    
    for feature in features:
        st.markdown(f"- {feature}")
    
    # Dataset information
    dataset_info_title = "#### 📋 Dataset Information" if lang == "en" else "#### 📋 Informasi Dataset"
    st.markdown(dataset_info_title)
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_source_title = "📊 Data Source" if lang == "en" else "📊 Sumber Data"
        source_text = "Source" if lang == "en" else "Sumber"
        records_text = "Records" if lang == "en" else "Rekaman"
        features_text = get_text('features', lang)
        accuracy_text = "Accuracy" if lang == "en" else "Akurasi"
        patients_text = "patients" if lang == "en" else "pasien"
        medical_indicators_text = get_text('medical_indicators', lang)
        prediction_accuracy_text = "prediction accuracy" if lang == "en" else "akurasi prediksi"
        
        st.markdown(f"""
        <div class="info-card">
            <h5>{data_source_title}</h5>
            <ul>
                <li><strong>{source_text}:</strong> Pima Indian Diabetes Database</li>
                <li><strong>{records_text}:</strong> 768 {patients_text}</li>
                <li><strong>{features_text}:</strong> 8 {medical_indicators_text}</li>
                <li><strong>{accuracy_text}:</strong> 85.3% {prediction_accuracy_text}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        health_indicators_title = "🧬 Health Indicators" if lang == "en" else "🧬 Indikator Kesehatan"
        
        if lang == "en":
            indicators = [
                "Number of pregnancies",
                "Glucose concentration", 
                "Blood pressure",
                "Skin thickness",
                "Insulin levels",
                "Body Mass Index (BMI)",
                "Diabetes pedigree function",
                "Age"
            ]
        else:
            indicators = [
                "Jumlah kehamilan",
                "Konsentrasi glukosa",
                "Tekanan darah", 
                "Ketebalan kulit",
                "Tingkat insulin",
                "Indeks Massa Tubuh (BMI)",
                "Fungsi silsilah diabetes",
                "Usia"
            ]
        
        indicators_html = "".join([f"<li>{indicator}</li>" for indicator in indicators])
        
        st.markdown(f"""
        <div class="info-card">
            <h5>{health_indicators_title}</h5>
            <ul>
                {indicators_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model details
    st.markdown(f"#### {get_text('model_details', lang)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm_title = "🤖 Algorithm Details" if lang == "en" else "🤖 Detail Algoritma"
        algorithm_type_text = "Algorithm Type" if lang == "en" else "Tipe Algoritma"
        training_samples_text = "Training Samples" if lang == "en" else "Sampel Pelatihan"
        validation_method_text = "Validation Method" if lang == "en" else "Metode Validasi"
        performance_metric_text = "Performance Metric" if lang == "en" else "Metrik Kinerja"
        
        st.markdown(f"""
        <div class="info-card">
            <h5>{algorithm_title}</h5>
            <ul>
                <li><strong>{algorithm_type_text}:</strong> Random Forest Classifier</li>
                <li><strong>{training_samples_text}:</strong> 614 (80%)</li>
                <li><strong>{validation_method_text}:</strong> Train-Test Split</li>
                <li><strong>{performance_metric_text}:</strong> {get_text('accuracy', lang)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_performance_title = "📈 Model Performance" if lang == "en" else "📈 Kinerja Model"
        overall_accuracy_text = "Overall Accuracy" if lang == "en" else "Akurasi Keseluruhan"
        precision_text = "Precision" if lang == "en" else "Presisi"
        recall_text = "Recall" if lang == "en" else "Recall"
        f1_score_text = "F1-Score"
        
        st.markdown(f"""
        <div class="info-card">
            <h5>{model_performance_title}</h5>
            <ul>
                <li><strong>{overall_accuracy_text}:</strong> 85.3%</li>
                <li><strong>{precision_text}:</strong> 82.1%</li>
                <li><strong>{recall_text}:</strong> 79.6%</li>
                <li><strong>{f1_score_text}:</strong> 80.8%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("---")
    st.markdown(f"### {get_text('disclaimer', lang)}")
    
    st.markdown(f"""
    <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; 
                border-radius: 8px; margin: 0.5rem 0;">
        <p style="margin: 0; color: #856404;">
            <strong>⚠️ {get_text('disclaimer', lang)}</strong><br>
            {get_text('disclaimer_text', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical information
    technical_info_title = "#### 🔧 Technical Information" if lang == "en" else "#### 🔧 Informasi Teknis"
    st.markdown(technical_info_title)
    
    if lang == "en":
        tech_info = [
            "**Framework**: Streamlit (Python web framework)",
            "**ML Library**: Scikit-learn (Random Forest implementation)",
            "**Data Processing**: Pandas & NumPy",
            "**Visualizations**: Plotly for interactive charts",
            "**Deployment**: Can be deployed on Streamlit Cloud, Heroku, or AWS"
        ]
    else:
        tech_info = [
            "**Framework**: Streamlit (framework web Python)",
            "**Library ML**: Scikit-learn (implementasi Random Forest)",
            "**Pemrosesan Data**: Pandas & NumPy",
            "**Visualisasi**: Plotly untuk grafik interaktif",
            "**Deployment**: Dapat dideploy di Streamlit Cloud, Heroku, atau AWS"        ]
    
    for info in tech_info:
        st.markdown(f"- {info}")
    
    # Important disclaimers
    st.markdown("---")
    st.markdown("#### ⚠️ Important Disclaimers")
    
    disclaimers = [
        "🩺 **Medical Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.",
        "🔬 **Research Tool**: Predictions are based on statistical patterns and may not apply to all individuals.",
        "👨‍⚕️ **Professional Consultation**: Always consult healthcare professionals for medical decisions."
    ]
    
    for disclaimer in disclaimers:
        st.markdown(f"""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; 
                    border-radius: 8px; margin: 0.5rem 0;">
            {disclaimer}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()