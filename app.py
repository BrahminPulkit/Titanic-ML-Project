# Titanic Streamlit Dashboard - app.py
# Fixed version with error corrections

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import os
import base64
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF

# Optional: SHAP (explainability). If not installed, app will skip SHAP section gracefully.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Optional: enable saving plotly static images using kaleido
KALEIDO_AVAILABLE = True
try:
    import kaleido
except Exception:
    KALEIDO_AVAILABLE = False

# Optional: joblib for model saving
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# -------------------------
# CONFIGURATION CLASS
# -------------------------
class Config:
    DATA_PATH = r"E:/Python Practice/Titanic-Machine Learning Disaster/titanic/train.csv"
    MODEL_PATH = r"E:/Python Practice/Titanic-Machine Learning Disaster/titanic/model_rf.pkl"
    PRIMARY_COLOR = "#0ea5a4"
    SECONDARY_COLOR = "#111827"
    CHART_HEIGHT = 400
    
    @classmethod
    def check_paths(cls):
        """Check if required paths exist"""
        if not os.path.exists(cls.DATA_PATH):
            st.warning(f"Data file not found at: {cls.DATA_PATH}")
            return False
        # Create directory for model if it doesn't exist
        model_dir = os.path.dirname(cls.MODEL_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        return True

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(
    page_title="Titanic — Pro Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🚢"
)

# -------------------------
# ENHANCED CSS / THEME
# -------------------------
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
    }}
    .section-title {{
        font-size: 22px;
        color: {Config.PRIMARY_COLOR};
        font-weight: 700;
        margin-bottom: 6px;
        border-bottom: 2px solid {Config.PRIMARY_COLOR};
        padding-bottom: 5px;
    }}
    .card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(2,6,23,0.6);
        border: 1px solid rgba(14, 165, 164, 0.2);
        margin-bottom: 15px;
    }}
    .kpi {{
        font-size: 20px;
        color: #fff;
        font-weight: bold;
    }}
    .small-muted {{
        color: #9ca3af;
        font-size: 12px;
    }}
    .success-box {{
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    .warning-box {{
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    .stProgress > div > div > div > div {{
        background-color: {Config.PRIMARY_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# ENHANCED HELPERS
# -------------------------
@st.cache_data
def load_data(path):
    """Robust data loading with multiple fallbacks"""
    try:
        df = pd.read_csv(path)
        st.sidebar.success("✓ Data loaded successfully")
        return df
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        # Provide sample data as fallback
        st.warning("Using sample data for demonstration")
        sample_data = {
            'PassengerId': range(1, 101),
            'Survived': np.random.randint(0, 2, 100),
            'Pclass': np.random.randint(1, 4, 100),
            'Name': [f'Passenger {i}' for i in range(1, 101)],
            'Sex': np.random.choice(['male', 'female'], 100),
            'Age': np.random.normal(30, 10, 100).clip(0, 80),
            'SibSp': np.random.randint(0, 4, 100),
            'Parch': np.random.randint(0, 3, 100),
            'Ticket': [f'Ticket_{i}' for i in range(1, 101)],
            'Fare': np.random.exponential(50, 100).clip(0, 500),
            'Cabin': [f'{deck}{num}' for deck in ['A', 'B', 'C', 'D', 'E'] for num in range(1, 21)][:100],
            'Embarked': np.random.choice(['S', 'C', 'Q'], 100)
        }
        return pd.DataFrame(sample_data)

@st.cache_data
def preprocess(df):
    """Enhanced preprocessing with additional features"""
    data = df.copy()
    
    # Name processing
    if 'Name' in data.columns:
        data['Title'] = data['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).fillna('Other')
        data['Surname'] = data['Name'].str.split(',').str[0].fillna('Unknown')
    
    # Family features
    data['Family_size'] = data.get('SibSp', 0) + data.get('Parch', 0) + 1
    data['Is_alone'] = (data['Family_size'] == 1).astype(int)
    
    # Enhanced family type classification
    def fam_type(x):
        if x == 1:
            return 'Single'
        elif x <= 3:
            return 'Small'
        elif x <= 5:
            return 'Medium'
        else:
            return 'Large'
    data['Family_type'] = data['Family_size'].apply(fam_type)
    
    # Cabin/Deck processing - FIXED: Convert to string first
    if 'Cabin' in data.columns:
        data['Deck'] = data['Cabin'].astype(str).str[0]
        data['Deck'] = data['Deck'].replace('n', np.nan)  # Handle 'nan' strings
        data['Has_cabin'] = data['Cabin'].notna().astype(int)
    else:
        data['Deck'] = np.nan
        data['Has_cabin'] = 0
    
    # Enhanced age processing
    if 'Age' in data.columns:
        data['Age_filled'] = data['Age'].fillna(data['Age'].median())
        # Convert to regular categories to avoid Categorical issues
        age_bins = [0, 12, 18, 35, 60, 100]
        age_labels = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
        data['Age_group'] = pd.cut(data['Age_filled'], bins=age_bins, labels=age_labels)
        data['Age_group'] = data['Age_group'].astype(str)  # Convert to string to avoid categorical issues
    else:
        data['Age_filled'] = 30
        data['Age_group'] = 'Adult'
    
    # Enhanced fare processing
    if 'Fare' in data.columns:
        data['Fare_per_person'] = data['Fare'] / data['Family_size']
        # Convert to regular categories
        try:
            data['Fare_group'] = pd.qcut(data['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
            data['Fare_group'] = data['Fare_group'].astype(str)  # Convert to string
        except ValueError:
            # Fallback if qcut fails
            data['Fare_group'] = 'Medium'
    else:
        data['Fare_per_person'] = 0
        data['Fare_group'] = 'Low'
    
    # Embarked with better handling - convert to string
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna('S').astype(str)  # Most common value
    
    # Convert other categorical columns to string to avoid issues
    categorical_cols = ['Title', 'Deck', 'Family_type']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    return data

@st.cache_data
def prepare_model_data(df):
    """Enhanced feature preparation for ML - FIXED categorical handling"""
    data = df.copy()
    
    # Base features
    features = ['Pclass', 'Sex', 'Age_filled', 'Fare', 'Family_size', 'Is_alone', 'Has_cabin']
    X = data[features].copy()
    
    # Encode categorical variables
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1}).fillna(0).astype(int)
    
    # One-hot encoding for categorical variables - FIXED: Handle categorical conversion
    for col in ['Title', 'Deck', 'Embarked', 'Age_group', 'Fare_group']:
        if col in data.columns:
            # Convert to string and fill NaN safely
            col_data = data[col].astype(str).fillna('Unknown')
            dummies = pd.get_dummies(col_data, prefix=col)
            X = pd.concat([X, dummies], axis=1)
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Target variable
    y = data['Survived'] if 'Survived' in data.columns else None
    
    return X, y

def get_model(X, y, force_retrain=False):
    """Enhanced model management with progress tracking"""
    model = None
    
    # Try to load existing model
    if not force_retrain and JOBLIB_AVAILABLE and os.path.exists(Config.MODEL_PATH):
        try:
            with st.spinner("Loading pre-trained model..."):
                model = joblib.load(Config.MODEL_PATH)
            st.sidebar.success("✓ Loaded pre-trained model")
        except Exception as e:
            st.sidebar.warning(f"Could not load saved model: {e}")
    
    # Train new model if needed
    if model is None:
        with st.spinner("Training Random Forest model... This may take a few seconds."):
            # Progress bar simulation
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate training time
                progress_bar.progress(i + 1)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Save model if joblib is available
            if JOBLIB_AVAILABLE:
                try:
                    joblib.dump(model, Config.MODEL_PATH)
                    st.sidebar.success("✓ Model trained and saved")
                except Exception as e:
                    st.sidebar.info("Model trained but not saved (save failed)")
            else:
                st.sidebar.info("Model trained but not saved (joblib unavailable)")
            
            progress_bar.empty()
    
    return model

def explain_prediction(model, row, feature_names):
    """Enhanced SHAP explanation with better error handling"""
    if not SHAP_AVAILABLE:
        return None
        
    try:
        with st.spinner("Computing SHAP explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(row)
            
            # Handle SHAP values format
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]  # For class 1 (survived)
            else:
                shap_vals = shap_values
                
            # Create feature importance plot
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_vals[0])
            }).sort_values('importance', ascending=True).tail(15)
            
            fig = px.bar(importance_df, x='importance', y='feature', 
                        orientation='h', 
                        title='Feature Impact on Prediction',
                        color='importance',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            return fig
            
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return None

def fig_to_image_bytes(fig, format='png'):
    """Enhanced image conversion with multiple fallbacks"""
    if KALEIDO_AVAILABLE:
        try:
            img_bytes = fig.to_image(format=format, engine='kaleido', width=800, height=600)
            return img_bytes
        except Exception as e:
            st.warning(f"Kaleido conversion failed: {e}")
    
    # Fallback: try different methods
    try:
        # Convert to HTML as last resort
        html = fig.to_html(include_plotlyjs='cdn')
        return html.encode('utf-8')
    except Exception as e:
        st.error(f"All image conversion methods failed: {e}")
        return None

def generate_pdf_report(title, insights_text, figs, out_path='titanic_report.pdf'):
    """Enhanced PDF generation with better error handling"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(10)
    
    # Insights
    pdf.set_font('Arial', '', 12)
    for line in insights_text.split('\n'):
        pdf.multi_cell(0, 8, line)
    pdf.ln(10)
    
    # Figures with enhanced error handling
    for i, fig in enumerate(figs):
        try:
            img_bytes = fig_to_image_bytes(fig, format='png')
            if img_bytes and isinstance(img_bytes, bytes):
                img_path = f'temp_fig_{i}.png'
                with open(img_path, 'wb') as f:
                    f.write(img_bytes)
                pdf.image(img_path, w=180, h=135)  # Adjusted size for better fit
                os.remove(img_path)
                pdf.ln(5)
            else:
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 8, f'Figure {i+1}: Could not render image', ln=True)
                
        except Exception as e:
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 8, f'Figure {i+1}: Error - {str(e)[:50]}...', ln=True)
    
    try:
        pdf.output(out_path)
        return out_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None


# DATA LOADING & INITIALIZATION

# Check paths and load data
if not Config.check_paths():
    st.warning("Using sample data due to path issues")

try:
    raw = load_data(Config.DATA_PATH)
except Exception as e:
    st.error(f"Critical error loading data: {e}")
    st.stop()

data = preprocess(raw)


# SIDEBAR CONTROLS

st.sidebar.header('🚢 Titanic Dashboard')
st.sidebar.markdown("---")

# Filters
st.sidebar.subheader('Data Filters')

sex_vals = data['Sex'].unique().tolist()
selected_sex = st.sidebar.multiselect('Sex', options=sex_vals, default=sex_vals)

pclass_vals = sorted(data['Pclass'].dropna().unique().tolist())
selected_pclass = st.sidebar.multiselect('Passenger Class', options=pclass_vals, default=pclass_vals)

embark_vals = data['Embarked'].fillna('S').unique().tolist()
selected_embark = st.sidebar.multiselect('Embarkation Port', options=embark_vals, default=embark_vals)

# Enhanced age filter with groups
age_min = int(data['Age_filled'].min())
age_max = int(data['Age_filled'].max())
selected_age = st.sidebar.slider('Age Range', min_value=age_min, max_value=age_max, 
                               value=(age_min, age_max), help="Filter passengers by age range")

fare_min = float(data['Fare'].min())
fare_max = float(data['Fare'].max())
selected_fare = st.sidebar.slider('Fare Range', fare_min, fare_max, (fare_min, fare_max),
                                help="Filter passengers by fare paid")

# Title filter
if 'Title' in data.columns:
    title_vals = data['Title'].unique().tolist()
    selected_titles = st.sidebar.multiselect('Titles', options=title_vals, default=title_vals)

st.sidebar.markdown("---")

# Model settings
st.sidebar.subheader('Model Settings')
train_model_now = st.sidebar.checkbox('Force retrain model', value=False, 
                                    help="Retrain the model (slower but uses latest data)")
show_shap = st.sidebar.checkbox('Enable SHAP explanations', value=False and SHAP_AVAILABLE, 
                              disabled=not SHAP_AVAILABLE,
                              help="Show feature importance explanations")

if show_shap and not SHAP_AVAILABLE:
    st.sidebar.warning("SHAP not installed. Run: pip install shap")

# Apply filters
filter_mask = (
    (data['Sex'].isin(selected_sex)) &
    (data['Pclass'].isin(selected_pclass)) &
    (data['Embarked'].isin(selected_embark)) &
    (data['Age_filled'].between(selected_age[0], selected_age[1])) &
    (data['Fare'].between(selected_fare[0], selected_fare[1]))
)

if 'Title' in data.columns and 'selected_titles' in locals():
    filter_mask = filter_mask & (data['Title'].isin(selected_titles))

filtered = data[filter_mask]

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'EDA', 'Prediction', 'Model Insights', 'Report & Download'])


# MAIN CONTENT

st.title("🚢 Titanic — Professional Dashboard")
st.markdown("***Analyze passenger survival patterns with interactive visualizations and machine learning***")

# Top banner / historical context
with st.expander('📜 Historical Context (click to expand)', expanded=False):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **RMS Titanic — April 15, 1912**  
        The RMS Titanic sank in the North Atlantic Ocean after hitting an iceberg on her maiden voyage. 
        This dashboard analyzes passenger data to explore which factors were associated with survival.
        
        - **Total passengers**: 2,224 (estimated)
        - **Survivors**: 706 (31.6%)
        - **Casualties**: 1,517 (68.4%)
        """)
    with col2:
        st.markdown("""
        **Key Factors:**
        - Women and children first
        - Higher class advantage
        - Port of embarkation
        """)

# OVERVIEW PAGE

if page == 'Overview':
    st.markdown('<div class="section-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Key metrics
    total = len(filtered)
    survivors = int(filtered['Survived'].sum()) if 'Survived' in filtered.columns else 0
    survival_rate = round(survivors/total*100, 2) if total > 0 else 0
    avg_fare = round(filtered['Fare'].mean() if total > 0 else 0, 2)
    avg_age = round(filtered['Age_filled'].mean() if total > 0 else 0, 1)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="card"><div class="kpi" style="color:#FF9800;">{}</div><div class="small-muted">Total Passengers</div></div>'.format(total), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div class="kpi" style="color:#9C27B0;">{}</div><div class="small-muted">Survivors</div></div>'.format(survivors), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="kpi" style="color:#FF5722;">{}%</div><div class="small-muted">Survival Rate</div></div>'.format(survival_rate), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card"><div class="kpi" style="color:#2196F3;">£{}</div><div class="small-muted">Avg Fare</div></div>'.format(avg_fare), unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="card"><div class="kpi" style="color:#4CAF50;">{} yrs</div><div class="small-muted">Avg Age</div></div>'.format(avg_age), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Survival Distribution")
        if total > 0:
            fig_pie = px.pie(filtered, names='Survived', hole=0.4, 
                           title='Survival Distribution',
                           color='Survived',
                           color_discrete_map={0: '#ef4444', 1: '#10b981'})
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with col2:
        st.subheader("Survival by Passenger Class")
        if total > 0:
            class_survival = filtered.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
            fig_class = px.bar(class_survival, x='Pclass', y='Count', color='Survived',
                             barmode='group', title='Survival by Class',
                             color_discrete_map={0: '#ef4444', 1: '#10b981'})
            st.plotly_chart(fig_class, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        if total > 0:
            fig_age = px.histogram(filtered, x='Age_filled', nbins=30, color='Survived',
                                 title='Age Distribution by Survival',
                                 color_discrete_map={0: '#ef4444', 1: '#10b981'})
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with col2:
        st.subheader("Fare Distribution")
        if total > 0:
            fig_fare = px.box(filtered, y='Fare', color='Survived',
                            title='Fare Distribution by Survival',
                            color_discrete_map={0: '#ef4444', 1: '#10b981'})
            st.plotly_chart(fig_fare, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    # Embarkation map
    st.markdown("---")
    st.subheader("🌍 Embarkation Ports")
    
    coords = {
        'C': (49.6333, -1.6167),  
        'Q': (51.8493, -8.2947),  
        'S': (50.9040, -1.4044),  
        'U': (51.0, -1.5)         
    }
    
    emb = filtered.groupby('Embarked').agg({'Survived': ['sum', 'count']}).reset_index()
    emb.columns = ['Embarked', 'Survived_sum', 'Total']
    emb['Survival_rate'] = emb['Survived_sum'] / emb['Total']
    emb['lat'] = emb['Embarked'].map(lambda x: coords.get(x, (None, None))[0])
    emb['lon'] = emb['Embarked'].map(lambda x: coords.get(x, (None, None))[1])
    emb = emb.dropna(subset=['lat'])
    
    if not emb.empty:
        fig_map = px.scatter_geo(emb, lat='lat', lon='lon', size='Total',
                               color='Survival_rate', hover_name='Embarked',
                               hover_data={'Survived_sum': True, 'Total': True, 'Survival_rate': ':.2%'},
                               title='Embarkation Ports and Survival Rates',
                               color_continuous_scale='viridis')
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Deck schematic
    st.markdown("---")
    st.subheader("🚢 Ship Deck Layout")
    
    if 'Deck' in filtered.columns:
        deck_summary = filtered.groupby('Deck').agg({'Survived': ['sum', 'count']}).reset_index()
        deck_summary.columns = ['Deck', 'Survived_sum', 'Total']
        deck_summary['Survival_rate'] = deck_summary['Survived_sum'] / deck_summary['Total']
        deck_summary = deck_summary.sort_values('Deck', na_position='last')
        
        fig_deck = go.Figure()
        fig_deck.add_trace(go.Bar(x=deck_summary['Deck'], y=deck_summary['Total'],
                                name='Total Passengers', marker_color='lightgrey'))
        fig_deck.add_trace(go.Bar(x=deck_summary['Deck'], y=deck_summary['Survived_sum'],
                                name='Survived', marker_color=Config.PRIMARY_COLOR))
        fig_deck.update_layout(barmode='overlay',
                             title='Passenger Count and Survival by Deck (A=Top, G=Bottom)',
                             xaxis_title='Deck', yaxis_title='Count')
        st.plotly_chart(fig_deck, use_container_width=True)


# EDA PAGE

elif page == 'EDA':
    st.markdown('<div class="section-title">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if len(filtered) == 0:
        st.warning("No data available for selected filters")
        st.stop()
    
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    num_cols = ['Survived', 'Pclass', 'Age_filled', 'Fare', 'Family_size', 'SibSp', 'Parch']
    num_present = [c for c in num_cols if c in filtered.columns]
    
    if num_present:
        corr = filtered[num_present].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                           title='Numeric Feature Correlation Matrix',
                           color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Family analysis
    st.markdown("---")
    st.subheader("Family Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Family size distribution
        fam_size_dist = filtered['Family_size'].value_counts().sort_index()
        fig_fam_size = px.bar(x=fam_size_dist.index, y=fam_size_dist.values,
                            title='Family Size Distribution',
                            labels={'x': 'Family Size', 'y': 'Count'})
        st.plotly_chart(fig_fam_size, use_container_width=True)
    
    with col2:
        # Family type survival - FIXED: Handle percentage formatting properly
        if 'Family_type' in filtered.columns:
            fam_survival = filtered.groupby('Family_type')['Survived'].mean().reset_index()
            # Convert to percentage for display
            fam_survival['Survival_Rate_Pct'] = fam_survival['Survived'] * 100
            fig_fam_survival = px.bar(fam_survival, x='Family_type', y='Survival_Rate_Pct',
                                    title='Survival Rate by Family Type',
                                    labels={'Survival_Rate_Pct': 'Survival Rate (%)', 'Family_type': 'Family Type'})
            st.plotly_chart(fig_fam_survival, use_container_width=True)
    
    # Title and demographic analysis
    st.markdown("---")
    st.subheader("Demographic Analysis")
    
    if 'Title' in filtered.columns:
        title_analysis = filtered.groupby('Title').agg({
            'Survived': ['count', 'mean'],
            'Age_filled': 'mean',
            'Fare': 'mean'
        }).round(2).reset_index()
        
        title_analysis.columns = ['Title', 'Count', 'Survival_Rate', 'Avg_Age', 'Avg_Fare']
        title_analysis = title_analysis.sort_values('Count', ascending=False)
        
        fig_title = px.bar(title_analysis.head(10), x='Title', y='Count',
                         color='Survival_Rate', title='Top 10 Titles with Survival Rates',
                         color_continuous_scale='viridis')
        fig_title.update_layout(showlegend=False)
        st.plotly_chart(fig_title, use_container_width=True)
    
    # Advanced visualizations
    st.markdown("---")
    st.subheader("Advanced Visualizations")
    
    # Treemap of families
    if 'Surname' in filtered.columns:
        st.write("Family Treemap (Top 20 Families)")
        surname_counts = filtered['Surname'].value_counts().head(20).reset_index()
        surname_counts.columns = ['Surname', 'Count']
        treedf = filtered[filtered['Surname'].isin(surname_counts['Surname'])]
        
        if not treedf.empty:
            fig_tree = px.treemap(treedf, path=['Surname', 'Title'], values='Family_size',
                                color='Survived', title='Family Structure and Survival',
                                color_continuous_scale='RdBu')
            st.plotly_chart(fig_tree, use_container_width=True)


# PREDICTION PAGE

elif page == 'Prediction':
    st.markdown('<div class="section-title">🤖 Survival Prediction</div>', unsafe_allow_html=True)
    
    # Prepare data and get model
    X, y = prepare_model_data(data)
    
    if y is None:
        st.error("No survival data available for modeling")
        st.stop()
    
    model = get_model(X, y, train_model_now)
    
    # Model performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Model Accuracy', f"{acc:.1%}")
    with col2:
        st.metric('Training Samples', len(X_train))
    with col3:
        st.metric('Test Samples', len(X_test))
    
    st.markdown("---")
    
    # Prediction form
    st.subheader("Predict Survival for New Passenger")
    
    with st.form('prediction_form'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            in_pclass = st.selectbox('Passenger Class', [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
            in_sex = st.selectbox('Sex', ['male', 'female'])
            in_age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        
        with col2:
            in_fare = st.number_input('Fare (£)', min_value=0.0, max_value=1000.0, 
                                    value=float(data['Fare'].median()), step=1.0)
            in_family = st.number_input('Family Size', min_value=1, max_value=15, value=1)
            in_embarked = st.selectbox('Embarked', options=sorted(data['Embarked'].fillna('S').unique()))
        
        with col3:
            in_title = st.selectbox('Title', options=sorted(data['Title'].unique()))
            in_deck = st.selectbox('Deck', options=sorted(data['Deck'].fillna('U').unique()))
            in_has_cabin = st.selectbox('Has Cabin', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        
        submitted = st.form_submit_button('Predict Survival', use_container_width=True)
    
    if submitted:
        # Create input row
        row_dict = {
            'Pclass': in_pclass,
            'Sex': 0 if in_sex == 'male' else 1,
            'Age_filled': in_age,
            'Fare': in_fare,
            'Family_size': in_family,
            'Is_alone': 1 if in_family == 1 else 0,
            'Has_cabin': in_has_cabin
        }
        
        row = pd.DataFrame([row_dict])
        
        # Add one-hot encoded features
        for col in ['Title', 'Deck', 'Embarked']:
            if col in data.columns:
                value = in_title if col == 'Title' else in_deck if col == 'Deck' else in_embarked
                dummies = pd.get_dummies(pd.Series([value]), prefix=col)
                for dummy_col in dummies.columns:
                    row[dummy_col] = dummies.iloc[0][dummy_col] if dummy_col in dummies else 0
        
        # Ensure all columns are present
        for col in X.columns:
            if col not in row.columns:
                row[col] = 0
        
        row = row[X.columns]
        
        # Make prediction
        proba = model.predict_proba(row)[0][1]
        pred = model.predict(row)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if pred == 1:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success(f"**Prediction: SURVIVED**")
                st.markdown(f"**Probability: {proba:.1%}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.error(f"**Prediction: DID NOT SURVIVE**")
                st.markdown(f"**Probability: {proba:.1%}**")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col2:
            # Feature importance explanation
            if show_shap and SHAP_AVAILABLE:
                shap_fig = explain_prediction(model, row, X.columns)
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True)
            elif show_shap:
                st.warning("SHAP not available. Install with: pip install shap")


# MODEL INSIGHTS PAGE

elif page == 'Model Insights':
    st.markdown('<div class="section-title">📈 Model Performance & Insights</div>', unsafe_allow_html=True)
    
    # Prepare data and train model
    X, y = prepare_model_data(data)
    
    if y is None:
        st.error("No survival data available for modeling")
        st.stop()
    
    model = get_model(X, y, train_model_now)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{acc:.1%}")
    with col2:
        precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
        st.metric("Precision", f"{precision:.1%}")
    with col3:
        recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
        st.metric("Recall", f"{recall:.1%}")
    with col4:
        f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
        st.metric("F1-Score", f"{f1:.1%}")
    
    # Confusion Matrix
    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, 
                      x=['Predicted Not Survived', 'Predicted Survived'],
                      y=['Actual Not Survived', 'Actual Survived'],
                      title='Confusion Matrix',
                      color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # ROC Curve
    st.markdown("---")
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                               name=f'Random Forest (AUC = {roc_auc:.3f})',
                               line=dict(color=Config.PRIMARY_COLOR, width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                               line=dict(dash='dash', color='gray'),
                               showlegend=False))
    fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("Feature Importance")
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    fig_fi = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                   title='Top 20 Most Important Features',
                   color='importance', color_continuous_scale='viridis')
    fig_fi.update_layout(showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)
    
    # SHAP Summary (if available)
    if SHAP_AVAILABLE and show_shap:
        st.markdown("---")
        st.subheader("SHAP Global Explanations")
        
        with st.spinner("Computing SHAP values (this may take a while)..."):
            try:
                # Sample data for faster computation
                X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Create summary plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")


# REPORT & DOWNLOAD PAGE

elif page == 'Report & Download':
    st.markdown('<div class="section-title">📄 Report Generation & Data Export</div>', unsafe_allow_html=True)
    
    if len(filtered) == 0:
        st.warning("No data available for selected filters")
        st.stop()
    
    # Insights generation
    st.subheader("Data Insights")
    
    total = len(filtered)
    survivors = int(filtered['Survived'].sum()) if 'Survived' in filtered.columns else 0
    survival_rate = round(survivors/total*100, 2) if total > 0 else 0
    
    insights = [
        f"Total passengers in selection: {total}",
        f"Survivors: {survivors}",
        f"Overall survival rate: {survival_rate}%",
        f"Average age: {round(filtered['Age_filled'].mean(), 1)} years",
        f"Average fare: £{round(filtered['Fare'].mean(), 2)}",
        f"Most common passenger class: {filtered['Pclass'].mode().iloc[0] if not filtered.empty else 'N/A'}"
    ]
    
    if 'Title' in filtered.columns:
        top_titles = filtered['Title'].value_counts().head(3).to_dict()
        insights.append("Most common titles: " + ", ".join([f"{k} ({v})" for k, v in top_titles.items()]))
    
    if 'Embarked' in filtered.columns:
        top_embark = filtered['Embarked'].value_counts().head(1).index[0]
        embark_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
        insights.append(f"Most common embarkation: {embark_names.get(top_embark, top_embark)}")
    
    # Display insights
    for insight in insights:
        st.write(f"• {insight}")
    
    # PDF Report Generation
    st.markdown("---")
    st.subheader("PDF Report Generation")
    
    if st.button("Generate Comprehensive PDF Report", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            # Create figures for PDF
            pdf_figures = []
            
            try:
                # Survival pie chart
                fig1 = px.pie(filtered, names='Survived', hole=0.4, 
                            title='Survival Distribution')
                pdf_figures.append(fig1)
                
                # Class survival
                class_data = filtered.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
                fig2 = px.bar(class_data, x='Pclass', y='Count', color='Survived',
                            barmode='group', title='Survival by Passenger Class')
                pdf_figures.append(fig2)
                
                # Age distribution
                fig3 = px.histogram(filtered, x='Age_filled', nbins=20, color='Survived',
                                  title='Age Distribution by Survival')
                pdf_figures.append(fig3)
                
                # Generate PDF
                insights_text = "\n".join(insights)
                pdf_path = generate_pdf_report("Titanic Analysis Report", insights_text, pdf_figures)
                
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    # Create download link
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="titanic_analysis_report.pdf" style="background-color: {Config.PRIMARY_COLOR}; color: white; padding: 12px 24px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px;">📥 Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("PDF report generated successfully!")
                    
                    # Clean up
                    try:
                        os.remove(pdf_path)
                    except:
                        pass
                else:
                    st.error("Failed to generate PDF report")
                    
            except Exception as e:
                st.error(f"Report generation failed: {e}")
    
    # Data Export
    st.markdown("---")
    st.subheader("Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="titanic_filtered_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON Export
        json_str = filtered.to_json(orient='records', indent=2)
        st.download_button(
            label="Download Filtered Data (JSON)",
            data=json_str,
            file_name="titanic_filtered_data.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Data preview
    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(filtered.head(10), use_container_width=True)
    
    st.write(f"Showing 10 of {len(filtered)} rows")
    st.write(f"Columns: {', '.join(filtered.columns.tolist())}")


# FOOTER

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**Built with:** Streamlit, Pandas, Scikit-learn, Plotly")

with footer_col2:
    st.markdown("**Optional dependencies:** SHAP, Kaleido, Joblib")

with footer_col3:
    st.markdown("**Data Source:** Kaggle Titanic Dataset")

st.markdown("*For educational and analytical purposes*")