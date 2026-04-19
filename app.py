import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# 1. Page Configuration & Premium Custom CSS
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Student Score Predictor App", page_icon="🎓", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* Apply font to overall app */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Dark mode base styling */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f1f5f9;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.8) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid #334155;
        }
        
        /* Card styling for main predict element */
        .prediction-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            margin-bottom: 2rem;
            transition: transform 0.2s ease-in-out;
        }
        .prediction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.2), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        /* Highlights and Gradients */
        .highlight-text {
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ------------------------------------------------------------------------------
# 2. Load Model & Data
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model_data():
    with open('rf_model_pipeline.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model_data()
pipeline = model_data['pipeline']
y_test = model_data['y_test']
y_pred_test = model_data['y_pred']
feature_names = model_data['feature_names']

# ------------------------------------------------------------------------------
# 3. Sidebar Input Form
# ------------------------------------------------------------------------------
st.sidebar.markdown(f"## 🎓 Configure Student")

gender = st.sidebar.selectbox("Gender", options=['female', 'male'])
race = st.sidebar.selectbox("Race/Ethnicity", options=['group A', 'group B', 'group C', 'group D', 'group E'])
education = st.sidebar.selectbox(
    "Parental Level of Education", 
    options=['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.sidebar.selectbox("Lunch", options=['standard', 'free/reduced'])
test_prep = st.sidebar.selectbox("Test Preparation Course", options=['none', 'completed'])

st.sidebar.markdown('---')
st.sidebar.markdown("### Scores")

reading_score = st.sidebar.slider("Reading Score", min_value=0, max_value=100, value=75, step=1)
writing_score = st.sidebar.slider("Writing Score", min_value=0, max_value=100, value=75, step=1)

# Construct a dataframe from input
input_data = pd.DataFrame({
    'gender': [gender],
    'race/ethnicity': [race],
    'parental level of education': [education],
    'lunch': [lunch],
    'test preparation course': [test_prep],
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# ------------------------------------------------------------------------------
# 4. Main Application UI & Predictions
# ------------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>AI Math Score <span class='highlight-text'>Predictor</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Powered by Random Forest Regression<br><em>Systematically evaluate model performance through hyperparameter optimization</em></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><a href='https://www.kaggle.com/datasets/spscientist/students-performance-in-exams' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: bold;'>📂 View Kaggle Dataset</a> &nbsp;&nbsp;|&nbsp;&nbsp; <a href='https://github.com/' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: bold;'>💻 GitHub Repository</a></p>", unsafe_allow_html=True)

# Create layout with Tabs for better UX
tab1, tab2 = st.tabs(["🚀 Live Prediction & Model Analytics", "📈 Exploratory Data Analysis (EDA)"])

with tab1:


    # Make Prediction
    prediction = pipeline.predict(input_data)[0]

    # UI for presenting prediction
    st.markdown(f"""
        <div class="prediction-card">
            <h2 style='margin:0; font-size: 1.5rem; color: #cbd5e1;'>Live Predicted Math Score</h2>
            <h1 style='margin:10px 0 0 0; font-size: 4rem; color: #10b981;'>{prediction:.1f} <span style="font-size: 1.5rem; color: #64748b;">/ 100</span></h1>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------------------
    # 5. Dashboard Visualizations (Plotly) & Metrics
    # ------------------------------------------------------------------------------
    st.markdown("### 📊 Model Evaluation & Metrics")
    st.markdown("---")
    
    # Calculate Regression Metrics
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calculate Classification Metrics (Threshold: 50 for Pass/Fail)
    pass_threshold = 50
    actual_pass = (y_test >= pass_threshold).astype(int)
    pred_pass = (y_pred_test >= pass_threshold).astype(int)
    
    acc = accuracy_score(actual_pass, pred_pass)
    prec = precision_score(actual_pass, pred_pass, zero_division=0)
    rec = recall_score(actual_pass, pred_pass, zero_division=0)
    f1 = f1_score(actual_pass, pred_pass, zero_division=0)

    # Display Metrics in Columns
    st.markdown("#### Regression Metrics (Predicting Exact Score)")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score", f"{r2:.3f}")
    mc2.metric("Mean Absolute Error", f"{mae:.2f}")
    mc3.metric("Root Mean Squared Error", f"{rmse:.2f}")
    
    st.markdown("#### Classification Metrics (Predicting Pass: Score ≥ 50)")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Accuracy", f"{acc:.1%}")
    cc2.metric("Precision", f"{prec:.1%}")
    cc3.metric("Recall", f"{rec:.1%}")
    cc4.metric("F1-Score", f"{f1:.1%}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    # --- Chart 1: Feature Importances ---
    with col1:
        st.markdown("#### Feature Importances")
        rf_model = pipeline.named_steps['model']
        importances = rf_model.feature_importances_
        
        # Map importances to feature names
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=True) # Ascending for horizontal bar
        
        fig1 = px.bar(imp_df, x='Importance', y='Feature', orientation='h', template='plotly_dark')
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Importance Weight",
            yaxis_title=""
        )
        # Give it a nice color gradient based on importance
        fig1.update_traces(marker=dict(color=imp_df['Importance'], colorscale='Blugrn'))
        st.plotly_chart(fig1, width='stretch')

    # --- Chart 2: Residuals Distribution ---
    with col2:
        st.markdown("#### Residuals (Error) Distribution")
        residuals = y_test - y_pred_test
        
        fig2 = px.histogram(
            x=residuals, 
            nbins=40, 
            template='plotly_dark',
            labels={'x': 'Prediction Error (Actual - Predicted)'}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Count",
            showlegend=False
        )
        fig2.update_traces(marker=dict(color='#f43f5e', line=dict(color='#881337', width=1)))
        st.plotly_chart(fig2, width='stretch')


with tab2:
    st.markdown("### 📈 Exploratory Data Analysis")
    st.markdown("Explore the original dataset distributions and relationships.")
    st.markdown("---")
    
    try:
        df = pd.read_csv('StudentsPerformance.csv')
        
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### Demographics: Race/Ethnicity")
            fig_race = px.pie(df, names='race/ethnicity', hole=0.4, template='plotly_dark', color_discrete_sequence=px.colors.sequential.Sunset)
            fig_race.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_race, width='stretch')
            
        with ec2:
            st.markdown("#### Demographics: Parental Education")
            fig_edu = px.pie(df, names='parental level of education', hole=0.4, template='plotly_dark', color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig_edu.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_edu, width='stretch')
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Score Distributions by Gender")
        
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            fig_v1 = px.violin(df, x='gender', y='math score', color='gender', box=True, template='plotly_dark')
            st.plotly_chart(fig_v1, width='stretch')
        with vc2:
            fig_v2 = px.violin(df, x='gender', y='reading score', color='gender', box=True, template='plotly_dark')
            st.plotly_chart(fig_v2, width='stretch')
        with vc3:
            fig_v3 = px.violin(df, x='gender', y='writing score', color='gender', box=True, template='plotly_dark')
            st.plotly_chart(fig_v3, width='stretch')
            
    except Exception as e:
        st.error(f"Could not load StudentsPerformance.csv for EDA: {e}")
