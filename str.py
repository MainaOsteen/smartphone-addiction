import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Smartphone Addiction Prediction", page_icon="📱", layout="wide")

st.title(" 📱 Screen Dependence Dashboard")
with st.expander("**ℹ️    About this app**", expanded=False):
    st.write(
        """
        This app predicts the likelihood of smartphone addiction based on user input features.
        The model is trained on a dataset of smartphone usage patterns and addiction labels.
        """
    )   

@st.cache_data
def load_data():
    if not os.path.exists("data_b.csv"):
        st.error("🚨 'data_b.csv' not found! Make sure it is in your project folder.")
        st.stop()
    return pd.read_csv("data_b.csv")

# Load data baseline
df = load_data()

# ====================================================================
# AUTO-TRAINING ENGINE (Fixes the "Missing Model File" error permanently)
# ====================================================================
@st.cache_resource
def get_trained_model(_data_df):
    model_filename = "smartphone_addiction_model.joblib"
    
    # If the file already exists, load it immediately
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
        
    # If file is missing, automatically build it using your exact notebook code!
    st.warning("⚙️ Model file not found. Auto-training your RandomForest model directly from data_b.csv...")
    
    features = [
        'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'notifications_per_day',
        'app_opens_per_day', 'weekend_screen_time',
        'nominal__gender_female', 'nominal__gender_male',
        'nominal__gender_other', 'nominal__gender_nan', 'ordinal__stress_level',
        'ordinal__academic_work_impact'
    ]
    
    X = _data_df[features]
    y = _data_df['addicted_label']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Exact hyperparameters copied directly from your .ipynb code cell [7]
    trained_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10, 
        max_samples=0.5, 
        n_jobs=-1, 
        random_state=42,
        class_weight='balanced'
    )
    trained_model.fit(X_train, y_train)
    
    # Save it so it loads instantly next time
    joblib.dump(trained_model, model_filename)
    st.success("✅ Model trained and saved successfully!")
    return trained_model

# Load or auto-train your model instantly
my_model = get_trained_model(df)

st.subheader("📊 Sample Training Dataset (Top 8 Rows)")
st.dataframe(df.head(8), use_container_width=True)

# ====================================================================
# 1. SIDEBAR CONFIGURATION (ALL 15 FEATURES SUPPORTED VIA INTERFACE)
# ====================================================================
with st.sidebar:
    st.header("**Input Features**")
    
    with st.expander("**📱 Screen Time Metrics**", expanded=True):
        feature9 = st.slider("Weekend Screen Time", min_value=0.5, max_value=18.0, value=5.0, step=0.5)
        feature2 = st.slider("Daily Screen Time Hours", min_value=0.5, max_value=15.0, value=5.0, step=0.5)
        feature3 = st.slider("Social Media Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        feature4 = st.slider("Gaming Hours", min_value=0.0, max_value=12.0, value=1.0, step=0.5)
        feature5 = st.slider("Work Study Hours", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
        feature6 = st.slider("Sleep Hours", min_value=1.0, max_value=16.0, value=7.0, step=0.5)

    with st.expander("**📋 Usage & Demographics**", expanded=False):
        feature1 = st.number_input("Age", min_value=1, max_value=120, value=22)
        feature7 = st.number_input("Notifications per Day", min_value=0, max_value=1000, value=120)
        feature8 = st.number_input("App Opens per Day", min_value=0, max_value=1000, value=60)
        
        gender_input = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        stress_input = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        academic_input = st.selectbox("Academic Work Impact?", ["Yes", "No", "Not Applicable"])

# ====================================================================
# 2. MATCH NOTEBOOK TRANSFORMATIONS MANUALLY
# ====================================================================
is_female = 1.0 if gender_input == "Female" else 0.0
is_male = 1.0 if gender_input == "Male" else 0.0
is_other = 1.0 if gender_input == "Other" else 0.0
is_nan = 1.0 if gender_input == "Prefer not to say" else 0.0

stress_map = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
val_stress = stress_map[stress_input]

academic_map = {"Yes": 0.0, "No": 1.0, "Not Applicable": 2.0}
val_academic = academic_map[academic_input]

X_single = pd.DataFrame([{
    'age': float(feature1),
    'daily_screen_time_hours': float(feature2),
    'social_media_hours': float(feature3),
    'gaming_hours': float(feature4),
    'work_study_hours': float(feature5),
    'sleep_hours': float(feature6),
    'notifications_per_day': float(feature7),
    'app_opens_per_day': float(feature8),
    'weekend_screen_time': float(feature9),
    'nominal__gender_female': is_female,
    'nominal__gender_male': is_male,
    'nominal__gender_other': is_other,
    'nominal__gender_nan': is_nan,
    'ordinal__stress_level': val_stress,
    'ordinal__academic_work_impact': val_academic
}])

# ====================================================================
# 3. ACTION EVENT RUNNER (PREDICT BUTTON CLICK)
# ====================================================================
# ====================================================================
# 3. ACTION EVENT RUNNER (PREDICT BUTTON CLICK)
# ====================================================================
st.markdown("---")
st.info("⚙️ Configuration locked. Click below to execute model calculations.")

if st.button("🔮 Predict Addiction Likelihood", type="primary", use_container_width=True):
    st.subheader("🎯 Prediction Output Metrics")
    
    # Restructured into 2 columns for wider, clean visual layout
    col_pie, col_plot = st.columns(2)
    
    probabilities = my_model.predict_proba(X_single)
    prob_addicted = float(probabilities[0][1])  # Class 1 probability
    prob_safe = float(probabilities[0][0])      # Class 0 probability
    
    # ====================================================================
    # THE PIE PERCENTAGE CHART
    # ====================================================================
    with col_pie:
        # Build the exact mapping layout for the Plotly engine
        pie_df = pd.DataFrame({
            "Risk Assessment": [f"Addiction Risk ({prob_addicted:.1%})", f"Safe Status ({prob_safe:.1%})"],
            "Probability Ratio": [prob_addicted, prob_safe]
        })
        
        fig_pie = px.pie(
            pie_df, 
            names="Risk Assessment", 
            values="Probability Ratio", 
            hole=0.5, # Makes it a modern donut chart
            color="Risk Assessment",
            color_discrete_map={
                f"Addiction Risk ({prob_addicted:.1%})": "#FF4B4B", # Red for danger
                f"Safe Status ({prob_safe:.1%})": "#00F0A0"       # Green for safe
            },
            title="🎯 Overall Addiction Risk Share Percentage"
        )
        
        # Display the custom chart
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Display a quick status banner below the pie
        if prob_addicted > 0.5:
            st.error(f"🚨 Status: High Likelihood of Smartphone Addiction ({prob_addicted:.1%})")
        else:
            st.success(f"✅ Status: Controlled Smartphone Activity Profile ({prob_safe:.1%})")

    # Baseline Data Metric Comparison Chart
    with col_plot:
        avg_data = pd.DataFrame({
            'Feature Class': ['Weekend Hours', 'Daily Weekday Hours'],
            'User Settings': [feature9, feature2],
            'Dataset Average Reference': [df['weekend_screen_time'].mean(), df['daily_screen_time_hours'].mean()]
        }).melt(id_vars='Feature Class', var_name='Metric Context', value_name='Hours')
        
        fig_bar = px.bar(avg_data, x='Feature Class', y='Hours', color='Metric Context', barmode='group', title="📊 Input Metrics Context Map")
        st.plotly_chart(fig_bar, use_container_width=True)


    # ====================================================================
    # 4. FIXED INDICES FOR MULTI-CLASS TREEEXPLAINER SHAP EXECUTION
    # ====================================================================
    st.markdown("---")
    st.subheader("🔍 Local Prediction Impact Breakdown (SHAP Chart)")
    
    with st.spinner("Decoding layout nodes via TreeExplainer..."):
        try:
            tree_explainer = shap.TreeExplainer(my_model)
            shap_values_single = tree_explainer(X_single)
            
            # Squeezing the multidimensional arrays cleanly to prevent IndexError loops
            exp_single = shap.Explanation(
                values=shap_values_single.values[0, :, 1], # Class 1
                base_values=shap_values_single.base_values[0, 1], # Squeezed base value for class 1
                data=X_single.values[0],
                feature_names=X_single.columns
            )
            
            fig, ax = plt.subplots(figsize=(12, 3))
            shap.plots.force(exp_single, matplotlib=True, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
            
        except Exception as e:
            st.error(f"Could not build internal validation graphics plot: {e}")
