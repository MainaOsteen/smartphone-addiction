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
    # Update the tracking name to match your zipped repository asset
    if not os.path.exists("data_b.zip"):
        st.error("🚨 'data_b.zip' not found! Make sure it is pushed to your GitHub project folder.")
        st.stop()
    
    # Pandas automatically detects the zip compression and parses the internal CSV!
    return pd.read_csv("data_b.zip")

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
    
    # Exact hyperparameters copied directly from your .ipynb code cell
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
st.markdown("---")
st.info("⚙️ Configuration locked. Click below to execute model calculations.")

if st.button("🔮 Predict Addiction Likelihood", type="primary", use_container_width=True):
    st.subheader("🎯 Prediction Output Metrics")
    
    # Calculate Model Probabilities
    probabilities = my_model.predict_proba(X_single)
    prob_addicted = float(probabilities[0][1])  # Class 1 probability
    prob_safe = float(probabilities[0][0])      # Class 0 probability

    # DYNAMICALLY FIND THE BEST FEATURE
    features_list = [
        'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'notifications_per_day',
        'app_opens_per_day', 'weekend_screen_time',
        'nominal__gender_female', 'nominal__gender_male',
        'nominal__gender_other', 'nominal__gender_nan', 'ordinal__stress_level',
        'ordinal__academic_work_impact'
    ]
    importances = my_model.feature_importances_
    best_feature_idx = np.argmax(importances)
    best_feature_name = features_list[best_feature_idx]
    readable_feature_name = best_feature_name.replace("_", " ").title()
    user_current_value = float(X_single[best_feature_name].iloc[0])

    # CALCULATE OPTIMAL THRESHOLD (The turning point before addiction risk spikes > 50%)
    # We test values across the feature range to find where the risk crosses 50%
    feature_min = float(df[best_feature_name].min())
    feature_max = float(df[best_feature_name].max())
    test_grid = np.linspace(feature_min, feature_max, 100)
    
    X_scan = pd.concat([X_single] * 100, ignore_index=True)
    X_scan[best_feature_name] = test_grid
    scan_probs = my_model.predict_proba(X_scan)[:, 1]
    
    # Find the maximum value allowed before risk exceeds 50%
    safe_indices = np.where(scan_probs <= 0.5)[0]
    if len(safe_indices) > 0:
        optimal_cutoff = float(test_grid[safe_indices[-1]])
    else:
        optimal_cutoff = float(feature_min) # Default fallback

    # ====================================================================
    # ROW 1: THE PIE CHART AND THE DENSITY CURVE SIDE-BY-SIDE (2 COLUMNS ONLY)
    # ====================================================================
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        pie_df = pd.DataFrame({
            "Risk Assessment": [f"Addiction Risk ({prob_addicted:.1%})", f"Safe Status ({prob_safe:.1%})"],
            "Probability Ratio": [prob_addicted, prob_safe]
        })
        fig_pie = px.pie(
            pie_df, names="Risk Assessment", values="Probability Ratio", hole=0.5,
            color="Risk Assessment",
            color_discrete_map={
                f"Addiction Risk ({prob_addicted:.1%})": "#FF4B4B",
                f"Safe Status ({prob_safe:.1%})": "#00F0A0"
            },
            title="🎯 Overall Addiction Risk Share"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        if prob_addicted > 0.5:
            st.error(f"🚨 **Status Alert**: This configuration carries a **{prob_addicted:.1%}** mathematical probability of addiction.")
        else:
            st.success(f"✅ **Status Safe**: This configuration carries only a **{prob_addicted:.1%}** addiction risk value.")

    with row1_col2:
        # Build smooth line density graph
        counts = df[best_feature_name].value_counts().sort_index().reset_index()
        counts.columns = [best_feature_name, 'Count']
        
        fig_dist = px.line(
            counts, x=best_feature_name, y='Count',
            title=f"📈 Population Density vs Your Inputs ({readable_feature_name})",
            labels={best_feature_name: readable_feature_name, 'Count': 'Frequency Density'},
            line_shape='spline', color_discrete_sequence=['#4A5A6A']
        )
        fig_dist.update_traces(fill='tozeroy')
        
        # Red Dash Line = Your Choice
        fig_dist.add_vline(
            x=user_current_value, line_width=3, line_dash="dash", line_color="#FF4B4B",
            annotation_text=" Your Input", annotation_position="top left"
        )
        
        # Solid Green Line = Optimal Limit
        fig_dist.add_vline(
            x=optimal_cutoff, line_width=4, line_color="#00F0A0",
            annotation_text=" Optimal Max Limit", annotation_position="bottom right"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.metric(
            label=f"🟢 Maximum {readable_feature_name} before risk hits 50%", 
            value=f"{optimal_cutoff:.1f} Hours"
        )


    # ====================================================================
    # ROW 2: BAR CHART (FULL WIDTH SO IT DOES NOT SQUEEZE)
    # ====================================================================
    st.markdown("---")
    avg_data = pd.DataFrame({
        'Feature Class': ['Weekend Hours', 'Daily Weekday Hours'],
        'Your Configuration': [feature9, feature2],
        'Dataset Average Reference': [df['weekend_screen_time'].mean(), df['daily_screen_time_hours'].mean()]
    }).melt(id_vars='Feature Class', var_name='Metric Context', value_name='Hours')
    
    fig_bar = px.bar(
        avg_data, x='Feature Class', y='Hours', color='Metric Context', barmode='group', 
        title="📊 Context Map: Your Metrics vs Dataset Averages"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
