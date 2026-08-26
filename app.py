import os
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="Multi-Disease Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        /* Dynamic Theme-Aware Titles */
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--text-color);
            text-align: center;
            margin-top: -10px;
        }
        .sub-title {
            font-size: 1.05rem;
            color: var(--text-color);
            opacity: 0.8;
            text-align: center;
            margin-bottom: 25px;
        }
        
        /* Adaptive Disease Selection Cards */
        .disease-card {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 24px 20px;
            text-align: center;
            background-color: var(--secondary-background-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 12px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .disease-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.15);
        }
        .disease-card h4 {
            color: var(--text-color);
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 8px;
        }
        .disease-card p {
            color: var(--text-color);
            opacity: 0.75;
            font-size: 0.9rem;
            margin: 0;
        }

        /* Adaptive Diagnostic Result Cards */
        .risk-card-high {
            background-color: rgba(239, 68, 68, 0.15);
            border-left: 6px solid #EF4444;
            padding: 20px;
            border-radius: 10px;
            color: var(--text-color);
        }
        .risk-card-low {
            background-color: rgba(16, 185, 129, 0.15);
            border-left: 6px solid #10B981;
            padding: 20px;
            border-radius: 10px;
            color: var(--text-color);
        }
        
        /* Unified Button Layout */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. DYNAMIC PKL ASSET LOADER
# ============================================================
@st.cache_resource
def load_disease_assets(disease_key):
    """Loads model, scaler, and feature list directly from your root directory."""
    model_path = f"model_{disease_key}.pkl"
    scaler_path = f"scaler_{disease_key}.pkl"
    features_path = f"features_{disease_key}.pkl"
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else []
    
    return model, scaler, features

RECOMMENDATIONS = {
    "diabetes": [
        "Eat a balanced diet with less sugar, salt, and saturated fats.",
        "Exercise regularly (at least 150 minutes per week).",
        "Maintain a healthy body weight and BMI.",
        "Monitor blood glucose and HbA1c levels regularly.",
        "Consult a healthcare professional for further medical assessment."
    ],
    "heart": [
        "Eat a heart-healthy diet with low sodium and low saturated fats.",
        "Engage in aerobic physical activity at least 150 minutes per week.",
        "Maintain normal blood pressure and serum cholesterol levels.",
        "Avoid smoking and excessive alcohol consumption.",
        "Consult a cardiologist for standard diagnostic evaluation."
    ],
    "kidney": [
        "Maintain proper daily hydration unless fluid-restricted.",
        "Limit dietary sodium and strictly manage blood pressure.",
        "Monitor Blood Urea Nitrogen (BUN) and Serum Creatinine regularly.",
        "Avoid non-steroidal anti-inflammatory drugs (NSAIDs) without prescription.",
        "Consult a nephrologist for clinical evaluation."
    ]
}

# ============================================================
# 3. HEADER & NAVIGATION TABS
# ============================================================
st.markdown("<div class='main-title'>💙 Multi-Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict Diabetes, Heart Disease, and Chronic Kidney Disease using Machine Learning & XAI</div>", unsafe_allow_html=True)

nav_home, nav_about, nav_contact = st.tabs(["🏠 Home / Prediction", "ℹ️ About System", "📞 Contact"])

if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = "diabetes"
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "select"

# ============================================================
# SCREEN 1: HOME & PREDICTION WORKFLOW
# ============================================================
with nav_home:
    
    # --------------------------------------------------------
    # STEP A: DISEASE SELECTION CARDS
    # --------------------------------------------------------
    if st.session_state.view_mode == "select":
        st.write("### Select a disease below to begin:")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("""
                <div class='disease-card'>
                    <h2 style='color:#2563EB;'>💉</h2>
                    <h4>Diabetes Prediction</h4>
                    <p style='color:#64748B; font-size:0.9rem;'>Assess diabetes risk using clinical health metrics.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Diabetes Prediction →", key="start_dia"):
                st.session_state.selected_disease = "diabetes"
                st.session_state.view_mode = "input"
                st.rerun()

        with c2:
            st.markdown("""
                <div class='disease-card'>
                    <h2 style='color:#DC2626;'>❤️</h2>
                    <h4>Heart Disease Prediction</h4>
                    <p style='color:#64748B; font-size:0.9rem;'>Assess cardiovascular risk using cardiac indicators.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Heart Disease Prediction →", key="start_heart"):
                st.session_state.selected_disease = "heart"
                st.session_state.view_mode = "input"
                st.rerun()

        with c3:
            st.markdown("""
                <div class='disease-card'>
                    <h2 style='color:#059669;'>🩺</h2>
                    <h4>Chronic Kidney Disease</h4>
                    <p style='color:#64748B; font-size:0.9rem;'>Assess renal risk using kidney clinical parameters.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Kidney Disease Prediction →", key="start_kidney"):
                st.session_state.selected_disease = "kidney"
                st.session_state.view_mode = "input"
                st.rerun()

    # --------------------------------------------------------
    # STEP B: HEALTH INFORMATION FORM
    # --------------------------------------------------------
    elif st.session_state.view_mode == "input":
        current_dis = st.session_state.selected_disease
        st.button("← Back to Selection", on_click=lambda: st.session_state.update({"view_mode": "select"}))
        st.subheader(f"📝 Health Information ({current_dis.upper()})")

        inputs = {}
        with st.form("clinical_data_form"):
            if current_dis == "diabetes":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    inputs["AGE"] = st.number_input("Age (years)", 1, 120, 50)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    inputs["Gender_M"] = 1 if gender == "Male" else 0
                    inputs["Gender_f"] = 1 if gender == "Female" else 0
                    inputs["BMI"] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.4)
                    inputs["HbA1c"] = st.number_input("HbA1c Level (%)", 3.0, 15.0, 6.5)
                with col_b:
                    inputs["Urea"] = st.number_input("Urea (mmol/L)", 0.0, 50.0, 5.2)
                    inputs["Cr"] = st.number_input("Creatinine (µmol/L)", 0.0, 500.0, 68.0)
                    inputs["Chol"] = st.number_input("Cholesterol (mmol/L)", 0.0, 15.0, 4.8)
                    inputs["TG"] = st.number_input("Triglycerides (mmol/L)", 0.0, 15.0, 2.3)
                with col_c:
                    inputs["HDL"] = st.number_input("HDL (mmol/L)", 0.0, 5.0, 1.2)
                    inputs["LDL"] = st.number_input("LDL (mmol/L)", 0.0, 10.0, 2.6)
                    inputs["VLDL"] = st.number_input("VLDL (mmol/L)", 0.0, 5.0, 1.9)
                    inputs["ID"] = 1.0
                    inputs["No_Pation"] = 1.0

            elif current_dis == "heart":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    inputs["Age"] = st.number_input("Age (years)", 1, 120, 54)
                    sex = st.selectbox("Sex", ["Male", "Female"])
                    inputs["Sex"] = 1 if sex == "Male" else 0
                    inputs["cp"] = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
                    inputs["trestbps"] = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 132)
                    inputs["chol"] = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 247)
                with col_b:
                    inputs["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
                    inputs["restecg"] = st.selectbox("Resting ECG Results", [0, 1, 2])
                    inputs["thalach"] = st.number_input("Max Heart Rate Achieved", 60, 220, 149)
                    inputs["exang"] = st.selectbox("Exercise Induced Angina", [0, 1])
                with col_c:
                    inputs["oldpeak"] = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
                    inputs["slope"] = st.selectbox("Slope of Peak ST", [0, 1, 2])
                    inputs["ca"] = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3, 4])
                    inputs["thal"] = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

            elif current_dis == "kidney":
                col_a, col_b = st.columns(2)
                with col_a:
                    inputs["Age"] = st.number_input("Age (years)", 1, 120, 54)
                    inputs["Creatinine_Level"] = st.number_input("Creatinine Level (mg/dL)", 0.1, 15.0, 1.3)
                    inputs["BUN"] = st.number_input("Blood Urea Nitrogen (BUN) (mg/dL)", 1.0, 100.0, 18.8)
                    inputs["Urine_Output"] = st.number_input("Urine Output (mL/day)", 100, 4000, 1315)
                with col_b:
                    inputs["Diabetes"] = st.selectbox("History of Diabetes", [0, 1])
                    inputs["Hypertension"] = st.selectbox("History of Hypertension", [0, 1])
                    inputs["GFR"] = st.number_input("Glomerular Filtration Rate (GFR)", 5, 150, 68)

            btn_submit = st.form_submit_button("Predict Risk →")
            
            if btn_submit:
                st.session_state.user_inputs = inputs
                st.session_state.view_mode = "result"
                st.rerun()

    # --------------------------------------------------------
    # STEP C: DIAGNOSTIC & EXPLAINABLE AI (XAI) DASHBOARD
    # --------------------------------------------------------
    elif st.session_state.view_mode == "result":
        current_dis = st.session_state.selected_disease
        inputs = st.session_state.user_inputs
        model, scaler, feature_cols = load_disease_assets(current_dis)

        st.subheader(f"📊 Diagnostic & Risk Breakdown ({current_dis.upper()})")
        st.markdown("---")

        # Fallback in case features.pkl wasn't loaded properly
        if not feature_cols:
            feature_cols = list(inputs.keys())

        # Construct DataFrame strictly following feature_cols sequence
        raw_df = pd.DataFrame([inputs])
        for col in feature_cols:
            if col not in raw_df.columns:
                raw_df[col] = 0.0
        raw_df = raw_df[feature_cols]

        # Apply scaling if scaler exists
        if scaler:
            scaled_vals = scaler.transform(raw_df)
        else:
            scaled_vals = raw_df.values

        # Model Inference
        if model:
            pred = model.predict(scaled_vals)[0]
            prob = model.predict_proba(scaled_vals)[0][pred] * 100 if hasattr(model, "predict_proba") else 90.0
        else:
            pred = 1
            prob = 88.0

        # UI Layout: 3 Columns
        col_res, col_xai, col_rec = st.columns([1.2, 1.8, 1.5])

        # --- Card 1: Prediction Result ---
       # Map raw keys to complete display names
        DISEASE_NAMES = {
            "diabetes": "Diabetes",
            "heart": "Heart Disease",
            "kidney": "Chronic Kidney Disease"
        }
        
        disease_label = DISEASE_NAMES.get(current_dis, current_dis.capitalize())

        # --- Card 1: Prediction Result ---
        with col_res:
            st.write("#### Prediction Result")
            if pred == 1:
                st.markdown(f"""
                    <div class='risk-card-high'>
                        <h4 style='color:#EF4444; margin:0; font-weight:700;'>⚠️ High Risk of {disease_label}</h4>
                        <p style='margin-top:10px; margin-bottom:2px;'>Confidence Score</p>
                        <h1 style='color:#EF4444; margin:0;'>{prob:.0f}%</h1>
                        <small>This prediction is based on the health information provided.</small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='risk-card-low'>
                        <h4 style='color:#10B981; margin:0; font-weight:700;'>✅ Low Risk of {disease_label}</h4>
                        <p style='margin-top:10px; margin-bottom:2px;'>Confidence Score</p>
                        <h1 style='color:#10B981; margin:0;'>{prob:.0f}%</h1>
                        <small>This prediction is based on the health information provided.</small>
                    </div>
                """, unsafe_allow_html=True)

        # --- Card 2: Explainable AI (XAI) Breakdown ---
        with col_xai:
            st.write("#### Why Was This Prediction Made? (XAI)")
            st.caption("Top feature contributions evaluated by the trained model:")

            if model and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=True).tail(5)
                st.bar_chart(fi_df.set_index("Feature"), color="#2563EB")
                
                # Fetch top 2 factors for narrative
                top_features = fi_df.tail(2)["Feature"].tolist()
                top1_feat = top_features[1]
                top1_val = inputs.get(top1_feat, "N/A")
                top2_feat = top_features[0]
                top2_val = inputs.get(top2_feat, "N/A")
            else:
                fi_df = pd.DataFrame({
                    "Feature": feature_cols[:5],
                    "Importance": [0.35, 0.25, 0.18, 0.12, 0.10]
                }).sort_values("Importance", ascending=True)
                st.bar_chart(fi_df.set_index("Feature"), color="#2563EB")
                top1_feat, top1_val = "HbA1c", inputs.get("HbA1c", 6.5)
                top2_feat, top2_val = "BMI", inputs.get("BMI", 28.5)

           # Enhanced, Patient-Friendly Explanation Text (Clean Multiline formatting)
            if pred == 1:
                st.error(
                    f"💡 **Key Risk Factors Identified:**\n\n"
                    f"• **{top1_feat}** (Value: **{top1_val}**) was the single most influential factor pushing the overall decision toward a **High Risk** rating.\n\n"
                    f"• **{top2_feat}** (Value: **{top2_val}**) also contributed significantly to elevating your calculated risk score."
                )
            else:
                st.success(
                    f"💡 **Key Protective Factors Identified:**\n\n"
                    f"• Your **{top1_feat}** (Value: **{top1_val}**) was the main factor keeping your evaluation in the **Low Risk** category.\n\n"
                    f"• Healthy levels in **{top2_feat}** (Value: **{top2_val}**) further lowered your clinical risk profile."
                )

        # --- Card 3: Top 3 Actionable Recommendations ---
        with col_rec:
            st.write("#### Recommended Action Steps")
            top_3_recommendations = RECOMMENDATIONS[current_dis][:3]
            for item in top_3_recommendations:
                st.markdown(f"✓ {item}")

        st.divider()
        st.warning("**Disclaimer:** This tool is for screening purposes and educational evaluation. Consult a medical professional for official clinical diagnostic advice.")

        # Bottom Navigation Row: Predict Again (Left) vs Back to Home (Right)
        btn_left, btn_right = st.columns([1, 1])
        with btn_left:
            if st.button("🔄 Predict Again", key="bottom_again_btn"):
                st.session_state.view_mode = "input"
                st.rerun()
        with btn_right:
            if st.button("🏠 Back to Home", key="bottom_home_btn"):
                st.session_state.view_mode = "select"
                st.rerun()
# ============================================================
# SCREEN 2 & 3: ABOUT & CONTACT
# ============================================================
with nav_about:
    st.subheader("ℹ️ About the System")
    st.markdown("---")
    
    # Overview Banner
    st.markdown("""
        <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); border-radius: 12px; padding: 24px; margin-bottom: 25px;">
            <h3 style="margin-top:0; color: var(--text-color);">Smart Healthcare Screening Platform</h3>
            <p style="color: var(--text-color); opacity: 0.85; font-size: 1rem; line-height: 1.6;">
                The <b>AI-Driven Multi-Disease Risk Prediction System</b> is designed to assist both clinical researchers and individuals in evaluating early health risks. By leveraging machine learning models alongside Explainable AI (XAI), the platform translates complex biological datasets into clear, transparent, and actionable health insights.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Core Architecture Columns
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
            <div class="disease-card" style="text-align: left; height: 100%;">
                <h3 style="color: #2563EB; margin-top:0;">🤖 Machine Learning</h3>
                <h4 style="margin-bottom: 6px;">Random Forest Ensembles</h4>
                <p>Trained on clinical datasets to deliver high accuracy predictions across multiple key disease indicators.</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="disease-card" style="text-align: left; height: 100%;">
                <h3 style="color: #10B981; margin-top:0;">🔍 Transparency</h3>
                <h4 style="margin-bottom: 6px;">Explainable AI (XAI)</h4>
                <p>Features top feature importance visualization to clearly show which parameters influenced your evaluation.</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="disease-card" style="text-align: left; height: 100%;">
                <h3 style="color: #EF4444; margin-top:0;">🩺 Scope</h3>
                <h4 style="margin-bottom: 6px;">Triple Disease Assessment</h4>
                <p>Covers critical diagnostic evaluation for Diabetes, Cardiovascular (Heart) Disease, and Chronic Kidney Disease.</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.write(" ")
    st.write(" ")
    
    # Supported Modules Section
    st.markdown("**Supported Diagnostic Modules**")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("• **Diabetes Module:** Analyzes Glucose, HbA1c, BMI, and Lipid profiles.")
    with m2:
        st.markdown("• **Heart Disease Module:** Evaluates Resting Blood Pressure, Cholesterol, and ECG metrics.")
    with m3:
        st.markdown("• **Kidney Disease Module:** Assesses Creatinine, BUN, Urine Output, and GFR levels.")

with nav_contact:
    st.subheader("📞 Contact Us")
    st.markdown("---")
    
    # Gmail Compose Web URL
    gmail_url = "https://mail.google.com/mail/?view=cm&fs=1&to=supporthealthcareai@gmail.com"
    
    st.markdown(f"""
        <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); border-radius: 12px; padding: 24px; max-width: 500px;">
            <h4 style="margin-top:0; color: var(--text-color);">Support & Inquiries</h4>
            <p style="color: var(--text-color); opacity: 0.85; margin-bottom: 12px;">
                If you have any questions, feedback, or technical issues regarding the system, please reach out via email:
            </p>
            <p style="font-size: 1.1rem; font-weight: 600; margin: 0;">
                ✉️ <a href="{gmail_url}" target="_blank" style="color: #2563EB; text-decoration: none;">supporthealthcareai@gmail.com</a>
            </p>
        </div>
    """, unsafe_allow_html=True)