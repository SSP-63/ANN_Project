import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("risk_model.h5")

# Question bank (replace these with your actual questions)
questions = [
    "1. How often do you feel the urge to use drugs?",
    "2. Do you use drugs to relieve stress?",
    "3. Have you experienced withdrawal symptoms?",
    "4. Do drugs interfere with your daily activities?",
    "5. Have you tried and failed to quit using drugs?",
    "6. Do you spend a lot of time thinking about drugs?",
    "7. Have your relationships been affected by drug use?",
    "8. Do you hide your drug use from others?",
    "9. Have you used drugs in risky situations (e.g. driving)?",
    "10. Do you feel dependent on drugs to function?"
]

# Options
options = [
    "Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"
]

st.set_page_config(page_title="Addiction Risk Assessment", page_icon="🧠", layout="wide")

# --- Custom CSS for enhanced UI ---
st.markdown("""
    <style>
    .main-header {
        font-size:2.5rem;
        font-weight:700;
        color:#2c3e50;
        margin-bottom:0.2em;
    }
    .subtitle {
        font-size:1.2rem;
        color:#bbb;
        margin-bottom:1.5em;
    }
    .question-card {
        background: #222; /* Changed from #f8f9fa to dark */
        color: #f1f1f1;   /* Light text for dark background */
        border-radius: 10px;
        padding: 1.2em 1em;
        margin-bottom: 1.2em;
        box-shadow: 0 2px 8px rgba(44,62,80,0.06);
    }
    .stButton>button {
        background: linear-gradient(90deg,#4e54c8,#8f94fb);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6em 2em;
        margin-top: 1em;
    }
    .footer {
        margin-top: 2em;
        color: #888;
        font-size: 0.95em;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.info("Answer each question using the options. Click 'Submit Answers' when done.")

# --- Custom Header ---
st.markdown('<div class="main-header">🧠 Addiction Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A simple tool to help you understand your risk of addiction. Please answer honestly for best results.</div>', unsafe_allow_html=True)

# --- Progress Bar ---
progress_placeholder = st.empty()

# Use a form to collect all responses
with st.form("risk_form"):
    responses = []
    for i, q in enumerate(questions):
        with st.container():
            st.markdown(f'<div class="question-card"><b>Q{i+1}.</b> {q}</div>', unsafe_allow_html=True)
            ans = st.radio("", options, index=0, key=f"q{i+1}")
            responses.append(int(ans[-2]))
        # Update progress bar
        progress_placeholder.progress((i + 1) / len(questions))
    submitted = st.form_submit_button("📋 Submit Answers")

# Display summary and results upon submission
if submitted:
    st.subheader("📝 Your Responses")
    for i, q in enumerate(questions):
        st.markdown(f'<div class="question-card"><b>Q{i+1}.</b> {q}<br><i>Your answer:</i> <b>{options[responses[i]]}</b></div>', unsafe_allow_html=True)
    st.subheader("🧠 Neural Network Visualization")
    st.image("neural_net_simulation.gif", caption="Neural Network Representation", use_container_width=True)
    input_array = np.array(responses).reshape(1, -1)
    total_score = np.sum(input_array)
    if total_score == 0:
        risk = 0
    else:
        prediction = model.predict(input_array)
        risk = np.argmax(prediction)
    st.subheader("🧾 Assessment Result")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if risk == 0:
            st.success("✅ **Low Risk**: You currently show minimal signs of addiction. Keep maintaining a healthy lifestyle.")
        elif risk == 1:
            st.warning("⚠️ **Moderate Risk**: Some signs of concern. Consider consulting a counselor or taking preventive steps.")
        else:
            st.error("🚨 **High Risk**: You may be showing strong signs of addiction. It's highly recommended to seek professional help.")
    st.markdown("---")
    st.caption("Note: This is not a clinical diagnosis. For real concerns, consult a professional.")

# --- Footer ---
st.markdown('<div class="footer">Made with ❤️ using Streamlit | © 2025 Addiction Risk Assessment Tool</div>', unsafe_allow_html=True)
