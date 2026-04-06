import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. CORE ANALYSIS LOGIC ---

def analyze_crop_health(image):
    """
    Approximates crop health using color analysis (HSV space).
    Healthy crops are vibrant green. Stressed crops are yellowish/brownish.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for 'Healthy Green'
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Define range for 'Stressed/Dry' (Yellowish/Brownish)
    lower_stressed = np.array([10, 40, 40])
    upper_stressed = np.array([30, 255, 255])
    
    # Create masks
    mask_healthy = cv2.inRange(hsv, lower_green, upper_green)
    mask_stressed = cv2.inRange(hsv, lower_stressed, upper_stressed)
    
    # Calculate percentages
    total_pixels = image.shape[0] * image.shape[1]
    healthy_pct = (np.sum(mask_healthy > 0) / total_pixels) * 100
    stressed_pct = (np.sum(mask_stressed > 0) / total_pixels) * 100
    critical_pct = 100 - (healthy_pct + stressed_pct) # Soil or dead vegetation
    
    # Generate Health Map (Visual representation)
    health_map = np.zeros_like(image)
    health_map[mask_healthy > 0] = [0, 255, 0]    # Green for healthy
    health_map[mask_stressed > 0] = [0, 255, 255] # Yellow for stressed
    health_map[(mask_healthy == 0) & (mask_stressed == 0)] = [0, 0, 255] # Red for critical/soil
    
    return healthy_pct, stressed_pct, critical_pct, health_map

def predict_risks(image, healthy_pct, stressed_pct):
    """
    Simple rule-based risk prediction.
    """
    risks = []
    
    # 1. Drought Risk
    if stressed_pct > 30:
        risks.append({"type": "High Drought Risk", "severity": "High", "reason": "High percentage of yellowing vegetation detected."})
    elif stressed_pct > 15:
        risks.append({"type": "Moderate Drought Risk", "severity": "Medium", "reason": "Increasing stressed zones observed."})
        
    # 2. Water Stress / Flood Risk (Simplified logic: look for dark/blueish patches)
    # In a real system, we'd use NIR or elevation. Here we use brightness/blue-ish tones.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < 50) / (image.shape[0] * image.shape[1]) * 100
    
    if dark_pixels > 20:
        risks.append({"type": "Potential Water Logging", "severity": "High", "reason": "Large dark/saturated areas detected (potential flooding)."})
    
    # 3. General Health Warning
    if healthy_pct < 50:
        risks.append({"type": "Low Yield Warning", "severity": "Critical", "reason": "Less than 50% of the area shows healthy crop growth."})
        
    return risks

def get_recommendations(risks):
    """
    Generates actionable advice based on detected risks.
    """
    recommendations = []
    for risk in risks:
        if "Drought" in risk['type']:
            recommendations.append("Increase irrigation frequency in yellow-marked zones.")
            recommendations.append("Apply mulch to retain soil moisture.")
        if "Water Logging" in risk['type']:
            recommendations.append("Check drainage channels for blockages.")
            recommendations.append("Avoid further irrigation until soil dries.")
        if "Low Yield" in risk['type']:
            recommendations.append("Conduct soil nutrient testing.")
            recommendations.append("Consider targeted fertilizer application.")
            
    if not recommendations:
        recommendations.append("Maintain current maintenance schedule.")
        recommendations.append("Monitor weather for upcoming changes.")
        
    return list(set(recommendations))

# --- NEW INNOVATIVE FEATURES ---

def simulate_improvement(h_pct, s_pct, c_pct, irr, fert):
    """Feature 1: Simulates health boost based on inputs."""
    boost = (s_pct * (irr/100) * 0.7) + (c_pct * (fert/100) * 0.3)
    new_h = min(100, h_pct + boost)
    new_s = max(0, s_pct - (boost * 0.6))
    return new_h, new_s, 100 - (new_h + new_s)

def generate_priority_map(h_map, risks):
    """Feature 2: Creates an urgency-based priority map."""
    p_map = h_map.copy()
    if any(r['severity'] in ['High', 'Critical'] for r in risks):
        # Enhance red zones for high-risk areas
        p_map[np.where((p_map == [0, 0, 255]).all(axis=2))] = [255, 0, 0]
    return p_map

def get_xai_logic(h_pct, s_pct, c_pct, risks):
    """Feature 3: Rule-based explanation for AI decisions."""
    logic = [
        f"🟢 Health Rule: {h_pct:.1f}% green pixels -> {'Optimal' if h_pct > 70 else 'Sub-optimal'} growth.",
        f"🟡 Stress Rule: {s_pct:.1f}% yellow pixels -> {'High' if s_pct > 15 else 'Low'} drought stress.",
        f"🔴 Critical Rule: {c_pct:.1f}% red pixels -> {'Significant' if c_pct > 20 else 'Minimal'} soil exposure."
    ]
    for r in risks:
        logic.append(f"🔍 Risk Logic: '{r['type']}' triggered because {r['reason'].lower()}")
    return logic

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="AgroDrone AI - InnovateX Demo", layout="wide")

st.title("🌾 AgroDrone AI: Precision Agriculture Analysis")
st.markdown("### Hackathon Demo for InnovateX Competition")

uploaded_file = st.file_uploader("Upload a Drone Image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Drone Image")
        st.image(image_rgb, use_container_width=True)
        
    # Run Analysis
    with st.spinner("Analyzing crop health..."):
        h_pct, s_pct, c_pct, h_map = analyze_crop_health(image)
        risks = predict_risks(image, h_pct, s_pct)
        recs = get_recommendations(risks)
        
    with col2:
        st.subheader("Crop Health Map")
        st.image(h_map, use_container_width=True)
        st.caption("🟢 Green: Healthy | 🟡 Yellow: Stressed | 🔴 Red: Critical/Soil")

    # --- NEW: PRIORITY MAP & XAI PANEL ---
    st.divider()
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.subheader("📍 Priority Urgency Map")
        p_map = generate_priority_map(h_map, risks)
        st.image(p_map, use_container_width=True)
        st.caption("Urgency Level: Red zones require immediate intervention.")
        
    with p_col2:
        st.subheader("🧠 Explainable AI (XAI) Panel")
        logic_points = get_xai_logic(h_pct, s_pct, c_pct, risks)
        for point in logic_points:
            st.write(point)

    # --- NEW: WHAT-IF SIMULATOR ---
    st.divider()
    st.subheader("🧪 What-If Simulator")
    st.info("Simulate how crop health improves with better resource management.")
    
    s1, s2 = st.columns(2)
    irr_val = s1.slider("Additional Irrigation (%)", 0, 100, 20)
    fert_val = s2.slider("Targeted Fertilizer (%)", 0, 100, 10)
    
    new_h, new_s, new_c = simulate_improvement(h_pct, s_pct, c_pct, irr_val, fert_val)
    
    c1, c2 = st.columns(2)
    c1.metric("Current Healthy %", f"{h_pct:.1f}%")
    c2.metric("Simulated Healthy %", f"{new_h:.1f}%", delta=f"{new_h - h_pct:.1f}%")
    
    st.progress(new_h / 100)

    # Metrics
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Healthy Vegetation", f"{h_pct:.1f}%")
    m2.metric("Stressed Vegetation", f"{s_pct:.1f}%")
    m3.metric("Critical/Soil Area", f"{c_pct:.1f}%")
    
    # Risks and Recommendations
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("⚠️ Risk Assessment")
        if risks:
            for r in risks:
                st.error(f"**{r['type']}** ({r['severity']})\n\n{r['reason']}")
        else:
            st.success("No significant risks detected in this sector.")
            
    with res_col2:
        st.subheader("💡 Actionable Recommendations")
        for i, rec in enumerate(recs):
            st.info(f"{i+1}. {rec}")

else:
    st.info("Please upload a drone image of a farmland to start the analysis.")
    st.markdown("""
    **How it works:**
    1. **Image Processing:** Uses OpenCV to segment colors in the HSV space.
    2. **NDVI Approximation:** Simulates the Normalized Difference Vegetation Index by analyzing the intensity of 'Green' vs 'Yellow/Brown' pixels.
    3. **Risk Engine:** A rule-based system that correlates color patterns with agricultural risks like drought or water-logging.
    """)
