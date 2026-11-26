"""Small presentation helpers; model inference remains in src.predict."""

import html
import streamlit as st


def apply_style():
    st.markdown("""<style>
    .block-container {max-width:1480px;padding-top:4.2rem;padding-bottom:2rem;}
    h1 {font-size:2.35rem!important;letter-spacing:-.055em;line-height:1.15!important;}
    h2 {font-size:1.65rem!important;letter-spacing:-.025em;}
    h3 {font-size:1.1rem!important;}
    [data-testid="stSidebar"] {border-right:1px solid #223149;}
    [data-testid="stMetric"] {padding:16px 18px;border:1px solid #28384f;border-radius:14px;background:#101d30;}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
    [data-testid="stImage"] img {border-radius:14px;}
    [data-testid="stTabs"] [role="tablist"] {gap:20px;margin:12px 0 20px;}
    [data-testid="stTabs"] [role="tab"] {font-size:.88rem;}
    .eyebrow {font-size:.72rem;letter-spacing:.18em;color:#54dfca;font-weight:700;margin:0 0 12px;}
    .quiet {color:#9bacbf;font-size:.88rem;line-height:1.6;}
    .value-card {background:linear-gradient(135deg,#122b35,#101e33);border:1px solid #2d5560;border-radius:16px;padding:24px;}
    .value-card .value {font-size:3.2rem;font-weight:650;letter-spacing:-.04em;font-variant-numeric:tabular-nums;}
    .gauge {position:relative;height:8px;background:linear-gradient(90deg,#4c73ce,#2b3c52 48%,#2b3c52 52%,#2dd4bf);border-radius:6px;margin:25px 7px 12px;}
    .gauge-dot {position:absolute;top:-6px;width:20px;height:20px;background:#f3fbff;border:4px solid #2dd4bf;border-radius:50%;transform:translateX(-50%);}
    .gauge-labels {display:flex;justify-content:space-between;color:#a0b2c7;font-size:.8rem;}
    .step-card {border-left:2px solid #2dd4bf;padding:3px 16px;margin:15px 0;color:#c4d0e0;}
    </style>""", unsafe_allow_html=True)


def steering_gauge(value):
    """Show only a numeric value: no invented physical steering angle."""
    extent = max(1.0, abs(value) * 1.1)
    position = (value / extent + 1) * 50
    st.markdown(f'''<div class="gauge"><span class="gauge-dot" style="left:{position:.2f}%"></span></div>
        <div class="gauge-labels"><span>{-extent:.1f}</span><span>0 · neutral</span><span>+{extent:.1f}</span></div>''',
        unsafe_allow_html=True)


def input_card(title, description):
    st.markdown(f'<div class="step-card"><strong>{html.escape(title)}</strong><br>'
                f'<span class="quiet">{html.escape(description)}</span></div>', unsafe_allow_html=True)
