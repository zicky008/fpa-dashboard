import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="FP&A Dashboard - FMCG Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }

    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
    }

    .insight-box {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main application entry point
def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">🏢 FP&A Dashboard - FMCG Analytics Platform</div>', 
                unsafe_allow_html=True)

    # Load data
    mastersale_data, fbmc_data, dept_data, dim_coa_data = load_data()

    # Sidebar configuration
    st.sidebar.title("📊 Dashboard Controls")

    # Fiscal Year Selection
    fiscal_years = ['FY2425', 'FY2526', 'FY2627', 'FY2930']
    selected_fy = st.sidebar.selectbox("🗓️ Select Fiscal Year", fiscal_years, index=0)

    # Analysis Type Selection
    analysis_type = st.sidebar.radio(
        "📈 Analysis Type",
        ["Revenue Analysis", "Cost Analysis", "P&L Analysis", "Variance Analysis"]
    )

    # Data Source Selection
    data_sources = st.sidebar.multiselect(
        "📋 Data Sources",
        ["Mastersale", "FBMC", "Department Costs"],
        default=["Mastersale", "FBMC"]
    )

    # Cost Hierarchy Level
    if analysis_type == "Cost Analysis":
        cost_level = st.sidebar.selectbox(
            "🏗️ Cost Hierarchy Level",
            ["Acc_Lv1", "Acc_Lv2", "Acc_Lv3", "Acc_Lv4", "Acc_Lv5"],
            index=0
        )
    else:
        cost_level = "Acc_Lv1"

    # Main content area
    if analysis_type == "Revenue Analysis":
        show_revenue_analysis(mastersale_data, fbmc_data, selected_fy, data_sources)
    elif analysis_type == "Cost Analysis":
        show_cost_analysis(dept_data, dim_coa_data, selected_fy, cost_level)
    elif analysis_type == "P&L Analysis":
        show_pl_analysis(mastersale_data, fbmc_data, dept_data, selected_fy)
    elif analysis_type == "Variance Analysis":
        show_variance_analysis(mastersale_data, fbmc_data, dept_data, selected_fy)

if __name__ == "__main__":
    main()
