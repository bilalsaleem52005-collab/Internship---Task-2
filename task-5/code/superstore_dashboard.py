"""
🌸 Global Superstore – Premium Interactive Dashboard
Dual Theme (Dark/Light) | Modern Glassmorphism Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import urllib.request
from datetime import datetime, timedelta

# -------------------------------
# 0. THEME MANAGEMENT
# -------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default to dark for dramatic effect

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Global Superstore", page_icon="🌸", layout="wide")

# Dynamic theme-based styling
theme = st.session_state.theme

# -------------------------------
# 2. CUSTOM CSS – Dual Theme
# -------------------------------
if theme == "dark":
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark Theme Backgrounds */
    .stApp {
        background: #0A0C10;
        background-image: radial-gradient(circle at 25% 0%, rgba(30, 100, 180, 0.08) 0%, rgba(0,0,0,0) 50%);
    }
    
    [data-testid="stSidebar"] {
        background: #0F1117;
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    
    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* KPI Cards - Dark */
    .metric-card {
        border-radius: 28px;
        padding: 24px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(18, 22, 30, 0.7);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94A3B8;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: 800;
        color: #F1F5F9;
        margin-top: 10px;
        line-height: 1;
        background: linear-gradient(135deg, #E2E8F0, #CBD5E1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-trend {
        margin-top: 12px;
        font-size: 13px;
        color: #A7F3D0;
        font-weight: 500;
    }
    
    /* Headers */
    h1 {
        font-size: 48px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #A78BFA, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    h2, h3, h4 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown p {
        color: #CBD5E1;
    }
    
    /* Chart containers */
    [data-testid="stPlotlyChart"] {
        background: rgba(18, 22, 30, 0.6);
        border-radius: 24px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(5px);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Buttons */
    .stButton button {
        border: none;
        border-radius: 40px;
        background: linear-gradient(135deg, #8B5CF6, #3B82F6);
        color: white;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59,130,246,0.4);
    }
    
    /* Sidebar elements */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 16px;
        background: rgba(30, 35, 48, 0.8);
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), rgba(96,165,250,0.5), transparent);
        margin: 30px 0;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* Toggle button */
    .theme-toggle {
        background: rgba(30, 35, 48, 0.9);
        border-radius: 40px;
        padding: 5px 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """
else:
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Light Theme Backgrounds */
    .stApp {
        background: #F0F4F8;
        background-image: radial-gradient(circle at 100% 0%, rgba(139, 92, 246, 0.05) 0%, rgba(0,0,0,0) 70%);
    }
    
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        backdrop-filter: blur(10px);
    }
    
    /* KPI Cards - Light Glassmorphism */
    .metric-card {
        border-radius: 28px;
        padding: 24px;
        margin-bottom: 12px;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.9);
        box-shadow: 0 8px 20px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.08);
        border-color: rgba(139, 92, 246, 0.2);
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #475569;
        font-weight: 700;
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 10px;
        line-height: 1;
    }
    
    .metric-trend {
        margin-top: 12px;
        font-size: 13px;
        color: #059669;
        font-weight: 600;
    }
    
    /* Headers */
    h1 {
        font-size: 48px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #7C3AED, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown p {
        color: #334155;
    }
    
    /* Chart containers */
    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #E5E7EB;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #C4B5FD, #93C5FD, #A7F3D0, transparent);
        margin: 30px 0;
        border-radius: 999px;
    }
    
    footer {
        visibility: hidden;
    }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)

# Theme toggle button in sidebar
st.sidebar.markdown("## 🎨 Appearance")
st.sidebar.button(
    "🌙 Dark" if theme == "light" else "☀️ Light",
    on_click=toggle_theme,
    use_container_width=True,
)

# -------------------------------
# 3. LOAD DATA (cached)
# -------------------------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "dataset", "global_superstore.csv")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    if not os.path.exists(dataset_path):
        with st.spinner("🌸 Loading data..."):
            url = "https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv"
            urllib.request.urlretrieve(url, dataset_path)
    
    df = pd.read_csv(dataset_path, encoding='latin1')
    df.columns = df.columns.str.strip()
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df = df.dropna(subset=['Order Date'])
    return df

df = load_data()

# -------------------------------
# 4. HELPER: Apply theme to Plotly figures
# -------------------------------
def apply_theme_to_fig(fig, theme):
    if theme == "dark":
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            title_font=dict(color="#F1F5F9"),
            legend=dict(font=dict(color="#CBD5E1")),
            xaxis=dict(
                gridcolor="rgba(100,116,139,0.2)",
                linecolor="rgba(100,116,139,0.3)",
                tickfont=dict(color="#94A3B8")
            ),
            yaxis=dict(
                gridcolor="rgba(100,116,139,0.2)",
                linecolor="rgba(100,116,139,0.3)",
                tickfont=dict(color="#94A3B8")
            )
        )
    else:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1E293B"),
            title_font=dict(color="#0F172A"),
            legend=dict(font=dict(color="#334155")),
            xaxis=dict(
                gridcolor="rgba(203,213,225,0.5)",
                linecolor="rgba(203,213,225,0.8)",
                tickfont=dict(color="#475569")
            ),
            yaxis=dict(
                gridcolor="rgba(203,213,225,0.5)",
                linecolor="rgba(203,213,225,0.8)",
                tickfont=dict(color="#475569")
            )
        )
    return fig

# -------------------------------
# 5. DATE RANGE & SIDEBAR FILTERS
# -------------------------------
st.sidebar.markdown("## 🔍 Filters")
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()

date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Order Date'].dt.date >= start_date) & (df['Order Date'].dt.date <= end_date)]

regions = df['Region'].dropna().unique()
selected_region = st.sidebar.multiselect("🌍 Region", regions, default=regions[:2] if len(regions)>2 else regions)

categories = df['Category'].dropna().unique()
selected_category = st.sidebar.multiselect("📦 Category", categories, default=categories[:2] if len(categories)>2 else categories)

filtered_df = df.copy()
if selected_region:
    filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]
if selected_category:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_category)]

subcats = filtered_df['Sub-Category'].dropna().unique()
selected_subcat = st.sidebar.multiselect("🏷️ Sub‑Category", subcats, default=[])
if selected_subcat:
    filtered_df = filtered_df[filtered_df['Sub-Category'].isin(selected_subcat)]

if 'Segment' in filtered_df.columns:
    segments = filtered_df['Segment'].dropna().unique()
    selected_segment = st.sidebar.multiselect("👥 Customer Segment", segments, default=[])
    if selected_segment:
        filtered_df = filtered_df[filtered_df['Segment'].isin(selected_segment)]

# -------------------------------
# 6. MAIN DASHBOARD – HEADER & KPIs
# -------------------------------
st.markdown('<h1>📈 Global Superstore</h1>', unsafe_allow_html=True)
st.caption("✨ Transparent Intelligence · Gentle Insights · Dual Theme")

# Compute KPIs
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
margin = (total_profit / total_sales * 100) if total_sales else 0
avg_discount = filtered_df['Discount'].mean() * 100 if 'Discount' in filtered_df.columns else 0
total_qty = filtered_df['Quantity'].sum() if 'Quantity' in filtered_df.columns else 0
num_orders = filtered_df['Order ID'].nunique() if 'Order ID' in filtered_df.columns else len(filtered_df)

# =========================
# KPI SECTION
# =========================
st.markdown("## 📊 Key Metrics")

row1 = st.columns(4)

with row1[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💰 Sales</div>
        <div class="metric-value">${total_sales:,.0f}</div>
        <div class="metric-trend">↑ Total Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with row1[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 Profit</div>
        <div class="metric-value">${total_profit:,.0f}</div>
        <div class="metric-trend">↑ Net Earnings</div>
    </div>
    """, unsafe_allow_html=True)

with row1[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎯 Margin</div>
        <div class="metric-value">{margin:.1f}%</div>
        <div class="metric-trend">Profitability Ratio</div>
    </div>
    """, unsafe_allow_html=True)

with row1[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏷️ Avg Discount</div>
        <div class="metric-value">{avg_discount:.1f}%</div>
        <div class="metric-trend">Promotion Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

row2 = st.columns(2)

with row2[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📦 Units Sold</div>
        <div class="metric-value">{total_qty:,.0f}</div>
        <div class="metric-trend">Products Delivered</div>
    </div>
    """, unsafe_allow_html=True)

with row2[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🛒 Orders</div>
        <div class="metric-value">{num_orders:,}</div>
        <div class="metric-trend">Customer Purchases</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -------------------------------
# 7. ROW 1: TIME SERIES & TOP CUSTOMERS
# -------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Sales & Profit Over Time")
    monthly = filtered_df.groupby(pd.Grouper(key='Order Date', freq='M')).agg({'Sales':'sum', 'Profit':'sum'}).reset_index()
    if not monthly.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Sales'], mode='lines+markers', name='Sales', line=dict(color='#8B5CF6', width=2.5), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Profit'], mode='lines+markers', name='Profit', line=dict(color='#10B981', width=2.5), marker=dict(size=5)))
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), height=360, margin=dict(l=0, r=0, t=30, b=0))
        fig = apply_theme_to_fig(fig, theme)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for time series")

with right:
    st.subheader("🏆 Top 5 Customers")
    top_cust = filtered_df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(5).reset_index()
    if not top_cust.empty:
        fig2 = px.bar(top_cust, x='Sales', y='Customer Name', orientation='h', color='Sales', color_continuous_scale='Tealgrn' if theme=='dark' else 'Blues', height=360)
        fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Sales ($)", yaxis_title="")
        fig2 = apply_theme_to_fig(fig2, theme)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No customer data available")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -------------------------------
# 8. ROW 2: CATEGORY BREAKDOWN & SUB‑CATEGORY TREEMAP
# -------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("📊 Sales by Category")
    cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
    if not cat_sales.empty:
        fig3 = px.pie(cat_sales, values='Sales', names='Category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0), showlegend=True)
        fig3 = apply_theme_to_fig(fig3, theme)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No category data")

with colB:
    st.subheader("🗂️ Sub‑Category Sales (Treemap)")
    subcat_sales = filtered_df.groupby('Sub-Category')['Sales'].sum().reset_index()
    if not subcat_sales.empty:
        fig4 = px.treemap(subcat_sales, path=['Sub-Category'], values='Sales', color='Sales', color_continuous_scale='Viridis' if theme=='dark' else 'Blues', height=380)
        fig4.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        fig4 = apply_theme_to_fig(fig4, theme)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No sub‑category data")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -------------------------------
# 9. ROW 3: REGION & SEGMENT PERFORMANCE
# -------------------------------
colX, colY = st.columns(2)

with colX:
    if 'Region' in filtered_df.columns:
        st.subheader("🌍 Sales by Region")
        region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
        if not region_sales.empty:
            fig5 = px.bar(region_sales, x='Region', y='Sales', color='Sales', color_continuous_scale='Mint' if theme=='dark' else 'Teal', height=360)
            fig5.update_layout(xaxis_title="Region", yaxis_title="Sales ($)", margin=dict(l=0, r=0, t=20, b=0))
            fig5 = apply_theme_to_fig(fig5, theme)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No region data")
    else:
        st.info("Region column not available")

with colY:
    if 'Segment' in filtered_df.columns:
        st.subheader("👥 Segment Performance")
        seg_sales = filtered_df.groupby('Segment')['Sales'].sum().reset_index()
        if not seg_sales.empty:
            fig6 = px.bar(seg_sales, x='Segment', y='Sales', color='Sales', color_continuous_scale='Purples' if theme=='dark' else 'Oranges', height=360)
            fig6.update_layout(xaxis_title="Segment", yaxis_title="Sales ($)", margin=dict(l=0, r=0, t=20, b=0))
            fig6 = apply_theme_to_fig(fig6, theme)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No segment data")
    else:
        st.info("Segment column not available")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -------------------------------
# 10. DOWNLOAD & DATA TABLE
# -------------------------------
colDL, colEmpty = st.columns([1, 5])
with colDL:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("🌸 Download Filtered Data", data=csv, file_name="superstore_filtered.csv", mime="text/csv", use_container_width=True)

with st.expander("🔍 View Filtered Data Table"):
    st.dataframe(filtered_df.style.set_properties(**{'background-color': 'transparent', 'border-color': '#e2e8f0'}), use_container_width=True)

st.markdown("---")
st.markdown("<center><small>🌸 Built with kindness · Data with clarity · Dual Theme Magic 🌸</small></center>", unsafe_allow_html=True)