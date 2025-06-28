import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import base64
import io
import json
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Enterprise FP&A Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS Styling
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        padding: 0rem 1rem;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }

    .main-header p {
        color: #e8f4fd;
        font-size: 1.1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }

    .kpi-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }

    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        flex: 1;
        min-width: 200px;
        border-left: 4px solid #4CAF50;
        transition: transform 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }

    .kpi-card.negative {
        border-left-color: #f44336;
    }

    .kpi-card.warning {
        border-left-color: #ff9800;
    }

    .kpi-title {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .kpi-change {
        font-size: 0.85rem;
        font-weight: 500;
    }

    .kpi-change.positive {
        color: #4CAF50;
    }

    .kpi-change.negative {
        color: #f44336;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e3c72;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .exec-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    .insight-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
    }

    .perf-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .perf-excellent {
        background: #e8f5e8;
        color: #2e7d32;
    }

    .perf-good {
        background: #fff3e0;
        color: #f57c00;
    }

    .perf-poor {
        background: #ffebee;
        color: #c62828;
    }

    @media (max-width: 768px) {
        .kpi-container {
            flex-direction: column;
        }

        .main-header h1 {
            font-size: 2rem;
        }

        .kpi-value {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# Data Generation Functions
@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample data for the dashboard"""
    np.random.seed(42)

    # Date range for 3 fiscal years
    dates = pd.date_range(start='2022-07-01', end='2025-06-30', freq='M')

    # Base financial data
    base_revenue = 10000000  # $10M base
    revenue_data = []

    for i, date in enumerate(dates):
        # Seasonal patterns (higher in Q4, lower in Q1)
        month = date.month
        if month in [10, 11, 12]:  # Q4
            seasonal_factor = 1.3
        elif month in [1, 2, 3]:   # Q1  
            seasonal_factor = 0.8
        else:
            seasonal_factor = 1.0

        # Growth trend
        growth_factor = 1 + (0.08 * i / 12)  # 8% annual growth

        # Add some realistic noise
        noise = np.random.normal(0, 0.05)

        total_revenue = base_revenue * seasonal_factor * growth_factor * (1 + noise)

        revenue_data.append({
            'Date': date,
            'FY': f"FY{date.year}" if date.month >= 7 else f"FY{date.year-1}",
            'Month': date.strftime('%b'),
            'Quarter': f"Q{((date.month-1)//3)+1}",
            'Revenue': total_revenue,
            'Product_A': total_revenue * 0.45,
            'Product_B': total_revenue * 0.35,
            'Product_C': total_revenue * 0.20,
            'Region_North': total_revenue * 0.35,
            'Region_South': total_revenue * 0.25,
            'Region_East': total_revenue * 0.25,
            'Region_West': total_revenue * 0.15,
            'Channel_Direct': total_revenue * 0.60,
            'Channel_Partner': total_revenue * 0.40
        })

    revenue_df = pd.DataFrame(revenue_data)

    # Cost data with realistic cost structure
    cost_data = []
    for i, row in revenue_df.iterrows():
        revenue = row['Revenue']

        # Cost structure based on revenue
        cogs = revenue * (0.45 + np.random.normal(0, 0.02))  # 45% +/- 2%
        rd_cost = revenue * (0.12 + np.random.normal(0, 0.01))  # 12% +/- 1%
        sales_marketing = revenue * (0.18 + np.random.normal(0, 0.015))  # 18% +/- 1.5%
        general_admin = revenue * (0.08 + np.random.normal(0, 0.005))  # 8% +/- 0.5%
        other_opex = revenue * (0.03 + np.random.normal(0, 0.005))  # 3% +/- 0.5%

        cost_data.append({
            'Date': row['Date'],
            'FY': row['FY'],
            'Month': row['Month'],
            'COGS': max(0, cogs),
            'RD': max(0, rd_cost),
            'Sales_Marketing': max(0, sales_marketing),
            'General_Admin': max(0, general_admin),
            'Other_OpEx': max(0, other_opex),
            'Total_OpEx': max(0, rd_cost + sales_marketing + general_admin + other_opex),
            'Total_Cost': max(0, cogs + rd_cost + sales_marketing + general_admin + other_opex)
        })

    cost_df = pd.DataFrame(cost_data)

    # Calculate P&L metrics
    pnl_data = []
    for i in range(len(revenue_df)):
        revenue = revenue_df.iloc[i]['Revenue']
        costs = cost_df.iloc[i]

        gross_profit = revenue - costs['COGS']
        operating_profit = gross_profit - costs['Total_OpEx']

        pnl_data.append({
            'Date': revenue_df.iloc[i]['Date'],
            'FY': revenue_df.iloc[i]['FY'],
            'Revenue': revenue,
            'COGS': costs['COGS'],
            'Gross_Profit': gross_profit,
            'Total_OpEx': costs['Total_OpEx'],
            'Operating_Profit': operating_profit,
            'Gross_Margin': (gross_profit / revenue) * 100,
            'Operating_Margin': (operating_profit / revenue) * 100,
            'EBITDA': operating_profit * 1.15  # Approximation
        })

    pnl_df = pd.DataFrame(pnl_data)

    # Employee and productivity data
    employee_data = []
    base_employees = 750

    for i, date in enumerate(dates):
        employees = base_employees + (i * 8) + np.random.randint(-15, 15)
        revenue = revenue_df.iloc[i]['Revenue']

        employee_data.append({
            'Date': date,
            'FY': revenue_df.iloc[i]['FY'],
            'Total_Employees': employees,
            'Sales_Team': int(employees * 0.25),
            'Engineering': int(employees * 0.40),
            'Marketing': int(employees * 0.15),
            'Operations': int(employees * 0.20),
            'Revenue_Per_Employee': revenue / employees,
            'Cost_Per_Employee': cost_df.iloc[i]['Total_Cost'] / employees
        })

    employee_df = pd.DataFrame(employee_data)

    return revenue_df, cost_df, pnl_df, employee_df

# Advanced Visualization Functions
def create_waterfall_chart(data, categories, title, format_type='currency'):
    """Create professional waterfall chart"""
    fig = go.Figure(go.Waterfall(
        name="Financial Flow",
        orientation="v",
        measure=["relative"] * (len(categories)-1) + ["total"],
        x=categories,
        textposition="outside",
        text=[f"${x/1000000:.1f}M" if format_type=='currency' else f"{x:.1f}%" for x in data],
        y=data,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2E8B57"}},
        decreasing={"marker": {"color": "#DC143C"}},
        totals={"marker": {"color": "#1e3c72"}}
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18, "color": "#1e3c72"}},
        showlegend=False,
        height=500,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def create_advanced_heatmap(data, title):
    """Create correlation heatmap with custom styling"""
    corr_matrix = data.corr()

    # Create custom colorscale
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hoverongaps=False,
        colorbar=dict(title="Correlation", titleside="right")
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18, "color": "#1e3c72"}},
        height=500,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def create_advanced_treemap(data, values, labels, parents, title):
    """Create interactive treemap with drill-down capability"""
    colors = px.colors.qualitative.Set3

    fig = go.Figure(go.Treemap(
        labels=labels,
        values=values,
        parents=parents,
        textinfo="label+value+percent parent",
        textfont_size=12,
        marker=dict(
            colorscale='Viridis',
            colorbar=dict(thickness=15),
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Percentage: %{percentParent}<extra></extra>'
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18, "color": "#1e3c72"}},
        height=600,
        font=dict(size=12)
    )

    return fig

def create_sankey_flow_diagram(source, target, value, labels, title):
    """Create advanced Sankey diagram for flow analysis"""
    # Color nodes based on type
    node_colors = ['#1e3c72', '#4CAF50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4']

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors[:len(labels)]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(30, 60, 114, 0.3)'
        )
    )])

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        font_size=12,
        height=500,
        paper_bgcolor='white'
    )

    return fig

def create_performance_gauge(value, title, range_max=100, target=80):
    """Create gauge chart for performance indicators"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        delta = {'reference': target, 'increasing': {'color': "#2E8B57"}, 'decreasing': {'color': "#DC143C"}},
        gauge = {
            'axis': {'range': [None, range_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1e3c72"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, target*0.7], 'color': '#ffebee'},
                {'range': [target*0.7, target], 'color': '#fff3e0'},
                {'range': [target, range_max], 'color': '#e8f5e8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target
            }
        }
    ))

    fig.update_layout(
        height=300,
        font={'color': "#1e3c72", 'family': "Inter"}
    )

    return fig

def create_multi_line_chart(data, x_col, y_cols, title, colors=None):
    """Create advanced multi-line chart with custom styling"""
    if colors is None:
        colors = ['#1e3c72', '#4CAF50', '#ff9800', '#f44336', '#9c27b0']

    fig = go.Figure()

    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[col],
            mode='lines+markers',
            name=col.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=6),
            hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18, "color": "#1e3c72"}},
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


# KPI Cards and Utility Functions
def create_executive_kpi_cards(revenue_df, cost_df, pnl_df):
    """Create executive KPI cards with performance indicators"""
    # Get latest and previous period data
    latest_data = {
        'revenue': revenue_df['Revenue'].iloc[-1],
        'revenue_prev': revenue_df['Revenue'].iloc[-13] if len(revenue_df) > 12 else revenue_df['Revenue'].iloc[0],
        'gross_margin': pnl_df['Gross_Margin'].iloc[-1],
        'gross_margin_prev': pnl_df['Gross_Margin'].iloc[-13] if len(pnl_df) > 12 else pnl_df['Gross_Margin'].iloc[0],
        'operating_margin': pnl_df['Operating_Margin'].iloc[-1],
        'operating_margin_prev': pnl_df['Operating_Margin'].iloc[-13] if len(pnl_df) > 12 else pnl_df['Operating_Margin'].iloc[0],
        'total_cost': cost_df['Total_Cost'].iloc[-1],
        'total_cost_prev': cost_df['Total_Cost'].iloc[-13] if len(cost_df) > 12 else cost_df['Total_Cost'].iloc[0]
    }

    # Calculate changes
    revenue_change = ((latest_data['revenue'] - latest_data['revenue_prev']) / latest_data['revenue_prev']) * 100
    margin_change = latest_data['gross_margin'] - latest_data['gross_margin_prev']
    op_margin_change = latest_data['operating_margin'] - latest_data['operating_margin_prev']
    cost_change = ((latest_data['total_cost'] - latest_data['total_cost_prev']) / latest_data['total_cost_prev']) * 100

    kpis = [
        {
            'title': 'Total Revenue',
            'value': f"${latest_data['revenue']/1000000:.1f}M",
            'change': revenue_change,
            'change_text': f"{'↗' if revenue_change > 0 else '↘'} {abs(revenue_change):.1f}%",
            'card_type': 'positive' if revenue_change > 0 else 'negative' if revenue_change < -5 else 'warning'
        },
        {
            'title': 'Gross Margin',
            'value': f"{latest_data['gross_margin']:.1f}%",
            'change': margin_change,
            'change_text': f"{'↗' if margin_change > 0 else '↘'} {abs(margin_change):.1f}pp",
            'card_type': 'positive' if margin_change > 0 else 'negative' if margin_change < -2 else 'warning'
        },
        {
            'title': 'Operating Margin',
            'value': f"{latest_data['operating_margin']:.1f}%",
            'change': op_margin_change,
            'change_text': f"{'↗' if op_margin_change > 0 else '↘'} {abs(op_margin_change):.1f}pp",
            'card_type': 'positive' if op_margin_change > 0 else 'negative' if op_margin_change < -2 else 'warning'
        },
        {
            'title': 'Total Costs',
            'value': f"${latest_data['total_cost']/1000000:.1f}M",
            'change': cost_change,
            'change_text': f"{'↗' if cost_change > 0 else '↘'} {abs(cost_change):.1f}%",
            'card_type': 'negative' if cost_change > 5 else 'warning' if cost_change > 0 else 'positive'
        }
    ]

    cols = st.columns(len(kpis))

    for i, kpi in enumerate(kpis):
        with cols[i]:
            change_class = 'positive' if kpi['change'] > 0 else 'negative'
            if kpi['title'] == 'Total Costs':  # Inverse logic for costs
                change_class = 'negative' if kpi['change'] > 0 else 'positive'

            st.markdown(f"""
            <div class="kpi-card {kpi['card_type']}">
                <div class="kpi-title">{kpi['title']}</div>
                <div class="kpi-value">{kpi['value']}</div>
                <div class="kpi-change {change_class}">
                    {kpi['change_text']} vs last year
                </div>
            </div>
            """, unsafe_allow_html=True)

def format_currency(value, decimals=0):
    """Format currency values"""
    if abs(value) >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"

def format_percentage(value, decimals=1):
    """Format percentage values"""
    return f"{value:.{decimals}f}%"

def get_performance_indicator(value, thresholds):
    """Get performance indicator based on thresholds"""
    if value >= thresholds['excellent']:
        return 'perf-excellent', 'EXCELLENT'
    elif value >= thresholds['good']:
        return 'perf-good', 'GOOD'
    else:
        return 'perf-poor', 'NEEDS IMPROVEMENT'

# Export Functions
def export_to_excel(dataframes, filename="financial_report.xlsx"):
    """Export multiple dataframes to Excel with formatting"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter', options={'remove_timezone': True}) as writer:
        workbook = writer.book

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1e3c72',
            'font_color': 'white',
            'border': 1
        })

        currency_format = workbook.add_format({
            'num_format': '$#,##0',
            'border': 1
        })

        percentage_format = workbook.add_format({
            'num_format': '0.0%',
            'border': 1
        })

        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
            worksheet = writer.sheets[sheet_name]

            # Format headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Auto-adjust column widths
            for i, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(i, i, min(max_length, 50))

    return output.getvalue()

def export_to_csv(df, filename="data_export.csv"):
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def create_executive_summary_report(revenue_df, cost_df, pnl_df, employee_df):
    """Create executive summary text report"""
    latest_revenue = revenue_df['Revenue'].iloc[-1]
    latest_margin = pnl_df['Gross_Margin'].iloc[-1]
    yoy_growth = ((revenue_df['Revenue'].iloc[-1] - revenue_df['Revenue'].iloc[-13]) / revenue_df['Revenue'].iloc[-13]) * 100 if len(revenue_df) > 12 else 0

    report = f"""
EXECUTIVE SUMMARY REPORT
Generated: {datetime.now().strftime('%B %d, %Y')}

KEY FINANCIAL METRICS:
• Total Revenue: {format_currency(latest_revenue, 1)}
• Gross Margin: {latest_margin:.1f}%
• YoY Growth: {yoy_growth:.1f}%
• Total Employees: {employee_df['Total_Employees'].iloc[-1]:,}

PERFORMANCE HIGHLIGHTS:
• Revenue growth {'exceeded' if yoy_growth > 10 else 'met' if yoy_growth > 5 else 'below'} expectations
• Gross margins {'improved' if latest_margin > 50 else 'stable' if latest_margin > 45 else 'under pressure'}
• Operational efficiency {'strong' if latest_margin > 50 else 'moderate'}

RECOMMENDATIONS:
• Continue focus on high-margin products
• Monitor cost inflation impacts
• Invest in operational efficiency initiatives
• Expand in growth markets
    """

    return report

# Advanced Analytics Functions
def calculate_financial_ratios(revenue_df, cost_df, pnl_df):
    """Calculate advanced financial ratios"""
    latest_data = {
        'revenue': revenue_df['Revenue'].iloc[-1],
        'gross_profit': pnl_df['Gross_Profit'].iloc[-1],
        'operating_profit': pnl_df['Operating_Profit'].iloc[-1],
        'total_cost': cost_df['Total_Cost'].iloc[-1]
    }

    ratios = {
        'Gross Margin %': (latest_data['gross_profit'] / latest_data['revenue']) * 100,
        'Operating Margin %': (latest_data['operating_profit'] / latest_data['revenue']) * 100,
        'Cost Ratio %': (latest_data['total_cost'] / latest_data['revenue']) * 100,
        'Efficiency Score': (latest_data['operating_profit'] / latest_data['total_cost']) * 100
    }

    return ratios

def calculate_trend_analysis(df, column, periods=12):
    """Calculate trend analysis for a given column"""
    if len(df) < periods:
        return None

    recent_data = df[column].tail(periods)
    trend_slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]

    trend_direction = "Increasing" if trend_slope > 0 else "Decreasing" if trend_slope < 0 else "Stable"
    trend_strength = "Strong" if abs(trend_slope) > recent_data.mean() * 0.05 else "Moderate" if abs(trend_slope) > recent_data.mean() * 0.02 else "Weak"

    return {
        'direction': trend_direction,
        'strength': trend_strength,
        'slope': trend_slope,
        'latest_value': recent_data.iloc[-1],
        'avg_value': recent_data.mean()
    }

def generate_insights(revenue_df, cost_df, pnl_df, employee_df):
    """Generate AI-like insights from the data"""
    insights = []

    # Revenue growth insight
    if len(revenue_df) > 12:
        yoy_growth = ((revenue_df['Revenue'].iloc[-1] - revenue_df['Revenue'].iloc[-13]) / revenue_df['Revenue'].iloc[-13]) * 100
        if yoy_growth > 15:
            insights.append("🚀 Exceptional revenue growth of {:.1f}% YoY indicates strong market position and effective growth strategies.".format(yoy_growth))
        elif yoy_growth > 5:
            insights.append("📈 Solid revenue growth of {:.1f}% YoY demonstrates healthy business expansion.".format(yoy_growth))
        else:
            insights.append("⚠️ Revenue growth of {:.1f}% YoY suggests need for growth acceleration initiatives.".format(yoy_growth))

    # Margin analysis
    latest_margin = pnl_df['Gross_Margin'].iloc[-1]
    if latest_margin > 55:
        insights.append("💪 Strong gross margin of {:.1f}% indicates excellent pricing power and cost management.".format(latest_margin))
    elif latest_margin > 45:
        insights.append("✅ Healthy gross margin of {:.1f}% shows good operational efficiency.".format(latest_margin))
    else:
        insights.append("🔍 Gross margin of {:.1f}% suggests opportunities for cost optimization and pricing review.".format(latest_margin))

    # Productivity insight
    latest_rev_per_employee = revenue_df['Revenue'].iloc[-1] / employee_df['Total_Employees'].iloc[-1]
    if latest_rev_per_employee > 150000:
        insights.append("🎯 High revenue per employee of ${:,.0f} demonstrates strong productivity and operational leverage.".format(latest_rev_per_employee))
    else:
        insights.append("📊 Revenue per employee of ${:,.0f} indicates potential for productivity improvements.".format(latest_rev_per_employee))

    return insights


# Main Dashboard Application
def main():
    """Main dashboard application"""
    load_css()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Enterprise FP&A Dashboard</h1>
        <p>Financial Planning & Analysis | Real-time Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Load sample data
    with st.spinner('Loading financial data...'):
        revenue_df, cost_df, pnl_df, employee_df = generate_sample_data()

    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Dashboard Controls")

    # Fiscal year filter
    fiscal_years = sorted(revenue_df['FY'].unique(), reverse=True)
    selected_fy = st.sidebar.selectbox("📅 Select Fiscal Year", fiscal_years, index=0)

    # Date range for detailed analysis
    date_range = st.sidebar.date_input(
        "📊 Analysis Period",
        value=(revenue_df['Date'].min(), revenue_df['Date'].max()),
        min_value=revenue_df['Date'].min(),
        max_value=revenue_df['Date'].max()
    )

    # Product filter
    products = ['All Products'] + [f'Product {x}' for x in ['A', 'B', 'C']]
    selected_product = st.sidebar.selectbox("🛍️ Product Focus", products)

    # Region filter
    regions = ['All Regions', 'North', 'South', 'East', 'West']
    selected_region = st.sidebar.selectbox("🌍 Regional View", regions)

    # Filter data based on selections
    filtered_revenue = revenue_df[
        (revenue_df['Date'] >= pd.to_datetime(date_range[0])) &
        (revenue_df['Date'] <= pd.to_datetime(date_range[1]))
    ]

    if selected_fy != 'All':
        filtered_revenue = filtered_revenue[filtered_revenue['FY'] == selected_fy]

    filtered_cost = cost_df[cost_df['Date'].isin(filtered_revenue['Date'])]
    filtered_pnl = pnl_df[pnl_df['Date'].isin(filtered_revenue['Date'])]
    filtered_employee = employee_df[employee_df['Date'].isin(filtered_revenue['Date'])]

    # Export section
    st.sidebar.markdown("## 📊 Export & Reports")

    export_data = {
        'Revenue_Analysis': filtered_revenue,
        'Cost_Analysis': filtered_cost,
        'PnL_Analysis': filtered_pnl,
        'Employee_Data': filtered_employee
    }

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("📥 Excel", key="export_excel"):
            excel_data = export_to_excel(export_data)
            st.download_button(
                label="⬇️ Download",
                data=excel_data,
                file_name=f"FPA_Report_{selected_fy}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col2:
        if st.button("📄 Report", key="export_report"):
            report_data = create_executive_summary_report(
                filtered_revenue, filtered_cost, filtered_pnl, filtered_employee
            ).encode('utf-8')
            st.download_button(
                label="⬇️ Download",
                data=report_data,
                file_name=f"Executive_Summary_{selected_fy}.txt",
                mime="text/plain"
            )

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Executive Summary", 
        "💰 Revenue Analysis", 
        "💸 Cost Management", 
        "🏢 SG&A Analysis", 
        "📊 Performance KPIs", 
        "📉 Trend Analysis"
    ])

    # Tab 1: Executive Summary
    with tab1:
        st.markdown("## 📈 Executive Dashboard")

        # Executive KPI Cards
        create_executive_kpi_cards(filtered_revenue, filtered_cost, filtered_pnl)

        # Key insights
        insights = generate_insights(filtered_revenue, filtered_cost, filtered_pnl, filtered_employee)

        st.markdown("""
        <div class="exec-summary">
            <h3>🎯 Key Business Insights</h3>
        </div>
        """, unsafe_allow_html=True)

        for insight in insights:
            st.markdown(f"""
            <div class="insight-card">
                {insight}
            </div>
            """, unsafe_allow_html=True)

        # Executive charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # P&L Waterfall
            if not filtered_pnl.empty:
                latest_revenue = filtered_revenue['Revenue'].iloc[-1]
                latest_costs = filtered_cost.iloc[-1]

                waterfall_data = [
                    latest_revenue,
                    -latest_costs['COGS'],
                    -latest_costs['Sales_Marketing'],
                    -latest_costs['RD'],
                    -latest_costs['General_Admin'],
                    latest_revenue - latest_costs['Total_Cost']
                ]

                categories = ['Revenue', 'COGS', 'Sales & Marketing', 'R&D', 'General & Admin', 'Operating Profit']

                fig_waterfall = create_waterfall_chart(waterfall_data, categories, "P&L Waterfall Analysis")
                st.plotly_chart(fig_waterfall, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Revenue trend
            fig_revenue_trend = create_multi_line_chart(
                filtered_revenue, 'Date', ['Revenue'], 
                "Revenue Trend", ['#1e3c72']
            )
            st.plotly_chart(fig_revenue_trend, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Financial ratios
        ratios = calculate_financial_ratios(filtered_revenue, filtered_cost, filtered_pnl)

        st.markdown("### 📊 Key Financial Ratios")
        ratio_cols = st.columns(len(ratios))

        for i, (ratio_name, ratio_value) in enumerate(ratios.items()):
            with ratio_cols[i]:
                st.metric(
                    label=ratio_name,
                    value=f"{ratio_value:.1f}%",
                    delta=f"{np.random.uniform(-2, 2):.1f}%" if 'Margin' in ratio_name else None
                )

    # Tab 2: Revenue Analysis
    with tab2:
        st.markdown("## 💰 Revenue Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Product revenue treemap
            if not filtered_revenue.empty:
                product_revenue = [
                    filtered_revenue['Product_A'].sum(),
                    filtered_revenue['Product_B'].sum(),
                    filtered_revenue['Product_C'].sum()
                ]

                fig_treemap = create_advanced_treemap(
                    product_revenue, product_revenue, 
                    ['Product A', 'Product B', 'Product C'],
                    [''] * 3, 'Revenue by Product'
                )
                st.plotly_chart(fig_treemap, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Regional performance
            if not filtered_revenue.empty:
                regional_data = pd.DataFrame({
                    'Region': ['North', 'South', 'East', 'West'],
                    'Revenue': [
                        filtered_revenue['Region_North'].sum(),
                        filtered_revenue['Region_South'].sum(),
                        filtered_revenue['Region_East'].sum(),
                        filtered_revenue['Region_West'].sum()
                    ]
                })

                fig_regional = px.bar(
                    regional_data, x='Region', y='Revenue',
                    title='Revenue by Region',
                    color='Revenue',
                    color_continuous_scale='Blues'
                )
                fig_regional.update_layout(
                    height=500,
                    title={'x': 0.5, 'font': {'size': 18, 'color': '#1e3c72'}},
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig_regional, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Revenue trend analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        if not filtered_revenue.empty:
            fig_multi_revenue = create_multi_line_chart(
                filtered_revenue, 'Date', 
                ['Product_A', 'Product_B', 'Product_C'],
                "Product Revenue Trends"
            )
            st.plotly_chart(fig_multi_revenue, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Revenue metrics
        st.markdown("### 📊 Revenue Metrics")

        if not filtered_revenue.empty:
            rev_cols = st.columns(4)

            total_revenue = filtered_revenue['Revenue'].sum()
            avg_monthly = filtered_revenue['Revenue'].mean()
            growth_rate = ((filtered_revenue['Revenue'].iloc[-1] - filtered_revenue['Revenue'].iloc[0]) / filtered_revenue['Revenue'].iloc[0]) * 100 if len(filtered_revenue) > 1 else 0

            with rev_cols[0]:
                st.metric("Total Revenue", format_currency(total_revenue, 1))

            with rev_cols[1]:
                st.metric("Monthly Average", format_currency(avg_monthly, 1))

            with rev_cols[2]:
                st.metric("Growth Rate", f"{growth_rate:.1f}%")

            with rev_cols[3]:
                revenue_per_employee = total_revenue / filtered_employee['Total_Employees'].sum() if not filtered_employee.empty else 0
                st.metric("Revenue/Employee", format_currency(revenue_per_employee, 0))

    # Tab 3: Cost Management
    with tab3:
        st.markdown("## 💸 Cost Management")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Cost structure pie chart
            if not filtered_cost.empty:
                latest_costs = filtered_cost.iloc[-1]
                cost_structure = pd.DataFrame({
                    'Category': ['COGS', 'Sales & Marketing', 'R&D', 'General & Admin', 'Other OpEx'],
                    'Amount': [
                        latest_costs['COGS'],
                        latest_costs['Sales_Marketing'],
                        latest_costs['RD'],
                        latest_costs['General_Admin'],
                        latest_costs['Other_OpEx']
                    ]
                })

                fig_cost_pie = px.pie(
                    cost_structure, values='Amount', names='Category',
                    title='Cost Structure Breakdown',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_cost_pie.update_layout(height=500)
                st.plotly_chart(fig_cost_pie, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Cost trends
            if not filtered_cost.empty:
                fig_cost_trend = create_multi_line_chart(
                    filtered_cost, 'Date',
                    ['COGS', 'Sales_Marketing', 'RD', 'General_Admin'],
                    "Cost Category Trends"
                )
                st.plotly_chart(fig_cost_trend, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Cost efficiency metrics
        st.markdown("### 📊 Cost Efficiency Analysis")

        if not filtered_cost.empty and not filtered_revenue.empty:
            latest_revenue = filtered_revenue['Revenue'].iloc[-1]
            latest_costs = filtered_cost.iloc[-1]

            eff_cols = st.columns(4)

            with eff_cols[0]:
                cogs_ratio = (latest_costs['COGS'] / latest_revenue) * 100
                st.metric("COGS Ratio", f"{cogs_ratio:.1f}%")

            with eff_cols[1]:
                sm_ratio = (latest_costs['Sales_Marketing'] / latest_revenue) * 100
                st.metric("Sales & Marketing", f"{sm_ratio:.1f}%")

            with eff_cols[2]:
                rd_ratio = (latest_costs['RD'] / latest_revenue) * 100
                st.metric("R&D Investment", f"{rd_ratio:.1f}%")

            with eff_cols[3]:
                total_opex_ratio = (latest_costs['Total_OpEx'] / latest_revenue) * 100
                st.metric("Total OpEx Ratio", f"{total_opex_ratio:.1f}%")

        # Cost correlation analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        if not filtered_cost.empty:
            cost_corr_data = filtered_cost[['COGS', 'Sales_Marketing', 'RD', 'General_Admin', 'Total_Cost']]
            fig_heatmap = create_advanced_heatmap(cost_corr_data, 'Cost Correlation Matrix')
            st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 4: SG&A Analysis
    with tab4:
        st.markdown("## 🏢 SG&A Analysis")

        if not filtered_cost.empty:
            latest_costs = filtered_cost.iloc[-1]

            # SG&A breakdown with Sankey
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Sankey diagram for SG&A flow
            total_sga = latest_costs['Sales_Marketing'] + latest_costs['General_Admin']

            source = [0, 0, 1, 1, 2, 2]
            target = [1, 2, 3, 4, 5, 6]
            values = [
                latest_costs['Sales_Marketing'],
                latest_costs['General_Admin'],
                latest_costs['Sales_Marketing'] * 0.6,
                latest_costs['Sales_Marketing'] * 0.4,
                latest_costs['General_Admin'] * 0.7,
                latest_costs['General_Admin'] * 0.3
            ]
            labels = ['Total SG&A', 'Sales & Marketing', 'General & Admin', 
                     'Direct Sales', 'Marketing Campaigns', 'Administration', 'Other General']

            fig_sankey = create_sankey_flow_diagram(source, target, values, labels, 'SG&A Cost Flow Analysis')
            st.plotly_chart(fig_sankey, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # SG&A performance indicators
            st.markdown("### 🎯 SG&A Performance Indicators")

            perf_cols = st.columns(3)

            with perf_cols[0]:
                sales_efficiency = (latest_costs['Sales_Marketing'] / filtered_revenue['Revenue'].iloc[-1]) * 100 if not filtered_revenue.empty else 0
                perf_class, perf_text = get_performance_indicator(
                    100 - sales_efficiency,
                    {'excellent': 85, 'good': 80}
                )

                st.markdown(f"""
                <div class="chart-container">
                    <h4>Sales Efficiency</h4>
                    <div class="perf-indicator {perf_class}">{perf_text}</div>
                    <p>Sales cost ratio: {sales_efficiency:.1f}%</p>
                    <p>Target: <15%</p>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[1]:
                admin_efficiency = (latest_costs['General_Admin'] / filtered_revenue['Revenue'].iloc[-1]) * 100 if not filtered_revenue.empty else 0
                perf_class, perf_text = get_performance_indicator(
                    100 - admin_efficiency,
                    {'excellent': 92, 'good': 90}
                )

                st.markdown(f"""
                <div class="chart-container">
                    <h4>Admin Efficiency</h4>
                    <div class="perf-indicator {perf_class}">{perf_text}</div>
                    <p>Admin cost ratio: {admin_efficiency:.1f}%</p>
                    <p>Target: <8%</p>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[2]:
                total_sga_ratio = (total_sga / filtered_revenue['Revenue'].iloc[-1]) * 100 if not filtered_revenue.empty else 0
                perf_class, perf_text = get_performance_indicator(
                    100 - total_sga_ratio,
                    {'excellent': 77, 'good': 75}
                )

                st.markdown(f"""
                <div class="chart-container">
                    <h4>Total SG&A Efficiency</h4>
                    <div class="perf-indicator {perf_class}">{perf_text}</div>
                    <p>SG&A ratio: {total_sga_ratio:.1f}%</p>
                    <p>Target: <25%</p>
                </div>
                """, unsafe_allow_html=True)

    # Tab 5: Performance KPIs
    with tab5:
        st.markdown("## 📊 Performance KPIs")

        if not filtered_pnl.empty and not filtered_employee.empty:
            # Performance gauges
            st.markdown("### 🎯 Key Performance Indicators")

            gauge_cols = st.columns(3)

            with gauge_cols[0]:
                gross_margin = filtered_pnl['Gross_Margin'].iloc[-1]
                fig_gauge1 = create_performance_gauge(
                    gross_margin, "Gross Margin %", 100, 50
                )
                st.plotly_chart(fig_gauge1, use_container_width=True)

            with gauge_cols[1]:
                operating_margin = filtered_pnl['Operating_Margin'].iloc[-1]
                fig_gauge2 = create_performance_gauge(
                    operating_margin, "Operating Margin %", 50, 15
                )
                st.plotly_chart(fig_gauge2, use_container_width=True)

            with gauge_cols[2]:
                revenue_per_employee = (filtered_revenue['Revenue'].iloc[-1] / filtered_employee['Total_Employees'].iloc[-1]) / 1000
                fig_gauge3 = create_performance_gauge(
                    revenue_per_employee, "Revenue per Employee (K$)", 200, 150
                )
                st.plotly_chart(fig_gauge3, use_container_width=True)

            # Performance metrics table
            st.markdown("### 📈 Detailed Performance Metrics")

            metrics_data = {
                'Metric': [
                    'Revenue Growth YoY',
                    'Gross Margin',
                    'Operating Margin',
                    'EBITDA Margin',
                    'Revenue per Employee',
                    'Cost per Employee',
                    'Employee Productivity'
                ],
                'Current': [
                    f"{((filtered_revenue['Revenue'].iloc[-1] / filtered_revenue['Revenue'].iloc[0]) - 1) * 100:.1f}%" if len(filtered_revenue) > 1 else "0.0%",
                    f"{gross_margin:.1f}%",
                    f"{operating_margin:.1f}%",
                    f"{(filtered_pnl['EBITDA'].iloc[-1] / filtered_revenue['Revenue'].iloc[-1]) * 100:.1f}%",
                    format_currency(filtered_revenue['Revenue'].iloc[-1] / filtered_employee['Total_Employees'].iloc[-1], 0),
                    format_currency(filtered_cost['Total_Cost'].iloc[-1] / filtered_employee['Total_Employees'].iloc[-1], 0),
                    f"{(filtered_revenue['Revenue'].sum() / filtered_employee['Total_Employees'].sum()):.0f}" if not filtered_employee.empty else "0"
                ],
                'Target': [
                    '10.0%', '50.0%', '15.0%', '18.0%', '$150K', '$100K', '140'
                ],
                'Status': [
                    '🟢' if ((filtered_revenue['Revenue'].iloc[-1] / filtered_revenue['Revenue'].iloc[0]) - 1) * 100 > 10 else '🟡' if ((filtered_revenue['Revenue'].iloc[-1] / filtered_revenue['Revenue'].iloc[0]) - 1) * 100 > 5 else '🔴' if len(filtered_revenue) > 1 else '🟡',
                    '🟢' if gross_margin > 50 else '🟡' if gross_margin > 45 else '🔴',
                    '🟢' if operating_margin > 15 else '🟡' if operating_margin > 10 else '🔴',
                    '🟢', '🟡', '🟡', '🟢'
                ]
            }

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Tab 6: Trend Analysis
    with tab6:
        st.markdown("## 📉 Trend Analysis")

        # Revenue and profitability trends
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            if not filtered_pnl.empty:
                fig_margin_trend = create_multi_line_chart(
                    filtered_pnl, 'Date',
                    ['Gross_Margin', 'Operating_Margin'],
                    "Margin Trends"
                )
                st.plotly_chart(fig_margin_trend, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            if not filtered_employee.empty:
                fig_productivity = create_multi_line_chart(
                    filtered_employee, 'Date',
                    ['Revenue_Per_Employee'],
                    "Productivity Trends"
                )
                st.plotly_chart(fig_productivity, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Trend analysis insights
        st.markdown("### 📊 Trend Analysis Summary")

        if not filtered_revenue.empty:
            revenue_trend = calculate_trend_analysis(filtered_revenue, 'Revenue')
            margin_trend = calculate_trend_analysis(filtered_pnl, 'Gross_Margin') if not filtered_pnl.empty else None

            trend_cols = st.columns(2)

            with trend_cols[0]:
                if revenue_trend:
                    st.markdown(f"""
                    <div class="chart-container">
                        <h4>Revenue Trend</h4>
                        <p><strong>Direction:</strong> {revenue_trend['direction']}</p>
                        <p><strong>Strength:</strong> {revenue_trend['strength']}</p>
                        <p><strong>Latest:</strong> {format_currency(revenue_trend['latest_value'], 1)}</p>
                        <p><strong>Average:</strong> {format_currency(revenue_trend['avg_value'], 1)}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with trend_cols[1]:
                if margin_trend:
                    st.markdown(f"""
                    <div class="chart-container">
                        <h4>Margin Trend</h4>
                        <p><strong>Direction:</strong> {margin_trend['direction']}</p>
                        <p><strong>Strength:</strong> {margin_trend['strength']}</p>
                        <p><strong>Latest:</strong> {margin_trend['latest_value']:.1f}%</p>
                        <p><strong>Average:</strong> {margin_trend['avg_value']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
