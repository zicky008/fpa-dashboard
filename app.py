import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FP&A Dashboard - Minimal Version",
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
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

def load_excel_data(uploaded_file):
    """Load and process Excel file with multiple sheets"""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(uploaded_file)
        sheets_data = {}

        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                if not df.empty:
                    sheets_data[sheet_name] = df
            except Exception as e:
                st.warning(f"Could not read sheet '{sheet_name}': {str(e)}")
                continue

        return sheets_data, list(sheets_data.keys())
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None, None

def clean_dataframe(df):
    """Basic data cleaning for any dataframe"""
    try:
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Convert date columns if they exist
        date_columns = [col for col in df.columns if any(keyword in str(col).lower() 
                       for keyword in ['date', 'time', 'created', 'updated', 'timestamp'])]

        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass

        # Identify numeric columns
        numeric_columns = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_columns.append(col)
                except:
                    pass

        return df, numeric_columns, date_columns
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return df, [], []

def create_summary_metrics(df, numeric_columns):
    """Create summary metrics from dataframe"""
    metrics = {}

    if not numeric_columns or df.empty:
        return metrics

    try:
        for col in numeric_columns[:6]:  # Limit to first 6 numeric columns
            if col in df.columns:
                col_data = pd.to_numeric(df[col], errors='coerce')
                if not col_data.isna().all():
                    metrics[col] = {
                        'sum': col_data.sum(),
                        'mean': col_data.mean(),
                        'count': col_data.count(),
                        'max': col_data.max(),
                        'min': col_data.min()
                    }
    except Exception as e:
        st.error(f"Error creating metrics: {str(e)}")

    return metrics

def create_visualizations(df, numeric_columns, date_columns, sheet_name):
    """Create various charts for the data"""
    charts = []

    if df.empty:
        return charts

    try:
        # 1. Bar chart for numeric data
        if len(numeric_columns) >= 1:
            col = numeric_columns[0]
            if col in df.columns:
                # Sample data if too large
                sample_df = df.head(20) if len(df) > 20 else df

                fig_bar = px.bar(
                    sample_df, 
                    y=col,
                    title=f"{sheet_name} - {col} Distribution (Top 20)",
                    template="plotly_white",
                    color_discrete_sequence=['#1f77b4']
                )
                fig_bar.update_layout(height=400, showlegend=False)
                charts.append(("Bar Chart", fig_bar))

        # 2. Time series chart if date column exists
        if date_columns and len(numeric_columns) >= 1:
            date_col = date_columns[0]
            value_col = numeric_columns[0]

            if date_col in df.columns and value_col in df.columns:
                # Sort by date and sample if needed
                df_time = df[[date_col, value_col]].dropna()
                df_time = df_time.sort_values(date_col)

                if len(df_time) > 100:
                    df_time = df_time.iloc[::len(df_time)//100]  # Sample evenly

                fig_line = px.line(
                    df_time,
                    x=date_col,
                    y=value_col,
                    title=f"{sheet_name} - {value_col} Trend Over Time",
                    template="plotly_white",
                    color_discrete_sequence=['#2ca02c']
                )
                fig_line.update_layout(height=400)
                charts.append(("Time Series", fig_line))

        # 3. Pie chart for categorical data
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns and len(numeric_columns) >= 1:
            cat_col = categorical_columns[0]
            value_col = numeric_columns[0]

            if cat_col in df.columns and value_col in df.columns:
                # Group by category and sum values
                pie_data = df.groupby(cat_col)[value_col].sum().head(10)

                if len(pie_data) > 1 and pie_data.sum() > 0:
                    fig_pie = px.pie(
                        values=pie_data.values,
                        names=pie_data.index,
                        title=f"{sheet_name} - {value_col} by {cat_col}",
                        template="plotly_white"
                    )
                    fig_pie.update_layout(height=400)
                    charts.append(("Category Distribution", fig_pie))

        # 4. Correlation heatmap if multiple numeric columns
        if len(numeric_columns) >= 2:
            numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
            if not numeric_df.empty and len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()

                fig_heatmap = px.imshow(
                    corr_matrix,
                    title=f"{sheet_name} - Correlation Matrix",
                    template="plotly_white",
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                fig_heatmap.update_layout(height=400)
                charts.append(("Correlation Heatmap", fig_heatmap))

        # 5. Box plot for distribution analysis
        if len(numeric_columns) >= 1:
            col = numeric_columns[0]
            if col in df.columns:
                fig_box = px.box(
                    df, 
                    y=col,
                    title=f"{sheet_name} - {col} Distribution Analysis",
                    template="plotly_white",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_box.update_layout(height=400)
                charts.append(("Distribution Analysis", fig_box))

    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")

    return charts

def create_data_summary_table(df):
    """Create a summary table of the dataframe"""
    try:
        summary_data = []

        for col in df.columns:
            col_info = {
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Unique Values': df[col].nunique()
            }

            # Add sample values for non-numeric columns
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                col_info['Sample Values'] = ', '.join(map(str, unique_vals[:3]))
            else:
                col_info['Sample Values'] = f"Min: {df[col].min()}, Max: {df[col].max()}"

            summary_data.append(col_info)

        return pd.DataFrame(summary_data)
    except Exception as e:
        st.error(f"Error creating summary table: {str(e)}")
        return pd.DataFrame()

def main():
    # Header
    st.markdown('<div class="main-header">📊 FP&A Dashboard - Minimal Version</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")

    # File upload section
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Excel file with FP&A data",
            type=['xlsx', 'xls'],
            help="Upload Excel file containing your financial data sheets"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Load data
        with st.spinner("🔄 Loading Excel data..."):
            sheets_data, sheet_names = load_excel_data(uploaded_file)

        if sheets_data is not None and sheet_names:
            st.success(f"✅ Successfully loaded {len(sheet_names)} sheets: {', '.join(sheet_names)}")

            # Sidebar sheet selection
            selected_sheet = st.sidebar.selectbox(
                "📊 Select Sheet to Analyze",
                sheet_names,
                help="Choose which sheet to display and analyze"
            )

            # Analysis options
            st.sidebar.markdown("### 🔧 Analysis Options")
            show_raw_data = st.sidebar.checkbox("Show Raw Data", value=True)
            show_summary = st.sidebar.checkbox("Show Data Summary", value=True)
            show_metrics = st.sidebar.checkbox("Show Key Metrics", value=True)
            show_charts = st.sidebar.checkbox("Show Visualizations", value=True)

            # Display selected sheet data
            if selected_sheet in sheets_data:
                df = sheets_data[selected_sheet]

                st.markdown(f'<div class="section-header">📋 {selected_sheet} - Data Analysis</div>', unsafe_allow_html=True)

                # Clean and process data
                with st.spinner("🧹 Processing data..."):
                    cleaned_df, numeric_columns, date_columns = clean_dataframe(df.copy())

                # Basic info metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📊 Total Rows", f"{len(cleaned_df):,}")

                with col2:
                    st.metric("📋 Total Columns", len(cleaned_df.columns))

                with col3:
                    st.metric("🔢 Numeric Columns", len(numeric_columns))

                with col4:
                    missing_data = cleaned_df.isnull().sum().sum()
                    st.metric("❓ Missing Values", f"{missing_data:,}")

                # Key metrics section
                if show_metrics and numeric_columns:
                    st.markdown('<div class="section-header">📊 Key Financial Metrics</div>', unsafe_allow_html=True)

                    metrics = create_summary_metrics(cleaned_df, numeric_columns)

                    if metrics:
                        # Display metrics in columns
                        metric_cols = st.columns(min(len(metrics), 3))
                        for idx, (col_name, metric_data) in enumerate(metrics.items()):
                            if idx < 3:  # Limit to 3 columns for better layout
                                with metric_cols[idx]:
                                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)

                                    # Format large numbers
                                    sum_val = metric_data['sum']
                                    if abs(sum_val) >= 1e9:
                                        display_sum = f"{sum_val/1e9:.1f}B"
                                    elif abs(sum_val) >= 1e6:
                                        display_sum = f"{sum_val/1e6:.1f}M"
                                    elif abs(sum_val) >= 1e3:
                                        display_sum = f"{sum_val/1e3:.1f}K"
                                    else:
                                        display_sum = f"{sum_val:,.0f}"

                                    st.metric(
                                        label=f"💰 {col_name}",
                                        value=display_sum
                                    )

                                    # Additional info
                                    avg_val = metric_data['mean']
                                    if abs(avg_val) >= 1e6:
                                        display_avg = f"{avg_val/1e6:.1f}M"
                                    elif abs(avg_val) >= 1e3:
                                        display_avg = f"{avg_val/1e3:.1f}K"
                                    else:
                                        display_avg = f"{avg_val:,.0f}"

                                    st.caption(f"Average: {display_avg}")
                                    st.caption(f"Records: {metric_data['count']:,}")
                                    st.markdown('</div>', unsafe_allow_html=True)

                # Data summary section
                if show_summary:
                    st.markdown('<div class="section-header">📋 Data Summary</div>', unsafe_allow_html=True)

                    summary_table = create_data_summary_table(cleaned_df)
                    if not summary_table.empty:
                        st.dataframe(
                            summary_table,
                            use_container_width=True,
                            height=300
                        )

                # Raw data preview
                if show_raw_data:
                    st.markdown('<div class="section-header">👀 Data Preview</div>', unsafe_allow_html=True)

                    # Show sample of data
                    preview_rows = st.slider("Number of rows to preview", 5, 50, 10)
                    st.dataframe(
                        cleaned_df.head(preview_rows),
                        use_container_width=True,
                        height=400
                    )

                # Visualizations section
                if show_charts and (numeric_columns or date_columns):
                    st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)

                    with st.spinner("🎨 Creating visualizations..."):
                        charts = create_visualizations(cleaned_df, numeric_columns, date_columns, selected_sheet)

                    if charts:
                        # Display charts in tabs
                        tab_names = [chart[0] for chart in charts]
                        tabs = st.tabs(tab_names)

                        for idx, (chart_name, fig) in enumerate(charts):
                            with tabs[idx]:
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📊 No suitable data found for visualization")

                # Export section
                st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    # CSV export
                    csv_buffer = io.StringIO()
                    cleaned_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=f"{selected_sheet}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    # JSON export
                    json_data = cleaned_df.to_json(orient='records', indent=2)

                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"{selected_sheet}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

    else:
        # Instructions when no file is uploaded
        st.markdown('<div class="section-header">🚀 Getting Started</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Welcome to the FP&A Dashboard!**

            This streamlined version provides comprehensive financial data analysis:

            **📊 Key Features:**
            - 📁 **Multi-sheet Excel processing** - Upload files with multiple worksheets
            - 📈 **Automatic visualizations** - Charts generated based on your data structure
            - 💰 **Financial metrics** - Key performance indicators and summaries
            - 🔍 **Data exploration** - Interactive data preview and analysis
            - 💾 **Export capabilities** - Download processed data in CSV/JSON formats

            **🎯 Perfect for:**
            - Financial Planning & Analysis teams
            - Budget vs Actual analysis
            - Revenue and cost tracking
            - Department performance analysis
            - Multi-period financial reporting
            """)

        with col2:
            st.markdown("""
            **📋 Supported Data Types:**
            - Mastersale sheets
            - FBMC data
            - Department budgets
            - Revenue reports
            - Cost analysis
            - Any Excel financial data

            **🔧 Analysis Capabilities:**
            - Trend analysis
            - Distribution charts
            - Correlation analysis
            - Summary statistics
            - Missing data detection
            """)

        # Sample data demonstration
        with st.expander("📊 View Sample Analysis Demo"):
            st.write("**Sample Financial Data Analysis:**")

            # Create realistic sample financial data
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=12, freq='M')

            sample_data = pd.DataFrame({
                'Month': dates,
                'Revenue': np.random.randint(80000, 120000, 12),
                'Operating_Costs': np.random.randint(40000, 70000, 12),
                'Marketing_Spend': np.random.randint(8000, 15000, 12),
                'Department': np.random.choice(['Sales', 'Marketing', 'Operations', 'Finance'], 12),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 12)
            })

            sample_data['Gross_Profit'] = sample_data['Revenue'] - sample_data['Operating_Costs']
            sample_data['Profit_Margin'] = (sample_data['Gross_Profit'] / sample_data['Revenue'] * 100).round(1)

            st.dataframe(sample_data, use_container_width=True)

            # Sample visualization
            fig_demo = px.line(
                sample_data, 
                x='Month', 
                y=['Revenue', 'Operating_Costs', 'Gross_Profit'],
                title="Sample Financial Performance Trend",
                template="plotly_white"
            )
            fig_demo.update_layout(height=400)
            st.plotly_chart(fig_demo, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666666; padding: 1rem;'>"
        "🏢 FP&A Dashboard - Professional Financial Analysis Tool<br>"
        "Built with Streamlit, Pandas & Plotly | Version 1.0"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
