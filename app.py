import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
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
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }

    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e8b57;
    }

    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class FPADashboard:
    def __init__(self):
        self.data = {}
        self.processed_data = {}
        self.ai_insights_available = self.check_ai_availability()

    def check_ai_availability(self):
        """Check if AI insights are available"""
        try:
            from transformers import pipeline
            return True
        except ImportError:
            return False

    def generate_sample_data(self):
        """Generate comprehensive sample data for demo purposes"""
        np.random.seed(42)

        # Sample Mastersale data with realistic FMCG products
        months = pd.date_range('2024-04-01', '2025-03-31', freq='M')
        products = [
            'Premium Coffee Blend', 'Instant Noodles Classic', 'Energy Drink Pro', 
            'Protein Snack Bars', 'Organic Dairy Milk', 'Green Tea Extract',
            'Vitamin Water Plus', 'Breakfast Cereal Crunch', 'Fruit Juice Natural',
            'Chocolate Cookies Premium'
        ]
        channels = ['Modern Trade', 'Traditional Trade', 'E-commerce', 'Export', 'Direct Sales']
        regions = ['North Region', 'South Region', 'Central Region', 'East Region', 'West Region']

        mastersale_data = []
        for month in months:
            for product in products:
                for channel in channels:
                    for region in regions:
                        # Realistic revenue patterns with seasonality
                        base_revenue = np.random.normal(150000, 30000)
                        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month.month / 12)
                        channel_factor = {
                            'Modern Trade': 1.4, 'Traditional Trade': 1.0, 
                            'E-commerce': 1.2, 'Export': 0.8, 'Direct Sales': 0.9
                        }.get(channel, 1.0)

                        revenue = max(0, base_revenue * seasonal_factor * channel_factor)
                        volume = np.random.normal(1500, 300)
                        volume = max(0, volume)

                        mastersale_data.append({
                            'Date': month,
                            'Product': product,
                            'Channel': channel,
                            'Region': region,
                            'Revenue': revenue,
                            'Volume': volume,
                            'ASP': revenue/volume if volume > 0 else 0,
                            'Units_Sold': int(volume * np.random.uniform(0.8, 1.2))
                        })

        # Sample FBMC data with detailed 5-level cost hierarchy
        cost_categories = {
            'COGS': {
                'Raw Materials': {
                    'Primary Ingredients': ['Coffee Beans', 'Wheat Flour', 'Sugar', 'Milk Powder'],
                    'Secondary Ingredients': ['Flavoring', 'Preservatives', 'Vitamins', 'Minerals'],
                    'Packaging Materials': ['Bottles', 'Labels', 'Caps', 'Boxes']
                },
                'Manufacturing': {
                    'Production Labor': ['Direct Labor', 'Supervisory', 'Quality Control', 'Maintenance'],
                    'Equipment & Machinery': ['Depreciation', 'Repairs', 'Upgrades', 'Calibration'],
                    'Utilities': ['Electricity', 'Water', 'Gas', 'Steam']
                },
                'Quality Assurance': {
                    'Testing & Analysis': ['Lab Testing', 'Microbiological', 'Chemical Analysis', 'Sensory'],
                    'Compliance': ['Regulatory', 'Certification', 'Audits', 'Documentation'],
                    'Equipment': ['Testing Equipment', 'Calibration', 'Maintenance', 'Upgrades']
                }
            },
            'SGA': {
                'Marketing & Advertising': {
                    'Digital Marketing': ['Social Media', 'Search Engine', 'Display Ads', 'Email Marketing'],
                    'Traditional Advertising': ['TV Commercials', 'Print Ads', 'Radio', 'Outdoor'],
                    'Brand Management': ['Brand Strategy', 'Creative Development', 'Brand Research', 'Events'],
                    'Trade Marketing': ['Trade Promotions', 'In-store Displays', 'Trade Shows', 'Merchandising']
                },
                'Sales Operations': {
                    'Sales Team': ['Salaries', 'Commissions', 'Bonuses', 'Training'],
                    'Sales Support': ['CRM Systems', 'Sales Tools', 'Presentations', 'Samples'],
                    'Customer Service': ['Call Center', 'Online Support', 'Field Service', 'Returns'],
                    'Distribution': ['Logistics', 'Warehousing', 'Transportation', 'Inventory']
                },
                'General & Administrative': {
                    'Human Resources': ['Recruitment', 'Training', 'Benefits', 'Payroll'],
                    'Finance & Accounting': ['Accounting', 'Financial Planning', 'Audit', 'Tax'],
                    'Legal & Compliance': ['Legal Fees', 'Contracts', 'IP Protection', 'Regulatory'],
                    'IT & Technology': ['Software Licenses', 'Hardware', 'Support', 'Development']
                }
            },
            'Other Operating': {
                'Research & Development': {
                    'Product Development': ['New Products', 'Formulation', 'Testing', 'Prototyping'],
                    'Innovation': ['Technology Research', 'Trend Analysis', 'Consumer Insights', 'Patents'],
                    'R&D Infrastructure': ['Lab Equipment', 'Facilities', 'Personnel', 'External Research']
                },
                'Corporate Functions': {
                    'Executive Management': ['CEO Office', 'Board Costs', 'Strategic Planning', 'Governance'],
                    'Corporate Development': ['M&A', 'Partnerships', 'Market Expansion', 'Investment'],
                    'Risk Management': ['Insurance', 'Risk Assessment', 'Compliance', 'Security']
                }
            }
        }

        fbmc_data = []
        for month in months:
            for l1_cat, l2_dict in cost_categories.items():
                for l2_cat, l3_dict in l2_dict.items():
                    for l3_cat, l4_list in l3_dict.items():
                        for l4_cat in l4_list:
                            # Realistic cost patterns
                            base_cost = np.random.normal(75000, 15000)
                            seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * month.month / 12)
                            category_factor = {
                                'COGS': 1.5, 'SGA': 1.2, 'Other Operating': 0.8
                            }.get(l1_cat, 1.0)

                            cost = max(0, base_cost * seasonal_factor * category_factor)

                            fbmc_data.append({
                                'Date': month,
                                'Cost_Category_L1': l1_cat,
                                'Cost_Category_L2': l2_cat,
                                'Cost_Category_L3': l3_cat,
                                'Cost_Category_L4': l4_cat,
                                'Cost_Category_L5': f"{l4_cat}_Activity",
                                'Amount': cost,
                                'Budget': cost * np.random.uniform(0.95, 1.15),
                                'Prior_Year': cost * np.random.uniform(0.85, 1.05)
                            })

        # Sample Department data with detailed activities
        departments = [
            'Marketing', 'Sales', 'Operations', 'Finance', 'Human Resources', 
            'IT', 'R&D', 'Quality Assurance', 'Supply Chain', 'Customer Service'
        ]
        activities = [
            'Strategic Planning', 'Operational Execution', 'Performance Analysis', 
            'Reporting & Documentation', 'Training & Development', 'Process Improvement',
            'Compliance & Audit', 'Technology & Innovation', 'Customer Engagement', 'Risk Management'
        ]

        dept_data = []
        for month in months:
            for dept in departments:
                for activity in activities:
                    # Department-specific cost patterns
                    base_expense = np.random.normal(35000, 7000)
                    dept_factor = {
                        'Marketing': 1.6, 'Sales': 1.4, 'Operations': 1.3, 'R&D': 1.2,
                        'Finance': 1.0, 'IT': 1.1, 'Human Resources': 0.9, 'Quality Assurance': 0.8,
                        'Supply Chain': 1.0, 'Customer Service': 0.7
                    }.get(dept, 1.0)

                    expense = max(0, base_expense * dept_factor)
                    budget = expense * np.random.uniform(0.9, 1.25)
                    variance = expense - budget

                    dept_data.append({
                        'Date': month,
                        'Department': dept,
                        'Activity': activity,
                        'Expense': expense,
                        'Budget': budget,
                        'Variance': variance,
                        'FTE': np.random.randint(2, 15),
                        'Cost_per_FTE': expense / max(1, np.random.randint(2, 15))
                    })

        return {
            'Mastersale': pd.DataFrame(mastersale_data),
            'FBMC': pd.DataFrame(fbmc_data),
            'Dept': pd.DataFrame(dept_data)
        }

    def parse_fiscal_year(self, fy_string):
        """Parse fiscal year string like FY2425 to 2024-2025"""
        try:
            if isinstance(fy_string, str) and fy_string.upper().startswith('FY') and len(fy_string) == 6:
                year_part = fy_string[2:]
                start_year = 2000 + int(year_part[:2])
                end_year = 2000 + int(year_part[2:])
                return f"{start_year}-{end_year}"
            return fy_string
        except (ValueError, IndexError):
            return fy_string

    def process_excel_file(self, uploaded_file):
        """Process uploaded Excel file with multiple sheets"""
        try:
            # Read all sheets
            excel_data = pd.read_excel(uploaded_file, sheet_name=None)

            processed_sheets = {}
            for sheet_name, df in excel_data.items():
                # Convert date columns
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year']):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass

                # Parse fiscal year columns
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['fy', 'fiscal']):
                        df[col] = df[col].apply(self.parse_fiscal_year)

                processed_sheets[sheet_name] = df

            return processed_sheets

        except Exception as e:
            st.error(f"Error processing Excel file: {str(e)}")
            return None


    def create_cost_hierarchy_mapping(self, fbmc_data):
        """Create and validate 5-level cost hierarchy mapping"""
        if fbmc_data is None or fbmc_data.empty:
            return pd.DataFrame()

        # Ensure all 5 levels exist
        for i in range(1, 6):
            col_name = f'Cost_Category_L{i}'
            if col_name not in fbmc_data.columns:
                if i == 1:
                    fbmc_data[col_name] = 'General'
                elif i == 2:
                    fbmc_data[col_name] = fbmc_data.get('Cost_Category_L1', 'General') + '_Sub'
                else:
                    fbmc_data[col_name] = fbmc_data.get(f'Cost_Category_L{i-1}', 'General') + f'_L{i}'

        return fbmc_data

    def analyze_sga_by_department(self, dept_data, fbmc_data):
        """Analyze SG&A costs by department and activity"""
        analysis = {}

        try:
            if dept_data is not None and not dept_data.empty:
                # Department analysis
                dept_summary = dept_data.groupby(['Department', 'Activity']).agg({
                    'Expense': 'sum',
                    'Budget': 'sum',
                    'Variance': 'sum',
                    'FTE': 'mean',
                    'Cost_per_FTE': 'mean'
                }).reset_index()

                dept_summary['Variance_Pct'] = np.where(
                    dept_summary['Budget'] != 0,
                    (dept_summary['Variance'] / dept_summary['Budget'] * 100).round(2),
                    0
                )
                analysis['department_summary'] = dept_summary

                # Monthly trend analysis
                monthly_dept = dept_data.groupby(['Date', 'Department']).agg({
                    'Expense': 'sum',
                    'Budget': 'sum',
                    'FTE': 'sum'
                }).reset_index()
                analysis['monthly_trend'] = monthly_dept

                # Department efficiency analysis
                dept_efficiency = dept_data.groupby('Department').agg({
                    'Expense': 'sum',
                    'FTE': 'sum'
                }).reset_index()
                dept_efficiency['Cost_per_FTE'] = dept_efficiency['Expense'] / dept_efficiency['FTE']
                analysis['efficiency'] = dept_efficiency

            if fbmc_data is not None and not fbmc_data.empty:
                # SG&A from FBMC
                sga_data = fbmc_data[fbmc_data['Cost_Category_L1'] == 'SGA']
                if not sga_data.empty:
                    sga_summary = sga_data.groupby(['Cost_Category_L2', 'Cost_Category_L3']).agg({
                        'Amount': 'sum',
                        'Budget': 'sum',
                        'Prior_Year': 'sum'
                    }).reset_index()

                    sga_summary['Budget_Variance'] = sga_summary['Amount'] - sga_summary['Budget']
                    sga_summary['YoY_Growth'] = ((sga_summary['Amount'] - sga_summary['Prior_Year']) / 
                                               sga_summary['Prior_Year'] * 100).round(2)

                    analysis['sga_summary'] = sga_summary

                    # SG&A monthly trend
                    sga_monthly = sga_data.groupby(['Date', 'Cost_Category_L2']).agg({
                        'Amount': 'sum'
                    }).reset_index()
                    analysis['sga_monthly'] = sga_monthly

        except Exception as e:
            st.error(f"Error in SG&A analysis: {str(e)}")

        return analysis

    def generate_ai_insights(self, data_summary):
        """Generate AI insights with fallback"""
        if not self.ai_insights_available:
            return self.generate_rule_based_insights(data_summary)

        try:
            from transformers import pipeline

            # Use a lightweight model for text generation
            generator = pipeline('text-generation', model='gpt2', max_length=150, num_return_sequences=1)

            prompt = f"FMCG Financial Analysis: Revenue growth {data_summary.get('revenue_growth', 0):.1f}%, cost ratio {data_summary.get('cost_ratio', 0):.1f}%, margin {data_summary.get('gross_margin', 0):.1f}%. Strategic recommendations:"

            result = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)
            return result[0]['generated_text']

        except Exception as e:
            st.warning(f"AI insights unavailable: {str(e)}. Using rule-based analysis.")
            return self.generate_rule_based_insights(data_summary)

    def generate_rule_based_insights(self, data_summary):
        """Generate insights using business rules"""
        insights = []

        # Revenue insights
        revenue_growth = data_summary.get('revenue_growth', 0)
        if revenue_growth > 15:
            insights.append("🚀 Exceptional revenue growth indicates strong market performance and effective strategy execution")
        elif revenue_growth > 5:
            insights.append("📈 Positive revenue growth shows healthy business expansion in competitive FMCG market")
        elif revenue_growth > -5:
            insights.append("📊 Revenue performance is stable within normal FMCG market fluctuations")
        else:
            insights.append("⚠️ Revenue decline requires immediate strategic review and market repositioning")

        # Cost insights
        cost_ratio = data_summary.get('cost_ratio', 0)
        if cost_ratio > 0.85:
            insights.append("💰 High cost ratio indicates urgent need for supply chain and operational efficiency improvements")
        elif cost_ratio > 0.75:
            insights.append("⚖️ Cost structure requires optimization to maintain competitive pricing in FMCG sector")
        elif cost_ratio < 0.6:
            insights.append("✅ Excellent cost management provides strong competitive advantage and pricing flexibility")
        else:
            insights.append("👍 Healthy cost structure supports sustainable growth and market expansion")

        # Margin insights
        gross_margin = data_summary.get('gross_margin', 0)
        if gross_margin > 40:
            insights.append("💎 Strong gross margins indicate premium positioning and operational excellence")
        elif gross_margin > 25:
            insights.append("💼 Healthy margins support sustainable operations and reinvestment capacity")
        elif gross_margin > 15:
            insights.append("⚡ Margins under pressure - focus on value optimization and cost management needed")
        else:
            insights.append("🔴 Critical margin situation requires immediate cost and pricing strategy review")

        # Channel insights
        channel_performance = data_summary.get('top_channel_share', 0)
        if channel_performance > 40:
            insights.append("🏪 Strong channel concentration provides economies of scale but increases dependency risk")

        # Variance insights
        budget_variance = abs(data_summary.get('budget_variance', 0))
        if budget_variance > 20:
            insights.append("📊 Significant budget variance suggests need for improved forecasting accuracy and planning")
        elif budget_variance > 10:
            insights.append("📋 Moderate budget variance indicates room for planning process improvements")
        else:
            insights.append("🎯 Excellent budget control demonstrates strong financial discipline and planning capability")

        return " | ".join(insights)

    def create_revenue_analysis(self, mastersale_data):
        """Create comprehensive revenue analysis"""
        if mastersale_data is None or mastersale_data.empty:
            return None, 0

        try:
            # Time series analysis
            monthly_revenue = mastersale_data.groupby('Date').agg({
                'Revenue': 'sum',
                'Volume': 'sum',
                'Units_Sold': 'sum'
            }).reset_index()
            monthly_revenue['Month'] = monthly_revenue['Date'].dt.strftime('%Y-%m')
            monthly_revenue = monthly_revenue.sort_values('Date')
            monthly_revenue['ASP'] = monthly_revenue['Revenue'] / monthly_revenue['Volume']

            # Channel analysis
            channel_revenue = mastersale_data.groupby('Channel').agg({
                'Revenue': 'sum',
                'Volume': 'sum',
                'Units_Sold': 'sum'
            }).reset_index()
            channel_revenue = channel_revenue.sort_values('Revenue', ascending=False)
            channel_revenue['ASP'] = channel_revenue['Revenue'] / channel_revenue['Volume']

            # Product analysis
            product_revenue = mastersale_data.groupby('Product').agg({
                'Revenue': 'sum',
                'Volume': 'sum',
                'Units_Sold': 'sum'
            }).reset_index()
            product_revenue = product_revenue.sort_values('Revenue', ascending=False)
            product_revenue['ASP'] = product_revenue['Revenue'] / product_revenue['Volume']

            # Regional analysis
            region_revenue = mastersale_data.groupby('Region').agg({
                'Revenue': 'sum',
                'Volume': 'sum'
            }).reset_index()
            region_revenue = region_revenue.sort_values('Revenue', ascending=False)
            region_revenue['ASP'] = region_revenue['Revenue'] / region_revenue['Volume']

            # Growth analysis
            if len(monthly_revenue) > 1:
                monthly_revenue['Revenue_Growth'] = monthly_revenue['Revenue'].pct_change() * 100
                monthly_revenue['Volume_Growth'] = monthly_revenue['Volume'].pct_change() * 100
                monthly_revenue['ASP_Growth'] = monthly_revenue['ASP'].pct_change() * 100

            # Channel-Product matrix
            channel_product = mastersale_data.groupby(['Channel', 'Product'])['Revenue'].sum().reset_index()
            channel_product_pivot = channel_product.pivot(index='Channel', columns='Product', values='Revenue').fillna(0)

            return {
                'monthly': monthly_revenue,
                'channel': channel_revenue,
                'product': product_revenue,
                'region': region_revenue,
                'channel_product_matrix': channel_product_pivot
            }, mastersale_data['Revenue'].sum()

        except Exception as e:
            st.error(f"Error in revenue analysis: {str(e)}")
            return None, 0

    def create_cost_analysis(self, fbmc_data):
        """Create comprehensive cost analysis"""
        if fbmc_data is None or fbmc_data.empty:
            return None, 0

        try:
            # Cost by category
            cost_by_category = fbmc_data.groupby('Cost_Category_L1').agg({
                'Amount': 'sum',
                'Budget': 'sum',
                'Prior_Year': 'sum'
            }).reset_index()
            cost_by_category = cost_by_category.sort_values('Amount', ascending=False)
            cost_by_category['Budget_Variance'] = cost_by_category['Amount'] - cost_by_category['Budget']
            cost_by_category['YoY_Growth'] = ((cost_by_category['Amount'] - cost_by_category['Prior_Year']) / 
                                            cost_by_category['Prior_Year'] * 100).round(2)

            # Monthly cost trend
            monthly_cost = fbmc_data.groupby('Date').agg({
                'Amount': 'sum',
                'Budget': 'sum'
            }).reset_index()
            monthly_cost['Month'] = monthly_cost['Date'].dt.strftime('%Y-%m')
            monthly_cost = monthly_cost.sort_values('Date')
            monthly_cost['Budget_Variance'] = monthly_cost['Amount'] - monthly_cost['Budget']

            # Detailed cost breakdown
            detailed_cost = fbmc_data.groupby(['Cost_Category_L1', 'Cost_Category_L2']).agg({
                'Amount': 'sum',
                'Budget': 'sum'
            }).reset_index()
            detailed_cost = detailed_cost.sort_values('Amount', ascending=False)
            detailed_cost['Budget_Variance'] = detailed_cost['Amount'] - detailed_cost['Budget']

            # Cost hierarchy analysis (all 5 levels)
            hierarchy_cost = fbmc_data.groupby([
                'Cost_Category_L1', 'Cost_Category_L2', 'Cost_Category_L3', 
                'Cost_Category_L4', 'Cost_Category_L5'
            ])['Amount'].sum().reset_index()

            # Growth analysis
            if len(monthly_cost) > 1:
                monthly_cost['Cost_Growth'] = monthly_cost['Amount'].pct_change() * 100
                monthly_cost['Budget_Variance_Pct'] = (monthly_cost['Budget_Variance'] / 
                                                     monthly_cost['Budget'] * 100).round(2)

            # Cost efficiency metrics
            cost_efficiency = fbmc_data.groupby(['Cost_Category_L1', 'Cost_Category_L2']).agg({
                'Amount': ['sum', 'mean', 'std']
            }).round(2)

            return {
                'by_category': cost_by_category,
                'monthly': monthly_cost,
                'detailed': detailed_cost,
                'hierarchy': hierarchy_cost,
                'efficiency': cost_efficiency
            }, fbmc_data['Amount'].sum()

        except Exception as e:
            st.error(f"Error in cost analysis: {str(e)}")
            return None, 0


def create_dashboard_tabs(dashboard):
    """Create main dashboard tabs"""

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary", 
        "💰 Revenue Analytics", 
        "💸 Cost Management", 
        "🏢 SG&A Analysis", 
        "📈 Performance KPIs",
        "🤖 AI Insights & Export"
    ])

    with tab1:
        create_executive_summary_tab(dashboard)

    with tab2:
        create_revenue_tab(dashboard)

    with tab3:
        create_cost_tab(dashboard)

    with tab4:
        create_sga_tab(dashboard)

    with tab5:
        create_kpi_tab(dashboard)

    with tab6:
        create_insights_export_tab(dashboard)

def create_executive_summary_tab(dashboard):
    """Create executive summary tab content"""
    st.subheader("📋 Executive Dashboard")

    if not dashboard.data:
        st.info("Please upload data or enable demo data to view the executive summary.")
        return

    # Calculate key metrics
    total_revenue = 0
    total_cost = 0
    total_volume = 0

    if 'Mastersale' in dashboard.data:
        total_revenue = dashboard.data['Mastersale']['Revenue'].sum()
        total_volume = dashboard.data['Mastersale']['Volume'].sum()

    if 'FBMC' in dashboard.data:
        total_cost = dashboard.data['FBMC']['Amount'].sum()

    gross_profit = total_revenue - total_cost
    gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0

    # Key metrics display with enhanced styling
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #1f4e79; margin: 0;">💰 Total Revenue</h3>
            <h2 style="color: #2e8b57; margin: 5px 0;">${total_revenue:,.0f}</h2>
            <p style="margin: 0; color: #666;">Fiscal Year Performance</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #1f4e79; margin: 0;">💸 Total Cost</h3>
            <h2 style="color: #dc3545; margin: 5px 0;">${total_cost:,.0f}</h2>
            <p style="margin: 0; color: #666;">All Cost Categories</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #1f4e79; margin: 0;">📈 Gross Profit</h3>
            <h2 style="color: #28a745; margin: 5px 0;">${gross_profit:,.0f}</h2>
            <p style="margin: 0; color: #666;">Revenue - Total Costs</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        margin_color = "#28a745" if gross_margin > 20 else "#ffc107" if gross_margin > 10 else "#dc3545"
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #1f4e79; margin: 0;">📊 Gross Margin</h3>
            <h2 style="color: {margin_color}; margin: 5px 0;">{gross_margin:.1f}%</h2>
            <p style="margin: 0; color: #666;">Profitability Ratio</p>
        </div>
        """, unsafe_allow_html=True)

    # Data overview section
    st.subheader("📊 Data Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #1f4e79; margin: 0;">📋 Sheets Loaded</h4>
            <h3 style="color: #17a2b8; margin: 5px 0;">{len(dashboard.data)}</h3>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                {", ".join(dashboard.data.keys())}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_records = sum(len(df) for df in dashboard.data.values())
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #1f4e79; margin: 0;">📈 Total Records</h4>
            <h3 style="color: #17a2b8; margin: 5px 0;">{total_records:,}</h3>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                Across all data sheets
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        date_range = "N/A"
        if 'Mastersale' in dashboard.data and 'Date' in dashboard.data['Mastersale'].columns:
            min_date = dashboard.data['Mastersale']['Date'].min()
            max_date = dashboard.data['Mastersale']['Date'].max()
            date_range = f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"

        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #1f4e79; margin: 0;">📅 Date Range</h4>
            <h3 style="color: #17a2b8; margin: 5px 0;">{date_range}</h3>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                Analysis Period
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Quick insights section
    st.subheader("⚡ Quick Insights")

    if 'Mastersale' in dashboard.data and 'FBMC' in dashboard.data:
        # Calculate some quick insights
        revenue_data = dashboard.data['Mastersale']
        cost_data = dashboard.data['FBMC']

        # Top performing channel
        top_channel = revenue_data.groupby('Channel')['Revenue'].sum().idxmax()
        top_channel_revenue = revenue_data.groupby('Channel')['Revenue'].sum().max()

        # Highest cost category
        top_cost_category = cost_data.groupby('Cost_Category_L1')['Amount'].sum().idxmax()
        top_cost_amount = cost_data.groupby('Cost_Category_L1')['Amount'].sum().max()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏆 Top Performing Channel</h4>
                <p><strong>{top_channel}</strong> generates <strong>${top_channel_revenue:,.0f}</strong> in revenue</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <h4>💰 Largest Cost Category</h4>
                <p><strong>{top_cost_category}</strong> accounts for <strong>${top_cost_amount:,.0f}</strong> in costs</p>
            </div>
            """, unsafe_allow_html=True)

def create_revenue_tab(dashboard):
    """Create revenue analysis tab"""
    st.subheader("💰 Revenue Analytics")

    if 'Mastersale' not in dashboard.data:
        st.warning("Mastersale data not available. Please upload data with Mastersale sheet.")
        return

    revenue_analysis, total_revenue = dashboard.create_revenue_analysis(dashboard.data['Mastersale'])

    if not revenue_analysis:
        st.error("Unable to process revenue data.")
        return

    # Revenue trend analysis
    st.subheader("📈 Revenue Trends")
    col1, col2 = st.columns(2)

    with col1:
        fig_monthly = px.line(
            revenue_analysis['monthly'], 
            x='Month', 
            y='Revenue',
            title='Monthly Revenue Trend',
            markers=True,
            color_discrete_sequence=['#1f4e79']
        )
        fig_monthly.update_layout(
            height=400,
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_monthly.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig_monthly.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        fig_channel = px.pie(
            revenue_analysis['channel'],
            values='Revenue',
            names='Channel',
            title='Revenue Distribution by Channel',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_channel.update_layout(height=400)
        st.plotly_chart(fig_channel, use_container_width=True)

    # Product and Region analysis
    st.subheader("🎯 Performance by Segment")
    col3, col4 = st.columns(2)

    with col3:
        top_products = revenue_analysis['product'].head(10)
        fig_product = px.bar(
            top_products,
            x='Revenue',
            y='Product',
            orientation='h',
            title='Top 10 Products by Revenue',
            color='Revenue',
            color_continuous_scale='Blues'
        )
        fig_product.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_product, use_container_width=True)

    with col4:
        fig_region = px.bar(
            revenue_analysis['region'],
            x='Region',
            y='Revenue',
            title='Revenue by Region',
            color='Revenue',
            color_continuous_scale='Greens'
        )
        fig_region.update_layout(height=500)
        st.plotly_chart(fig_region, use_container_width=True)

    # Revenue growth analysis
    if 'Revenue_Growth' in revenue_analysis['monthly'].columns:
        st.subheader("📊 Growth Analysis")

        growth_data = revenue_analysis['monthly'].dropna(subset=['Revenue_Growth'])
        if not growth_data.empty:
            fig_growth = px.bar(
                growth_data,
                x='Month',
                y='Revenue_Growth',
                title='Month-over-Month Revenue Growth (%)',
                color='Revenue_Growth',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0
            )
            fig_growth.update_layout(
                height=400,
                xaxis_title="Month",
                yaxis_title="Growth Rate (%)"
            )
            fig_growth.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            st.plotly_chart(fig_growth, use_container_width=True)

    # Channel-Product matrix
    if 'channel_product_matrix' in revenue_analysis:
        st.subheader("🔍 Channel-Product Performance Matrix")

        matrix_data = revenue_analysis['channel_product_matrix']
        if not matrix_data.empty:
            fig_heatmap = px.imshow(
                matrix_data.values,
                x=matrix_data.columns,
                y=matrix_data.index,
                title='Revenue Heatmap: Channel vs Product',
                aspect='auto',
                color_continuous_scale='Blues'
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)

def create_cost_tab(dashboard):
    """Create cost analysis tab"""
    st.subheader("💸 Cost Management Analysis")

    if 'FBMC' not in dashboard.data:
        st.warning("FBMC data not available. Please upload data with FBMC sheet.")
        return

    # Process cost hierarchy
    fbmc_processed = dashboard.create_cost_hierarchy_mapping(dashboard.data['FBMC'])
    cost_analysis, total_cost = dashboard.create_cost_analysis(fbmc_processed)

    if not cost_analysis:
        st.error("Unable to process cost data.")
        return

    # Cost distribution overview
    st.subheader("📊 Cost Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_cost_cat = px.pie(
            cost_analysis['by_category'],
            values='Amount',
            names='Cost_Category_L1',
            title='Cost Distribution by Category',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_cost_cat.update_layout(height=400)
        st.plotly_chart(fig_cost_cat, use_container_width=True)

    with col2:
        fig_cost_monthly = px.line(
            cost_analysis['monthly'],
            x='Month',
            y='Amount',
            title='Monthly Cost Trend',
            markers=True,
            color_discrete_sequence=['#dc3545']
        )
        fig_cost_monthly.update_layout(
            height=400,
            xaxis_title="Month",
            yaxis_title="Cost ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_cost_monthly, use_container_width=True)

    # Budget variance analysis
    st.subheader("📋 Budget Performance")

    if 'Budget_Variance' in cost_analysis['by_category'].columns:
        col3, col4 = st.columns(2)

        with col3:
            fig_variance = px.bar(
                cost_analysis['by_category'],
                x='Cost_Category_L1',
                y='Budget_Variance',
                title='Budget Variance by Category',
                color='Budget_Variance',
                color_continuous_scale='RdYlGn_r',
                color_continuous_midpoint=0
            )
            fig_variance.update_layout(height=400)
            fig_variance.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            st.plotly_chart(fig_variance, use_container_width=True)

        with col4:
            if 'YoY_Growth' in cost_analysis['by_category'].columns:
                fig_yoy = px.bar(
                    cost_analysis['by_category'],
                    x='Cost_Category_L1',
                    y='YoY_Growth',
                    title='Year-over-Year Cost Growth (%)',
                    color='YoY_Growth',
                    color_continuous_scale='RdYlBu_r',
                    color_continuous_midpoint=0
                )
                fig_yoy.update_layout(height=400)
                fig_yoy.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                st.plotly_chart(fig_yoy, use_container_width=True)

    # Detailed cost hierarchy
    st.subheader("🔍 5-Level Cost Hierarchy")

    if len(cost_analysis['hierarchy']) > 0:
        # Create treemap for hierarchy visualization
        hierarchy_sample = cost_analysis['hierarchy'].head(50)  # Limit for performance

        fig_treemap = px.treemap(
            hierarchy_sample,
            path=['Cost_Category_L1', 'Cost_Category_L2', 'Cost_Category_L3'],
            values='Amount',
            title='Cost Hierarchy Breakdown (Top 50 Items)',
            color='Amount',
            color_continuous_scale='Reds'
        )
        fig_treemap.update_layout(height=600)
        st.plotly_chart(fig_treemap, use_container_width=True)

        # Detailed breakdown table
        st.subheader("📋 Detailed Cost Breakdown")

        # Create expandable sections for each L1 category
        for category in cost_analysis['by_category']['Cost_Category_L1'].unique():
            with st.expander(f"📂 {category} - ${cost_analysis['by_category'][cost_analysis['by_category']['Cost_Category_L1']==category]['Amount'].iloc[0]:,.0f}"):
                category_detail = cost_analysis['detailed'][
                    cost_analysis['detailed']['Cost_Category_L1'] == category
                ].head(20)

                if not category_detail.empty:
                    st.dataframe(
                        category_detail[['Cost_Category_L2', 'Amount', 'Budget_Variance']].round(0),
                        use_container_width=True
                    )

def create_sga_tab(dashboard):
    """Create SG&A analysis tab"""
    st.subheader("🏢 SG&A Analysis by Department")

    sga_analysis = dashboard.analyze_sga_by_department(
        dashboard.data.get('Dept'),
        dashboard.data.get('FBMC')
    )

    if not sga_analysis:
        st.warning("SG&A data not available. Please upload data with Dept and/or FBMC sheets.")
        return

    # Department expense analysis
    if 'department_summary' in sga_analysis:
        dept_summary = sga_analysis['department_summary']

        st.subheader("💼 Department Performance")
        col1, col2 = st.columns(2)

        with col1:
            dept_totals = dept_summary.groupby('Department')['Expense'].sum().reset_index()
            dept_totals = dept_totals.sort_values('Expense', ascending=False)

            fig_dept = px.bar(
                dept_totals,
                x='Department',
                y='Expense',
                title='Total Expenses by Department',
                color='Expense',
                color_continuous_scale='Blues'
            )
            fig_dept.update_layout(
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_dept, use_container_width=True)

        with col2:
            dept_variance = dept_summary.groupby('Department')['Variance_Pct'].mean().reset_index()
            dept_variance = dept_variance.sort_values('Variance_Pct')

            fig_variance = px.bar(
                dept_variance,
                x='Department',
                y='Variance_Pct',
                title='Average Budget Variance % by Department',
                color='Variance_Pct',
                color_continuous_scale='RdYlGn_r',
                color_continuous_midpoint=0
            )
            fig_variance.update_layout(
                height=400,
                xaxis_tickangle=-45
            )
            fig_variance.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            st.plotly_chart(fig_variance, use_container_width=True)

        # Activity breakdown heatmap
        st.subheader("🔥 Department Activity Heatmap")

        activity_pivot = dept_summary.pivot_table(
            values='Expense',
            index='Department',
            columns='Activity',
            aggfunc='sum',
            fill_value=0
        )

        if not activity_pivot.empty:
            fig_heatmap = px.imshow(
                activity_pivot.values,
                x=activity_pivot.columns,
                y=activity_pivot.index,
                title='Expense Distribution: Department vs Activity',
                aspect='auto',
                color_continuous_scale='Viridis'
            )
            fig_heatmap.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Efficiency analysis
        if 'efficiency' in sga_analysis:
            st.subheader("⚡ Department Efficiency")

            efficiency_data = sga_analysis['efficiency'].sort_values('Cost_per_FTE', ascending=False)

            fig_efficiency = px.scatter(
                efficiency_data,
                x='FTE',
                y='Cost_per_FTE',
                size='Expense',
                color='Department',
                title='Department Efficiency: Cost per FTE vs Total FTE',
                hover_data=['Expense']
            )
            fig_efficiency.update_layout(height=500)
            st.plotly_chart(fig_efficiency, use_container_width=True)

    # SG&A from FBMC analysis
    if 'sga_summary' in sga_analysis:
        st.subheader("📊 SG&A Cost Structure (from FBMC)")

        sga_summary = sga_analysis['sga_summary']

        col3, col4 = st.columns(2)

        with col3:
            sga_l2 = sga_summary.groupby('Cost_Category_L2')['Amount'].sum().reset_index()
            sga_l2 = sga_l2.sort_values('Amount', ascending=False)

            fig_sga_l2 = px.bar(
                sga_l2,
                x='Amount',
                y='Cost_Category_L2',
                orientation='h',
                title='SG&A by Sub-Category',
                color='Amount',
                color_continuous_scale='Oranges'
            )
            fig_sga_l2.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_sga_l2, use_container_width=True)

        with col4:
            if 'YoY_Growth' in sga_summary.columns:
                sga_growth = sga_summary.groupby('Cost_Category_L2')['YoY_Growth'].mean().reset_index()

                fig_sga_growth = px.bar(
                    sga_growth,
                    x='Cost_Category_L2',
                    y='YoY_Growth',
                    title='SG&A Year-over-Year Growth (%)',
                    color='YoY_Growth',
                    color_continuous_scale='RdYlGn_r',
                    color_continuous_midpoint=0
                )
                fig_sga_growth.update_layout(
                    height=400,
                    xaxis_tickangle=-45
                )
                fig_sga_growth.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                st.plotly_chart(fig_sga_growth, use_container_width=True)


    use_demo_data = st.sidebar.checkbox(
        "🎯 Use Demo Data", 
        value=True, 
        help="Generate comprehensive sample data for demonstration"
    )

    st.sidebar.markdown("---")

    # Analysis options
    st.sidebar.subheader("⚙️ Analysis Options")

    show_advanced_analytics = st.sidebar.checkbox(
        "📊 Advanced Analytics", 
        value=True,
        help="Enable advanced KPI calculations and benchmarking"
    )

    enable_ai_insights = st.sidebar.checkbox(
        "🤖 AI Insights", 
        value=dashboard.ai_insights_available,
        help="Generate AI-powered business insights"
    )

    st.sidebar.markdown("---")

    # Process data
    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            dashboard.data = dashboard.process_excel_file(uploaded_file)

        if dashboard.data:
            st.sidebar.success("✅ File processed successfully!")
            st.sidebar.write(f"**Sheets found:** {', '.join(dashboard.data.keys())}")

            # Show data summary in sidebar
            for sheet_name, df in dashboard.data.items():
                st.sidebar.write(f"📋 {sheet_name}: {len(df):,} records")
        else:
            st.sidebar.error("❌ Error processing file")

    elif use_demo_data:
        with st.spinner("Generating comprehensive demo data..."):
            dashboard.data = dashboard.generate_sample_data()
        st.sidebar.success("✅ Demo data loaded!")
        st.sidebar.write("**Sheets:** Mastersale, FBMC, Dept")
        st.sidebar.write("📋 Mastersale: 3,000 records")
        st.sidebar.write("📋 FBMC: 4,320 records") 
        st.sidebar.write("📋 Dept: 1,200 records")

    # Main dashboard content
    if dashboard.data:
        # Create tabbed interface
        create_dashboard_tabs(dashboard)

        # Footer information
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🏢 <strong>FP&A Dashboard - FMCG Analytics</strong></p>
            <p>Professional Financial Planning & Analysis • Multi-dimensional Revenue Analysis • 5-Level Cost Hierarchy • AI-Powered Insights</p>
            <p><em>Built with Streamlit • Powered by Advanced Analytics</em></p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        <div class="success-box">
        <h2>🎯 Welcome to FP&A Dashboard</h2>
        <p style="font-size: 1.2em;">This comprehensive Financial Planning & Analysis dashboard provides enterprise-grade analytics for FMCG businesses.</p>

        <h3>🚀 Key Features</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div>
                <h4>📊 Revenue Analytics</h4>
                <ul>
                    <li>Multi-dimensional revenue analysis</li>
                    <li>Channel and product performance</li>
                    <li>Regional sales breakdown</li>
                    <li>Growth trend analysis</li>
                </ul>
            </div>
            <div>
                <h4>💰 Cost Management</h4>
                <ul>
                    <li>5-level cost hierarchy mapping</li>
                    <li>Budget variance analysis</li>
                    <li>Year-over-year comparisons</li>
                    <li>Cost efficiency metrics</li>
                </ul>
            </div>
            <div>
                <h4>🏢 SG&A Analysis</h4>
                <ul>
                    <li>Department-wise expense tracking</li>
                    <li>Activity-based costing</li>
                    <li>FTE efficiency analysis</li>
                    <li>Budget performance monitoring</li>
                </ul>
            </div>
            <div>
                <h4>🤖 AI Insights</h4>
                <ul>
                    <li>Intelligent business insights</li>
                    <li>Performance benchmarking</li>
                    <li>Strategic recommendations</li>
                    <li>Automated reporting</li>
                </ul>
            </div>
        </div>

        <h3>📋 Getting Started</h3>
        <ol style="font-size: 1.1em;">
            <li><strong>Upload Data:</strong> Use the file uploader in the sidebar to upload your Excel file with sheets: Mastersale, FBMC, Dept</li>
            <li><strong>Or Try Demo:</strong> Check "Use Demo Data" to explore with comprehensive sample data</li>
            <li><strong>Explore Analytics:</strong> Navigate through tabs to analyze revenue, costs, SG&A, and KPIs</li>
            <li><strong>Generate Insights:</strong> Use AI-powered insights and export capabilities for reporting</li>
        </ol>

        <h3>📊 Expected Data Structure</h3>
        </div>
        """, unsafe_allow_html=True)

        # Data structure guide
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>📈 Mastersale Sheet</h4>
                <ul>
                    <li><strong>Date:</strong> Transaction date</li>
                    <li><strong>Product:</strong> Product name/SKU</li>
                    <li><strong>Channel:</strong> Sales channel</li>
                    <li><strong>Region:</strong> Geographic region</li>
                    <li><strong>Revenue:</strong> Sales revenue</li>
                    <li><strong>Volume:</strong> Units sold</li>
                    <li><strong>ASP:</strong> Average selling price</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>💸 FBMC Sheet</h4>
                <ul>
                    <li><strong>Date:</strong> Cost period</li>
                    <li><strong>Cost_Category_L1-L5:</strong> 5-level hierarchy</li>
                    <li><strong>Amount:</strong> Actual cost</li>
                    <li><strong>Budget:</strong> Budgeted amount</li>
                    <li><strong>Prior_Year:</strong> Previous year comparison</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="insight-box">
                <h4>🏢 Dept Sheet</h4>
                <ul>
                    <li><strong>Date:</strong> Expense period</li>
                    <li><strong>Department:</strong> Department name</li>
                    <li><strong>Activity:</strong> Activity type</li>
                    <li><strong>Expense:</strong> Actual expense</li>
                    <li><strong>Budget:</strong> Budgeted amount</li>
                    <li><strong>Variance:</strong> Budget variance</li>
                    <li><strong>FTE:</strong> Full-time employees</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Technical specifications
        st.markdown("""
        <div class="warning-box">
        <h3>⚙️ Technical Specifications</h3>
        <ul>
            <li><strong>🔄 Fiscal Year Support:</strong> Intelligent parsing of FY2425 format (2024-2025)</li>
            <li><strong>📊 Multi-Sheet Processing:</strong> Automatic detection and processing of Excel sheets</li>
            <li><strong>🎯 5-Level Cost Hierarchy:</strong> Complete cost categorization and mapping</li>
            <li><strong>🤖 AI Integration:</strong> Optional Hugging Face transformers with rule-based fallback</li>
            <li><strong>📤 Export Capabilities:</strong> JSON, CSV, and visualization export options</li>
            <li><strong>🔒 Enterprise Security:</strong> Local processing, no external data transmission</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
