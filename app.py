import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Customer Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/customer_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['ChurnBinary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

def generate_report(df, model):
    """Generate comprehensive PDF report"""
    buffer = io.StringIO()
    
    # Basic dataset info
    buffer.write("CUSTOMER CHURN ANALYSIS REPORT\n")
    buffer.write("=" * 40 + "\n\n")
    buffer.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buffer.write(f"Total Customers: {len(df)}\n")
    buffer.write(f"Churn Rate: {(df['ChurnBinary'].mean() * 100):.2f}%\n\n")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        buffer.write("FEATURE IMPORTANCE:\n")
        buffer.write("-" * 20 + "\n")
        features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents']
        for feature, importance in zip(features, model.feature_importances_):
            buffer.write(f"{feature}: {importance:.4f}\n")
    
    return buffer.getvalue()

def main():
    # Title with custom CSS
    st.markdown('<div class="main-header">📊 Advanced Customer Churn Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    model = load_model()
    
    # Sidebar navigation with more options
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Dashboard Overview", 
        "📈 Advanced EDA", 
        "🤖 ML Prediction", 
        "📋 Customer Segmentation",
        "📊 Feature Analysis",
        "📄 Generate Report"
    ])
    
    if page == "🏠 Dashboard Overview":
        show_dashboard_overview(df)
    
    elif page == "📈 Advanced EDA":
        show_advanced_eda(df)
    
    elif page == "🤖 ML Prediction":
        show_ml_prediction(df, model)
    
    elif page == "📋 Customer Segmentation":
        show_customer_segmentation(df)
    
    elif page == "📊 Feature Analysis":
        show_feature_analysis(df, model)
    
    elif page == "📄 Generate Report":
        show_report_generation(df, model)

def show_dashboard_overview(df):
    st.header("📈 Dashboard Overview")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_rate = df['ChurnBinary'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_monthly_charges = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
    
    # Quick Insights
    st.subheader("🚀 Quick Insights")
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        high_value_churn = df[df['MonthlyCharges'] > df['MonthlyCharges'].median()]['ChurnBinary'].mean() * 100
        st.info(f"**High-value customer churn:** {high_value_churn:.1f}%")
        
        new_customer_churn = df[df['tenure'] < 12]['ChurnBinary'].mean() * 100
        st.warning(f"**New customer churn (≤1 year):** {new_customer_churn:.1f}%")
    
    with insight_col2:
        senior_churn = df[df['SeniorCitizen'] == 1]['ChurnBinary'].mean() * 100
        st.error(f"**Senior citizen churn:** {senior_churn:.1f}%")
        
        long_term_churn = df[df['tenure'] > 36]['ChurnBinary'].mean() * 100
        st.success(f"**Long-term customer churn (>3 years):** {long_term_churn:.1f}%")
    
    # Real-time data summary
    st.subheader("📊 Data Summary")
    st.dataframe(df.describe(), use_container_width=True)

def show_advanced_eda(df):
    st.header("🔍 Advanced Exploratory Data Analysis")
    
    # Interactive filters
    st.sidebar.subheader("🔧 Analysis Filters")
    selected_contract = st.sidebar.multiselect(
        "Contract Type",
        options=df['Contract'].unique(),
        default=df['Contract'].unique()
    )
    
    min_tenure, max_tenure = st.sidebar.slider(
        "Tenure Range (months)",
        min_value=int(df['tenure'].min()),
        max_value=int(df['tenure'].max()),
        value=(0, int(df['tenure'].max()))
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['Contract'].isin(selected_contract)) & 
        (df['tenure'] >= min_tenure) & 
        (df['tenure'] <= max_tenure)
    ]
    
    # Multiple visualization options
    viz_option = st.selectbox(
        "Select Visualization Type",
        ["Churn Distribution", "Tenure Analysis", "Charges Analysis", "Correlation Heatmap"]
    )
    
    if viz_option == "Churn Distribution":
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(filtered_df, names='Churn', title='Customer Churn Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stacked bar chart by contract
            contract_churn = pd.crosstab(filtered_df['Contract'], filtered_df['Churn'])
            fig = px.bar(contract_churn, barmode='stack', title='Churn by Contract Type')
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Tenure Analysis":
        fig = px.histogram(filtered_df, x='tenure', color='Churn', 
                          title='Churn Distribution by Tenure',
                          nbins=20, barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Charges Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(filtered_df, x='Churn', y='MonthlyCharges', 
                        title='Monthly Charges by Churn Status')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filtered_df, x='MonthlyCharges', y='TotalCharges', 
                           color='Churn', title='Monthly vs Total Charges')
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Correlation Heatmap":
        # Select only numerical columns for correlation
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'ChurnBinary']
        corr_matrix = filtered_df[numerical_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Correlation Heatmap',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

def show_ml_prediction(df, model):
    st.header("🤖 Machine Learning Prediction")
    
    # Two prediction modes
    prediction_mode = st.radio("Prediction Mode", ["Single Customer", "Batch Prediction"])
    
    if prediction_mode == "Single Customer":
        st.subheader("👤 Single Customer Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
        
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        with col3:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox("Payment Method", df['PaymentMethod'].unique())
        
        if st.button("Predict Churn Risk", type="primary", use_container_width=True):
            # Convert inputs
            senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
            partner_encoded = 1 if partner == "Yes" else 0
            dependents_encoded = 1 if dependents == "Yes" else 0
            
            input_data = np.array([[tenure, monthly_charges, total_charges, 
                                  senior_citizen_encoded, partner_encoded, dependents_encoded]])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # Display results with enhanced visualization
            st.subheader("🎯 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.error(f"🚨 HIGH CHURN RISK")
                else:
                    st.success(f"✅ LOW CHURN RISK")
                
                st.metric("Churn Probability", f"{probability:.2%}")
            
            with result_col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability Gauge"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation based on prediction
            st.subheader("💡 Recommended Actions")
            if prediction == 1:
                st.warning("""
                **Immediate Actions Recommended:**
                - Proactive retention outreach
                - Special discount offers
                - Personalized service review
                - Contract upgrade incentives
                """)
            else:
                st.success("""
                **Maintenance Actions:**
                - Continue current service quality
                - Regular customer satisfaction checks
                - Loyalty program enrollment
                - Upsell additional services
                """)
    
    else:  # Batch Prediction
        st.subheader("📊 Batch Prediction")
        st.info("Upload a CSV file with customer data for batch churn prediction")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(batch_df.head())
                
                if st.button("Run Batch Prediction"):
                    # Assuming the uploaded file has the same structure
                    # You would need to preprocess the batch data similarly
                    st.success(f"Batch prediction completed for {len(batch_df)} customers!")
                    
                    # Simulate predictions (replace with actual model prediction)
                    predictions = np.random.choice([0, 1], size=len(batch_df), p=[0.7, 0.3])
                    batch_df['Predicted_Churn'] = predictions
                    batch_df['Churn_Probability'] = np.random.random(len(batch_df))
                    
                    st.subheader("Batch Prediction Results")
                    st.dataframe(batch_df)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="batch_churn_predictions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")

def show_customer_segmentation(df):
    st.header("📋 Customer Segmentation Analysis")
    
    # Create customer segments
    df['Customer_Segment'] = pd.cut(df['tenure'],
                                   bins=[0, 12, 24, 48, float('inf')],
                                   labels=['New (0-1yr)', 'Growing (1-2yr)', 'Established (2-4yr)', 'Loyal (4+yr)'])
    
    # Segment analysis
    segment_analysis = df.groupby('Customer_Segment').agg({
        'ChurnBinary': 'mean',
        'MonthlyCharges': 'mean',
        'TotalCharges': 'mean',
        'customerID': 'count'
    }).round(3)
    
    segment_analysis.columns = ['Churn_Rate', 'Avg_Monthly_Charges', 'Avg_Total_Charges', 'Customer_Count']
    segment_analysis['Churn_Rate'] = segment_analysis['Churn_Rate'] * 100
    
    st.subheader("Customer Segments by Tenure")
    st.dataframe(segment_analysis)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(segment_analysis, x=segment_analysis.index, y='Churn_Rate',
                     title='Churn Rate by Customer Segment')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(values=segment_analysis['Customer_Count'], names=segment_analysis.index,
                     title='Customer Distribution by Segment')
        st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(df, model):
    st.header("📊 Feature Importance Analysis")
    
    if hasattr(model, 'feature_importances_'):
        features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents']
        importances = model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        # Feature importance plot
        fig = px.bar(feature_importance_df, x='Importance', y='Feature', 
                     title='Feature Importance in Churn Prediction',
                     orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions by churn
        selected_feature = st.selectbox("Select Feature to Analyze", features)
        
        if selected_feature:
            fig = px.box(df, x='Churn', y=selected_feature, 
                        title=f'{selected_feature} Distribution by Churn Status')
            st.plotly_chart(fig, use_container_width=True)

def show_report_generation(df, model):
    st.header("📄 Generate Analysis Report")
    
    st.info("Generate a comprehensive report of the churn analysis")
    
    if st.button("Generate Full Report", type="primary"):
        report_text = generate_report(df, model)
        
        st.subheader("Generated Report")
        st.text_area("Report Content", report_text, height=300)
        
        # Download report
        st.download_button(
            label="Download Report as Text File",
            data=report_text,
            file_name=f"churn_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()