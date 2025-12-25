import streamlit as st
import pandas as pd
import numpy as np
import traceback
from io import StringIO

# Import all modules
import utils
import eda
import issue_detection
import preprocessing
import models
import optimization
import evaluation
import report_generator


# Page configuration
st.set_page_config(
    page_title="AutoML Classification System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #f59e0b;
        --secondary-color: #1f2937;
        --success-color: #10b981;
        --error-color: #ef4444;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: #ffffff;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #f59e0b;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #f59e0b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
        transition: transform 0.2s;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: white;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: white;
    }
    
    /* Section headers */
    .section-header {
        background: #1f2937;
        color: #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Success box */
    .success-box {
        background: #10b981;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Error box */
    .error-box {
        background: #ef4444;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Warning box */
    .warning-box {
        background: #f59e0b;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Info box */
    .info-box {
        background: #1f2937;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1f2937;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
        background-color: #f59e0b;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        background-color: #d97706;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #f59e0b;
    }
    
    /* Step badge */
    .step-badge {
        display: inline-block;
        background: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    
    h2, h3 {
        color: #f59e0b;
    }
    
    p, span, div {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'diagnostics' not in st.session_state:
    st.session_state.diagnostics = None
if 'preprocessing_config' not in st.session_state:
    st.session_state.preprocessing_config = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None


def main():
    # Title and description with custom styling
    st.markdown('<h1 class="main-title">🤖 AutoML Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ End-to-end automated machine learning for classification tasks ✨</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Workflow Steps")
        steps = [
            "1️⃣ Upload Dataset",
            "2️⃣ Exploratory Data Analysis",
            "3️⃣ Issue Detection",
            "4️⃣ Select Target Variable",
            "5️⃣ Configure Preprocessing",
            "6️⃣ Train Models",
            "7️⃣ Optimize Hyperparameters",
            "8️⃣ Evaluate & Compare",
            "9️⃣ Generate Report"
        ]
        
        for i, step_name in enumerate(steps, 1):
            if i < st.session_state.step:
                st.success(step_name + " ✓")
            elif i == st.session_state.step:
                st.info(step_name + " ⏳")
            else:
                st.text(step_name)
        
        st.markdown("---")
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Step 1: Upload Dataset
    if st.session_state.step == 1:
        step_upload_dataset()
    
    # Step 2: EDA
    elif st.session_state.step == 2:
        step_exploratory_analysis()
    
    # Step 3: Issue Detection
    elif st.session_state.step == 3:
        step_issue_detection()
    
    # Step 4: Target Selection
    elif st.session_state.step == 4:
        step_select_target()
    
    # Step 5: Preprocessing
    elif st.session_state.step == 5:
        step_configure_preprocessing()
    
    # Step 6: Train Models
    elif st.session_state.step == 6:
        step_train_models()
    
    # Step 7: Optimize
    elif st.session_state.step == 7:
        step_optimize_models()
    
    # Step 8: Evaluate
    elif st.session_state.step == 8:
        step_evaluate_models()
    
    # Step 9: Report
    elif st.session_state.step == 9:
        step_generate_report()


def step_upload_dataset():
    """Step 1: Upload and validate dataset."""
    st.markdown('<div class="section-header">1️⃣ Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">📁 Upload a CSV file (max 100 MB). The dataset must have at least 2 columns and 10 rows.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader')
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            # Validate
            is_valid, error_msg = utils.validate_dataframe(df)
            
            if not is_valid:
                st.error(f"❌ Invalid dataset: {error_msg}")
                return
            
            st.markdown('<div class="success-box">✅ Dataset uploaded successfully!</div>', unsafe_allow_html=True)
            
            # Display basic info with custom metric cards
            basic_stats = utils.get_basic_stats(df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Rows</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value">{len(df.columns)}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Numeric</div>
                    <div class="metric-value">{basic_stats['n_numeric']}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Categorical</div>
                    <div class="metric-value">{basic_stats['n_categorical']}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Show preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show data types
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtypes_df, use_container_width=True)
            
            # Data type conversion option
            st.subheader("🔄 Convert Data Types (Optional)")
            with st.expander("Click to modify column data types"):
                st.info("💡 Convert columns to different data types. Only valid conversions will be applied.")
                
                conversion_made = False
                dtype_options = ['Keep Current', 'int', 'float', 'string', 'category']
                
                # Create columns for better layout
                for col in df.columns:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.text(f"**{col}**")
                    with col2:
                        st.text(f"Current: {df[col].dtype}")
                    with col3:
                        new_dtype = st.selectbox(
                            "Convert to",
                            dtype_options,
                            key=f"dtype_{col}",
                            label_visibility="collapsed"
                        )
                        
                        if new_dtype != 'Keep Current':
                            try:
                                if new_dtype == 'int':
                                    df[col] = pd.to_numeric(df[col], errors='raise').astype('int64')
                                    st.success(f"✓ Converted to int", icon="✅")
                                    conversion_made = True
                                elif new_dtype == 'float':
                                    df[col] = pd.to_numeric(df[col], errors='raise').astype('float64')
                                    st.success(f"✓ Converted to float", icon="✅")
                                    conversion_made = True
                                elif new_dtype == 'string':
                                    df[col] = df[col].astype(str)
                                    st.success(f"✓ Converted to string", icon="✅")
                                    conversion_made = True
                                elif new_dtype == 'category':
                                    df[col] = df[col].astype('category')
                                    st.success(f"✓ Converted to category", icon="✅")
                                    conversion_made = True
                            except Exception as e:
                                st.error(f"❌ Cannot convert to {new_dtype}: {str(e)}", icon="⚠️")
                
                if conversion_made:
                    st.success("✅ Data type conversions applied successfully!")
                    st.info("Updated data types will be used for analysis. Click 'Proceed to EDA' to continue.")
            
            # Proceed button
            if st.button("➡️ Proceed to EDA", type="primary", use_container_width=True):
                st.session_state.df = df
                st.session_state.step = 2
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.code(traceback.format_exc())


def step_exploratory_analysis():
    """Step 2: Exploratory Data Analysis."""
    st.markdown('<div class="section-header">2️⃣ Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    feature_types = utils.get_feature_types(df)
    
    # Basic statistics
    st.subheader("📊 Dataset Statistics")
    basic_stats = utils.get_basic_stats(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", basic_stats['n_rows'])
        st.metric("Numeric Features", basic_stats['n_numeric'])
    with col2:
        st.metric("Total Columns", basic_stats['n_columns'])
        st.metric("Categorical Features", basic_stats['n_categorical'])
    with col3:
        st.metric("Memory Usage", f"{basic_stats['memory_mb']:.2f} MB")
        st.metric("Duplicate Rows", basic_stats['duplicate_rows'])
    
    # Missing values
    st.subheader("🔍 Missing Values")
    missing_summary = eda.get_missing_value_summary(df)
    
    if not missing_summary.empty:
        st.dataframe(missing_summary, use_container_width=True)
        fig = eda.generate_missing_value_plot(df)
        if fig:
            st.pyplot(fig)
    else:
        st.success("✅ No missing values detected!")
    
    # Numeric features
    if feature_types['numeric']:
        st.subheader("📈 Numeric Features Distribution")
        fig = eda.generate_numeric_histograms(df, feature_types['numeric'])
        if fig:
            st.pyplot(fig)
        
        st.subheader("📦 Boxplots (Outlier Detection)")
        fig = eda.generate_boxplots(df, feature_types['numeric'])
        if fig:
            st.pyplot(fig)
        
        st.subheader("🔥 Correlation Heatmap")
        fig = eda.generate_correlation_heatmap(df, feature_types['numeric'])
        if fig:
            st.pyplot(fig)
        
        # Strong correlations
        strong_corr = eda.get_strong_correlations(df, feature_types['numeric'], threshold=0.7)
        if strong_corr:
            st.subheader("⚠️ Strong Correlations (|r| ≥ 0.7)")
            corr_df = pd.DataFrame(strong_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
            st.dataframe(corr_df, use_container_width=True)
    
    # Categorical features
    if feature_types['categorical']:
        st.subheader("📊 Categorical Features Distribution")
        fig = eda.generate_categorical_barplots(df, feature_types['categorical'])
        if fig:
            st.pyplot(fig)
    
    # Proceed button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("➡️ Proceed to Issue Detection", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


def step_issue_detection():
    """Step 3: Detect data quality issues."""
    st.markdown('<div class="section-header">3️⃣ Issue Detection</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    feature_types = utils.get_feature_types(df)
    
    # Outlier detection method selection
    st.info("🔧 Configure outlier detection method before running diagnostics")
    col1, col2 = st.columns([1, 3])
    with col1:
        outlier_method = st.radio(
            "Outlier Detection Method",
            ['iqr', 'zscore'],
            format_func=lambda x: 'IQR Method (Interquartile Range)' if x == 'iqr' else 'Z-Score Method (Standard Deviations)',
            help="**IQR**: Detects outliers beyond 1.5×IQR from Q1/Q3. Good for skewed data.\n\n**Z-Score**: Detects values >3 standard deviations from mean. Assumes normal distribution."
        )
    with col2:
        if outlier_method == 'iqr':
            st.markdown("""
            **IQR Method (Recommended for most cases)**
            - Detects values beyond Q1 - 1.5×IQR and Q3 + 1.5×IQR
            - Robust to skewed distributions
            - Less sensitive to extreme values
            - Works well with non-normal data
            """)
        else:
            st.markdown("""
            **Z-Score Method**
            - Detects values beyond ±3 standard deviations from mean
            - Assumes data follows normal distribution
            - More sensitive to extreme values
            - Best for symmetric, bell-shaped distributions
            """)
    
    with st.spinner("🔍 Running comprehensive diagnostics..."):
        diagnostics = issue_detection.run_comprehensive_diagnostics(
            df, feature_types['numeric'], feature_types['categorical'],
            outlier_method=outlier_method
        )
        st.session_state.diagnostics = diagnostics
    
    # Summary
    issues_summary = issue_detection.get_issues_summary(diagnostics)
    
    st.subheader("📋 Issues Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missing Values", issues_summary['missing_values'])
    with col2:
        st.metric("Outliers", issues_summary['outliers'])
    with col3:
        st.metric("High Cardinality", issues_summary['high_cardinality'])
    with col4:
        st.metric("Total Issues", issues_summary['total_issues'])
    
    # Detailed findings
    if diagnostics['missing_values']['has_missing']:
        with st.expander("⚠️ Missing Values Details", expanded=True):
            st.write(f"**Total Missing:** {diagnostics['missing_values']['total_missing']} "
                    f"({diagnostics['missing_values']['global_percentage']:.2f}%)")
            
            missing_df = pd.DataFrame([
                {'Column': col, 'Count': info['count'], 'Percentage': f"{info['percentage']:.2f}%"}
                for col, info in diagnostics['missing_values']['columns_with_missing'].items()
            ])
            st.dataframe(missing_df, use_container_width=True)
    
    if diagnostics['outliers']['columns_with_outliers']:
        with st.expander("⚠️ Outliers Details", expanded=True):
            outlier_df = pd.DataFrame([
                {'Column': col, 'Count': info['count'], 'Percentage': f"{info['percentage']:.2f}%",
                 'Method': info['method']}
                for col, info in diagnostics['outliers']['outliers_per_column'].items()
                if info['count'] > 0
            ])
            st.dataframe(outlier_df, use_container_width=True)
    
    if diagnostics['high_cardinality']:
        with st.expander("⚠️ High Cardinality Features"):
            st.dataframe(pd.DataFrame(diagnostics['high_cardinality']), use_container_width=True)
    
    if diagnostics['constant_features']:
        with st.expander("⚠️ Near-Constant Features"):
            st.dataframe(pd.DataFrame(diagnostics['constant_features']), use_container_width=True)
    
    if diagnostics['duplicates']['has_duplicates']:
        with st.expander("⚠️ Duplicate Rows"):
            st.write(f"**Found {diagnostics['duplicates']['n_duplicates']} duplicate rows "
                    f"({diagnostics['duplicates']['percentage']:.2f}%)**")
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("➡️ Select Target Variable", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def step_select_target():
    """Step 4: Select target variable."""
    st.markdown('<div class="section-header">4️⃣ Select Target Variable</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    st.markdown('<div class="info-box">📌 Select the column you want to predict (target variable)</div>', unsafe_allow_html=True)
    
    target_col = st.selectbox(
        "Target Variable",
        options=['-- Select --'] + list(df.columns),
        key='target_selector'
    )
    
    if target_col != '-- Select --':
        st.session_state.target_col = target_col
        
        # Show target distribution
        st.subheader("🎯 Target Distribution")
        fig = eda.generate_target_distribution(df, target_col)
        if fig:
            st.pyplot(fig)
        
        # Class imbalance analysis
        class_imbalance = issue_detection.detect_class_imbalance(df[target_col])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Classes", class_imbalance['n_classes'])
        with col2:
            st.metric("Majority Class %", f"{class_imbalance['majority_percentage']:.2f}%")
        with col3:
            st.metric("Minority Class %", f"{class_imbalance['minority_percentage']:.2f}%")
        
        if class_imbalance['is_imbalanced']:
            st.markdown(f'<div class="warning-box">⚠️ Class imbalance detected! Imbalance ratio: {class_imbalance["imbalance_ratio"]:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ Classes are relatively balanced</div>', unsafe_allow_html=True)
        
        # Store in diagnostics
        if st.session_state.diagnostics:
            st.session_state.diagnostics['class_imbalance'] = class_imbalance
        
        # Navigation
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("⬅️ Back"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("➡️ Configure Preprocessing", type="primary", use_container_width=True):
                st.session_state.step = 5
                st.rerun()
    else:
        st.warning("⚠️ Please select a target variable to proceed")
        if st.button("⬅️ Back"):
            st.session_state.step = 3
            st.rerun()


def step_configure_preprocessing():
    """Step 5: Configure preprocessing options."""
    st.markdown('<div class="section-header">5️⃣ Configure Preprocessing</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    diagnostics = st.session_state.diagnostics
    target_col = st.session_state.target_col
    
    st.markdown('<div class="info-box">⚙️ Configure how to handle data issues. All transformations will be fit on training data only.</div>', unsafe_allow_html=True)
    
    config = {}
    
    # Train-test split
    st.subheader("📊 Train-Test Split")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        config['test_size'] = test_size
    with col2:
        stratify = st.checkbox("Stratified Split", value=True)
        config['stratify'] = stratify
    
    config['random_state'] = 42
    
    # Missing values
    if diagnostics['missing_values']['has_missing']:
        st.subheader("🔧 Missing Value Imputation")
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        config['handle_missing'] = handle_missing
        
        if handle_missing:
            col1, col2 = st.columns(2)
            with col1:
                numeric_strategy = st.selectbox(
                    "Numeric Strategy",
                    ['mean', 'median', 'constant']
                )
                config['numeric_imputation'] = numeric_strategy
                if numeric_strategy == 'constant':
                    config['constant_value'] = st.number_input("Constant Value", value=0)
            
            with col2:
                categorical_strategy = st.selectbox(
                    "Categorical Strategy",
                    ['most_frequent', 'constant']
                )
                config['categorical_imputation'] = categorical_strategy
    
    # Outliers
    if diagnostics['outliers']['columns_with_outliers']:
        st.subheader("📉 Outlier Handling")
        remove_outliers = st.checkbox("Remove Outliers", value=False)
        config['remove_outliers'] = remove_outliers
        
        if remove_outliers:
            st.warning("⚠️ This will remove outlier rows from the training set only")
            # Collect all outlier indices (would need implementation)
            config['outlier_indices'] = []
    
    # Encoding
    st.subheader("🔤 Categorical Encoding")
    encode_categorical = st.checkbox("Encode Categorical Features", value=True)
    config['encode_categorical'] = encode_categorical
    
    if encode_categorical:
        encoding_type = st.radio(
            "Encoding Method",
            ['onehot', 'ordinal'],
            format_func=lambda x: 'One-Hot Encoding' if x == 'onehot' else 'Ordinal Encoding'
        )
        config['encoding_type'] = encoding_type
    
    # Scaling
    st.subheader("⚖️ Feature Scaling")
    scale_features = st.checkbox("Scale Features", value=True)
    config['scale_features'] = scale_features
    
    if scale_features:
        scaling_type = st.radio(
            "Scaling Method",
            ['standard', 'minmax'],
            format_func=lambda x: 'StandardScaler (z-score)' if x == 'standard' else 'MinMaxScaler (0-1)'
        )
        config['scaling_type'] = scaling_type
    
    # Class imbalance
    if diagnostics.get('class_imbalance', {}).get('is_imbalanced', False):
        st.subheader("⚖️ Class Imbalance Handling")
        handle_imbalance = st.checkbox("Handle Class Imbalance", value=False)
        config['handle_imbalance'] = handle_imbalance
        
        if handle_imbalance:
            imbalance_method = st.selectbox(
                "Resampling Method",
                ['none', 'undersample', 'oversample', 'smote', 'adasyn'],
                format_func=lambda x: {
                    'none': 'None',
                    'undersample': 'Random Undersampling',
                    'oversample': 'Random Oversampling',
                    'smote': 'SMOTE (Synthetic Minority Over-sampling)',
                    'adasyn': 'ADASYN (Adaptive Synthetic Sampling)'
                }[x]
            )
            config['imbalance_method'] = imbalance_method
    
    st.session_state.preprocessing_config = config
    
    # Show summary
    st.subheader("📝 Configuration Summary")
    st.json(config)
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("➡️ Train Models", type="primary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()


@st.cache_data
def run_preprocessing(_df, target_col, config):
    """Cached preprocessing execution."""
    return preprocessing.prepare_data_pipeline(_df, target_col, config)


def step_train_models():
    """Step 6: Train all models."""
    st.markdown('<div class="section-header">6️⃣ Train Models</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    target_col = st.session_state.target_col
    config = st.session_state.preprocessing_config
    
    # Execute preprocessing
    if st.session_state.processed_data is None:
        with st.spinner("⚙️ Preprocessing data..."):
            processed_data = run_preprocessing(df, target_col, config)
            st.session_state.processed_data = processed_data
        
        st.success("✅ Data preprocessing completed!")
        
        # Show summary
        summary = processed_data['summary']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", summary['final_shape'][0])
        with col2:
            st.metric("Features", summary['final_shape'][1])
        with col3:
            st.metric("Rows Removed", summary['rows_removed'])
    
    processed_data = st.session_state.processed_data
    
    # Train models
    if st.session_state.model_results is None:
        with st.spinner("🤖 Training all models... This may take a while."):
            model_results = models.train_all_models(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            st.session_state.model_results = model_results
        
        st.success("✅ All models trained successfully!")
    
    model_results = st.session_state.model_results
    
    # Show training summary
    st.subheader("📊 Training Summary")
    summary_table = models.create_model_summary_table(model_results)
    st.dataframe(summary_table, use_container_width=True)
    
    # Quick evaluation
    st.subheader("⚡ Quick Evaluation")
    with st.spinner("📊 Evaluating models..."):
        eval_results = evaluation.evaluate_all_models(
            model_results,
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        st.session_state.evaluation_results = eval_results
    
    comparison_table = evaluation.create_comparison_table(eval_results)
    st.dataframe(comparison_table, use_container_width=True)
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 5
            st.rerun()
    with col2:
        if st.button("➡️ Optimize Hyperparameters", type="primary", use_container_width=True):
            st.session_state.step = 7
            st.rerun()


def step_optimize_models():
    """Step 7: Hyperparameter optimization."""
    st.markdown('<div class="section-header">7️⃣ Hyperparameter Optimization</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">🔧 Optimize model hyperparameters using cross-validation</div>', unsafe_allow_html=True)
    
    # Optimization settings
    col1, col2, col3 = st.columns(3)
    with col1:
        opt_method = st.radio("Search Method", ['grid', 'random'],
                             format_func=lambda x: 'GridSearchCV' if x == 'grid' else 'RandomizedSearchCV')
    with col2:
        cv_folds = st.slider("CV Folds", 3, 10, 5)
    with col3:
        if opt_method == 'random':
            n_iter = st.slider("Iterations", 10, 100, 50)
        else:
            n_iter = 50
    
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        model_results = st.session_state.model_results
        processed_data = st.session_state.processed_data
        model_configs = models.get_model_configs()
        
        with st.spinner("🔍 Optimizing hyperparameters... This will take several minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            opt_results = optimization.optimize_all_models(
                model_results,
                model_configs,
                processed_data['X_train'],
                processed_data['y_train'],
                method=opt_method,
                n_iter=n_iter,
                cv=cv_folds
            )
            
            progress_bar.progress(100)
            st.session_state.optimization_results = opt_results
        
        st.success("✅ Optimization completed!")
    
    # Show results
    if st.session_state.optimization_results:
        opt_results = st.session_state.optimization_results
        
        st.subheader("📊 Optimization Results")
        summary_table = optimization.create_optimization_summary(opt_results)
        st.dataframe(summary_table, use_container_width=True)
        
        st.subheader("🎯 Best Parameters")
        best_params = optimization.get_best_params_summary(opt_results)
        for model_name, params in best_params.items():
            with st.expander(f"{model_name}"):
                st.json(params)
        
        # Retrain with best params
        if st.button("🔄 Retrain with Best Parameters"):
            with st.spinner("🤖 Retraining models with optimized parameters..."):
                processed_data = st.session_state.processed_data
                retrained_results = optimization.retrain_with_best_params(
                    opt_results,
                    processed_data['X_train'],
                    processed_data['y_train'],
                    processed_data['X_test'],
                    processed_data['y_test']
                )
                st.session_state.model_results = retrained_results
            st.success("✅ Models retrained with optimized parameters!")
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 6
            st.rerun()
    with col2:
        if st.button("➡️ Evaluate & Compare", type="primary", use_container_width=True):
            st.session_state.step = 8
            st.rerun()


def step_evaluate_models():
    """Step 8: Comprehensive evaluation and comparison."""
    st.markdown('<div class="section-header">8️⃣ Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    
    eval_results = st.session_state.evaluation_results
    processed_data = st.session_state.processed_data
    
    # Re-evaluate if optimization was done
    if st.session_state.optimization_results:
        with st.spinner("📊 Re-evaluating optimized models..."):
            eval_results = evaluation.evaluate_all_models(
                st.session_state.model_results,
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            st.session_state.evaluation_results = eval_results
    
    # Comparison table
    st.subheader("📊 Performance Comparison")
    comparison_table = evaluation.create_comparison_table(eval_results)
    st.dataframe(comparison_table, use_container_width=True)
    
    # Metrics visualization
    st.subheader("📈 Metrics Comparison")
    fig = evaluation.plot_metrics_comparison(eval_results)
    if fig:
        st.pyplot(fig)
    
    # Confusion matrices
    st.subheader("🎯 Confusion Matrices")
    fig = evaluation.plot_all_confusion_matrices(eval_results)
    if fig:
        st.pyplot(fig)
    
    # ROC curves (if binary classification)
    y_test = processed_data['y_test']
    if len(np.unique(y_test)) == 2:
        st.subheader("📉 ROC Curves")
        fig = evaluation.plot_all_roc_curves(eval_results, y_test)
        if fig:
            st.pyplot(fig)
    
    # Best model
    st.subheader("🏆 Best Performing Model")
    best_model_name, best_result = evaluation.get_best_model(eval_results, metric='f1_score')
    
    if best_model_name:
        st.success(f"**{best_model_name}** achieved the best F1-score!")
        
        metrics = best_result['test_metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    
    # Download comparison
    st.subheader("💾 Download Results")
    csv = comparison_table.to_csv(index=False)
    st.download_button(
        "📥 Download Comparison Table (CSV)",
        csv,
        "model_comparison.csv",
        "text/csv",
        use_container_width=True
    )
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 7
            st.rerun()
    with col2:
        if st.button("➡️ Generate Report", type="primary", use_container_width=True):
            st.session_state.step = 9
            st.rerun()


def step_generate_report():
    """Step 9: Generate and download comprehensive report."""
    st.markdown('<div class="section-header">9️⃣ Generate Report</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">📄 Generate a comprehensive report with all findings</div>', unsafe_allow_html=True)
    
    # Collect all data
    dataset_info = utils.get_basic_stats(st.session_state.df)
    feature_types = utils.get_feature_types(st.session_state.df)
    eda_summary = eda.compute_eda_summary(
        st.session_state.df,
        feature_types['numeric'],
        feature_types['categorical']
    )
    
    # Get best model
    best_model_name, _ = evaluation.get_best_model(
        st.session_state.evaluation_results,
        metric='f1_score'
    )
    
    # Generate report
    with st.spinner("📝 Generating report..."):
        markdown_report = report_generator.generate_markdown_report(
            dataset_info=dataset_info,
            eda_summary=eda_summary,
            diagnostics=st.session_state.diagnostics,
            preprocessing_summary=st.session_state.processed_data['summary'],
            evaluation_results=st.session_state.evaluation_results,
            optimization_results=st.session_state.optimization_results,
            best_model_name=best_model_name
        )
    
    st.success("✅ Report generated successfully!")
    
    # Preview
    with st.expander("👀 Preview Report", expanded=True):
        st.markdown(markdown_report)
    
    # Download options
    st.subheader("💾 Download Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        markdown_bytes = report_generator.create_downloadable_report('markdown', markdown_report)
        st.download_button(
            "📥 Download Markdown",
            markdown_bytes,
            "automl_report.md",
            "text/markdown",
            use_container_width=True
        )
    
    with col2:
        html_bytes = report_generator.create_downloadable_report('html', markdown_report)
        st.download_button(
            "📥 Download HTML",
            html_bytes,
            "automl_report.html",
            "text/html",
            use_container_width=True
        )
    
    with col3:
        try:
            pdf_bytes = report_generator.create_downloadable_report('pdf', markdown_report)
            st.download_button(
                "📥 Download PDF",
                pdf_bytes,
                "automl_report.pdf",
                "application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"PDF generation unavailable: {str(e)}")
    
    # Final message with enhanced styling
    st.markdown("---")
    st.markdown('<div class="success-box" style="text-align: center; font-size: 1.2rem;">🎉 AutoML workflow completed successfully! 🎉</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="text-align: center;">You can now reset the application to analyze another dataset.</div>', unsafe_allow_html=True)
    
    if st.button("⬅️ Back"):
        st.session_state.step = 8
        st.rerun()


if __name__ == "__main__":
    main()
