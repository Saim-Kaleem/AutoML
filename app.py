import streamlit as st
import pandas as pd
import numpy as np
import traceback
import logging
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    
    /* Main container styling - Use light background for Streamlit Cloud compatibility */
    .main {
        background: #f8f9fa;
    }
    
    /* Override Streamlit default text colors for light theme */
    .stApp {
        background: #f8f9fa;
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
        background: #1f2937 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    
    /* Sidebar text elements */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText {
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
    
    /* Text colors - Use dark colors for light background */
    h1 {
        color: #1f2937 !important;
    }
    
    h2, h3 {
        color: #f59e0b !important;
    }
    
    h4, h5, h6 {
        color: #1f2937 !important;
    }
    
    /* Default text should be dark on light background */
    p {
        color: #374151 !important;
    }
    
    /* Labels and general text */
    label {
        color: #1f2937 !important;
    }
    
    /* Streamlit native elements */
    .stMarkdown {
        color: #374151;
    }
    
    /* Streamlit native metrics styling */
    [data-testid="stMetricValue"] {
        color: #f59e0b !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
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
if 'initial_evaluation_results' not in st.session_state:
    st.session_state.initial_evaluation_results = None
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
        st.header("Workflow Steps")
        steps = [
            "1. Upload Dataset",
            "2. Exploratory Data Analysis",
            "3. Issue Detection",
            "4. Select Target Variable",
            "5. Configure Preprocessing",
            "6. Train Models",
            "7. Optimize Hyperparameters",
            "8. Evaluate & Compare",
            "9. Generate Report"
        ]
        
        for i, step_name in enumerate(steps, 1):
            if i < st.session_state.step:
                st.success(step_name + " ✓")
            elif i == st.session_state.step:
                st.info(step_name)
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
    st.markdown('<div class="section-header">Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">📁 Upload a CSV file (max 100 MB). The dataset must have at least 2 columns and 10 rows.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader', label_visibility="hidden")
    
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
            st.subheader("Convert Data Types (Optional)")
            with st.expander("Click to modify column data types"):
                st.info("💡 Convert columns to different data types. Only valid conversions will be applied.")
                
                conversion_made = False
                dtype_options = ['Keep Current', 'int', 'float', 'string', 'category']
                
                # Create columns for better layout
                for col in df.columns:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.text(f"{col}")
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
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    feature_types = utils.get_feature_types(df)
    
    # Basic statistics
    st.subheader("Dataset Statistics")
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
    st.subheader("Missing Values")
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
        st.subheader("Numeric Features Distribution")
        fig = eda.generate_numeric_histograms(df, feature_types['numeric'])
        if fig:
            st.pyplot(fig)
        
        st.subheader("Boxplots (Outlier Detection)")
        fig = eda.generate_boxplots(df, feature_types['numeric'])
        if fig:
            st.pyplot(fig)
        
        st.subheader("Correlation Heatmap")
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
        st.subheader("Categorical Features Distribution")
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
    st.markdown('<div class="section-header">Issue Detection</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    feature_types = utils.get_feature_types(df)
    
    # Detection configuration
    st.info("🔧 Configure detection methods before running diagnostics")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        outlier_method = st.radio(
            "Outlier Detection Method",
            ['iqr', 'zscore'],
            format_func=lambda x: 'IQR Method (Interquartile Range)' if x == 'iqr' else 'Z-Score Method (Standard Deviations)',
            help="**IQR**: Detects outliers beyond 1.5×IQR from Q1/Q3. Good for skewed data.\n\n**Z-Score**: Detects values >3 standard deviations from mean. Assumes normal distribution."
        )
    with col3:
        cardinality_threshold = st.number_input(
            "High Cardinality Threshold",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="**Cardinality** = number of unique values in a column. High cardinality columns (like IDs) create too many features when encoded. This threshold flags columns that should be excluded. Recommended: 10-20 for classification."
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
            outlier_method=outlier_method,
            cardinality_threshold=cardinality_threshold
        )
        st.session_state.diagnostics = diagnostics
    
    # Summary
    issues_summary = issue_detection.get_issues_summary(diagnostics)
    
    st.subheader("Issues Summary")
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
    st.markdown('<div class="section-header">Select Target Variable</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    diagnostics = st.session_state.diagnostics
    
    st.markdown('<div class="info-box">📌 Select the column you want to predict (target variable)</div>', unsafe_allow_html=True)
    
    # Get cardinality threshold from diagnostics or use default
    cardinality_threshold = 20  # Default for classification
    
    st.info(f"ℹ️ Target variable must be categorical with ≤ {cardinality_threshold} unique classes")
    
    target_col = st.selectbox(
        "Target Variable",
        options=['-- Select --'] + list(df.columns),
        key='target_selector'
    )
    
    if target_col != '-- Select --':
        target_data = df[target_col].copy()
        n_unique = target_data.nunique()
        original_dtype = target_data.dtype
        
        logger.info(f"Target selected: {target_col}, dtype: {original_dtype}, unique values: {n_unique}")
        
        # Validation and conversion logic
        is_valid = True
        conversion_performed = False
        error_message = None
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(target_data):
            st.warning(f"⚠️ Target '{target_col}' is numeric ({original_dtype}). Attempting to convert to categorical...")
            
            # Check cardinality before conversion
            if n_unique > cardinality_threshold:
                is_valid = False
                error_message = f"❌ Cannot use '{target_col}' as target: {n_unique} unique values exceeds threshold of {cardinality_threshold}. This is a regression problem, not classification."
                logger.error(f"Target validation failed: {n_unique} unique values > {cardinality_threshold}")
            else:
                # Convert to categorical
                try:
                    df[target_col] = df[target_col].astype('category')
                    st.session_state.df = df
                    target_data = df[target_col]
                    conversion_performed = True
                    st.success(f"✅ Successfully converted '{target_col}' to categorical type")
                    logger.info(f"Converted target '{target_col}' to categorical")
                except Exception as e:
                    is_valid = False
                    error_message = f"❌ Failed to convert '{target_col}' to categorical: {str(e)}"
                    logger.error(f"Target conversion failed: {str(e)}")
        
        # Check if categorical/object type
        elif pd.api.types.is_object_dtype(target_data) or pd.api.types.is_categorical_dtype(target_data):
            # Check cardinality
            if n_unique > cardinality_threshold:
                is_valid = False
                error_message = f"❌ Cannot use '{target_col}' as target: {n_unique} unique categories exceeds threshold of {cardinality_threshold}. Consider grouping categories or choosing a different target."
                logger.error(f"Target validation failed: {n_unique} categories > {cardinality_threshold}")
            else:
                st.success(f"✅ Target '{target_col}' is categorical with {n_unique} classes")
                logger.info(f"Target '{target_col}' validated: {n_unique} classes")
        else:
            is_valid = False
            error_message = f"❌ Unsupported data type for target '{target_col}': {original_dtype}"
            logger.error(f"Target validation failed: unsupported dtype {original_dtype}")
        
        # Display validation result
        if not is_valid:
            st.error(error_message)
            st.info("💡 **Tip**: For classification, target should have a reasonable number of distinct categories (typically 2-20). For more classes, consider multi-class classification or grouping similar categories.")
            
            if st.button("⬅️ Back to Issue Detection"):
                st.session_state.step = 3
                st.rerun()
            return
        
        # If validation passed, store and continue
        st.session_state.target_col = target_col
        
        # Show target distribution
        st.subheader("Target Distribution")
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
    st.markdown('<div class="section-header">Configure Preprocessing</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    diagnostics = st.session_state.diagnostics
    target_col = st.session_state.target_col
    
    st.markdown('<div class="info-box">⚙️ Configure how to handle data issues. All transformations will be fit on training data only.</div>', unsafe_allow_html=True)
    
    config = {}
    
    # Train-test split
    st.subheader("Train-Test Split")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Test Set Size", 
            0.1, 0.4, 0.2, 0.05,
            help="Percentage of data reserved for testing. 20% is standard. Higher = more reliable test metrics but less training data."
        )
        config['test_size'] = test_size
    with col2:
        stratify = st.checkbox(
            "Stratified Split", 
            value=True,
            help="Maintains the same class distribution in train and test sets. Recommended for imbalanced datasets to ensure both sets represent all classes proportionally."
        )
        config['stratify'] = stratify
    
    config['random_state'] = 42
    
    # Feature Selection
    st.subheader("Feature Selection")
    st.markdown("Choose which features to include in the model training.")
    
    # Get all feature columns (excluding target)
    all_features = [col for col in df.columns if col != target_col]
    
    with st.expander("Manual Feature Selection", expanded=False):
        st.info("💡 **Tip**: Exclude irrelevant columns like IDs, timestamps, or domain-specific features that shouldn't influence predictions.")
        
        # Option to select features to keep or drop
        selection_mode = st.radio(
            "Selection Mode",
            ["Keep All Features", "Select Features to Keep", "Select Features to Drop"],
            help="Choose how you want to control feature selection"
        )
        
        if selection_mode == "Select Features to Keep":
            selected_features = st.multiselect(
                "Select Features to Keep",
                all_features,
                default=all_features,
                help="Only these features will be used for training"
            )
            config['manual_feature_selection'] = True
            config['features_to_keep'] = selected_features
            config['features_to_drop'] = [col for col in all_features if col not in selected_features]
            
            if len(selected_features) == 0:
                st.error("⚠️ You must select at least one feature!")
            else:
                st.success(f"✅ Selected {len(selected_features)} out of {len(all_features)} features")
        
        elif selection_mode == "Select Features to Drop":
            features_to_drop = st.multiselect(
                "Select Features to Drop",
                all_features,
                help="These features will be excluded from training"
            )
            config['manual_feature_selection'] = True
            config['features_to_drop'] = features_to_drop
            config['features_to_keep'] = [col for col in all_features if col not in features_to_drop]
            
            remaining_features = len(all_features) - len(features_to_drop)
            if remaining_features == 0:
                st.error("⚠️ You cannot drop all features!")
            elif len(features_to_drop) > 0:
                st.success(f"✅ Will drop {len(features_to_drop)} features, keeping {remaining_features}")
            else:
                st.info("No features selected for dropping")
        
        else:  # Keep All Features
            config['manual_feature_selection'] = False
            config['features_to_drop'] = []
            config['features_to_keep'] = all_features
    
    # Constant features
    if diagnostics['constant_features']:
        st.subheader("Constant Features Handling")
        
        # Separate truly constant from near-constant
        truly_constant = [f for f in diagnostics['constant_features'] if f['unique_values'] == 1]
        near_constant = [f for f in diagnostics['constant_features'] if f['unique_values'] > 1]
        
        if truly_constant:
            st.warning(f"⚠️ Found {len(truly_constant)} truly constant feature(s) (all values identical). These will be automatically removed.")
            with st.expander("View Constant Features"):
                const_df = pd.DataFrame(truly_constant)
                st.dataframe(const_df, use_container_width=True)
            config['remove_constant_features'] = True
            config['constant_features_to_remove'] = [f['column'] for f in truly_constant]
        else:
            config['remove_constant_features'] = False
            config['constant_features_to_remove'] = []
        
        if near_constant:
            st.info(f"ℹ️ Found {len(near_constant)} near-constant feature(s) (one value dominates >95%)")
            with st.expander("View Near-Constant Features"):
                near_const_df = pd.DataFrame(near_constant)
                st.dataframe(near_const_df, use_container_width=True)
            
            remove_near_constant = st.checkbox(
                "Remove Near-Constant Features",
                value=False,
                help="Remove features where one value dominates >95% of the data. These provide little information but consume resources. Generally safe to remove."
            )
            config['remove_near_constant_features'] = remove_near_constant
            if remove_near_constant:
                config['near_constant_features_to_remove'] = [f['column'] for f in near_constant]
                st.success(f"✅ Will remove {len(near_constant)} near-constant features")
            else:
                config['near_constant_features_to_remove'] = []
        else:
            config['remove_near_constant_features'] = False
            config['near_constant_features_to_remove'] = []
    else:
        config['remove_constant_features'] = False
        config['constant_features_to_remove'] = []
        config['remove_near_constant_features'] = False
        config['near_constant_features_to_remove'] = []
    
    # Missing values
    if diagnostics['missing_values']['has_missing']:
        st.subheader("🔧 Missing Value Imputation")
        handle_missing = st.checkbox(
            "Handle Missing Values", 
            value=True,
            help="Replace missing values with estimated values. Recommended to prevent errors during model training."
        )
        config['handle_missing'] = handle_missing
        
        if handle_missing:
            col1, col2 = st.columns(2)
            with col1:
                numeric_strategy = st.selectbox(
                    "Numeric Strategy",
                    ['mean', 'median', 'constant'],
                    help="**Mean**: Average value (sensitive to outliers)\n**Median**: Middle value (robust to outliers, recommended)\n**Constant**: Fill with a specific number"
                )
                config['numeric_imputation'] = numeric_strategy
                if numeric_strategy == 'constant':
                    config['constant_value'] = st.number_input(
                        "Constant Value", 
                        value=0,
                        help="The specific number to use for filling missing values"
                    )
            
            with col2:
                categorical_strategy = st.selectbox(
                    "Categorical Strategy",
                    ['most_frequent', 'constant'],
                    help="**Most Frequent**: Use the most common category (recommended)\n**Constant**: Fill with a specific value like 'Unknown'"
                )
                config['categorical_imputation'] = categorical_strategy
    
    # Outliers
    if diagnostics['outliers']['columns_with_outliers']:
        st.subheader("Outlier Handling")
        remove_outliers = st.checkbox(
            "Remove Outliers", 
            value=False,
            help="Remove extreme values that may distort model training. ⚠️ Use cautiously: outliers may be legitimate data points or important anomalies. Only removes from training set, not test set."
        )
        config['remove_outliers'] = remove_outliers
        
        if remove_outliers:
            st.warning("⚠️ This will remove outlier rows from the training set only")
            # Collect all outlier indices (would need implementation)
            config['outlier_indices'] = []
    
    # Encoding
    st.subheader("Categorical Encoding")
    encode_categorical = st.checkbox(
        "Encode Categorical Features", 
        value=True,
        help="Convert text categories into numbers that machine learning models can understand. Required for most algorithms."
    )
    config['encode_categorical'] = encode_categorical
    
    if encode_categorical:
        encoding_type = st.radio(
            "Encoding Method",
            ['onehot', 'ordinal'],
            format_func=lambda x: 'One-Hot Encoding' if x == 'onehot' else 'Ordinal Encoding',
            help="**One-Hot**: Creates binary columns for each category (e.g., Red→[1,0,0], Green→[0,1,0]). Best for nominal categories with no order.\n\n**Ordinal**: Assigns numbers to categories (e.g., Low=1, Medium=2, High=3). Use only when categories have a natural order."
        )
        config['encoding_type'] = encoding_type
        
        # High cardinality handling
        if diagnostics.get('high_cardinality'):
            st.warning(f"⚠️ Found {len(diagnostics['high_cardinality'])} high-cardinality columns")
            
            with st.expander("View High-Cardinality Columns", expanded=True):
                hc_df = pd.DataFrame(diagnostics['high_cardinality'])
                st.dataframe(hc_df, use_container_width=True)
                
                st.info("💡 **Recommendation**: Exclude high-cardinality columns (like IDs) from encoding to prevent feature explosion.")
                
                exclude_high_card = st.checkbox(
                    "Exclude High-Cardinality Columns from Encoding",
                    value=True,
                    help="Prevents creating hundreds/thousands of features from ID-like columns"
                )
                config['exclude_high_cardinality'] = exclude_high_card
                
                if exclude_high_card:
                    # Store the list of high-cardinality column names to exclude
                    high_card_cols = [item['column'] for item in diagnostics['high_cardinality']]
                    config['exclude_high_cardinality_cols'] = high_card_cols
                    st.success(f"✅ These {len(high_card_cols)} columns will be dropped before encoding: {', '.join(high_card_cols)}")
                else:
                    config['exclude_high_cardinality_cols'] = []
                    st.error("⚠️ Warning: Encoding these columns may create thousands of features and cause memory issues!")
    
    # Scaling
    st.subheader("Feature Scaling")
    scale_features = st.checkbox(
        "Scale Features", 
        value=True,
        help="Normalize features to similar ranges. Important for distance-based algorithms (KNN, SVM) and neural networks. Not critical for tree-based models."
    )
    config['scale_features'] = scale_features
    
    if scale_features:
        scaling_type = st.radio(
            "Scaling Method",
            ['standard', 'minmax'],
            format_func=lambda x: 'StandardScaler (z-score)' if x == 'standard' else 'MinMaxScaler (0-1)',
            help="**StandardScaler**: Centers data around mean=0 with std=1. Good for normally distributed data. Values can be negative.\n\n**MinMaxScaler**: Scales data to range [0,1]. Good for bounded data. All values stay positive."
        )
        config['scaling_type'] = scaling_type
    
    # Class imbalance
    if diagnostics.get('class_imbalance', {}).get('is_imbalanced', False):
        st.subheader("Class Imbalance Handling")
        
        imbalance_strategy = st.radio(
            "Choose Strategy",
            ["No Handling", "Class Weights", "Resampling"],
            help="**No Handling**: Train models as-is. May favor majority class.\n\n**Class Weights**: Give higher importance to minority class during training. Memory-efficient, works for most models.\n\n**Resampling**: Modify dataset by adding/removing samples. Changes data distribution."
        )
        
        if imbalance_strategy == "Class Weights":
            config['handle_imbalance'] = False  # No data resampling
            config['use_class_weights'] = True
            
            weight_method = st.selectbox(
                "Class Weight Method",
                ['balanced', 'balanced_subsample'],
                help="**Balanced**: Adjusts weights inversely proportional to class frequencies. Formula: n_samples / (n_classes × class_count)\n\n**Balanced Subsample**: Similar to balanced but computed on bootstrap samples (for Random Forest only)"
            )
            config['class_weight_method'] = weight_method
            
            st.info("✅ Models will automatically use class weights during training (LogisticRegression, RandomForest, SVM)")
            st.warning("⚠️ Note: Some models (KNN, NaiveBayes) don't support class weights and will use default behavior")
            
        elif imbalance_strategy == "Resampling":
            config['handle_imbalance'] = True
            config['use_class_weights'] = False
            
            imbalance_method = st.selectbox(
                "Resampling Method",
                ['undersample', 'oversample', 'smote', 'adasyn'],
                format_func=lambda x: {
                    'undersample': 'Random Undersampling',
                    'oversample': 'Random Oversampling',
                    'smote': 'SMOTE (Synthetic Minority Over-sampling)',
                    'adasyn': 'ADASYN (Adaptive Synthetic Sampling)'
                }[x],
                help="**Undersample**: Randomly remove majority class samples. Fast but loses data.\n\n**Oversample**: Duplicate minority class samples. May cause overfitting.\n\n**SMOTE**: Create synthetic minority samples by interpolating between existing ones. Recommended for most cases.\n\n**ADASYN**: Like SMOTE but focuses on harder-to-learn samples. Best for complex boundaries."
            )
            config['imbalance_method'] = imbalance_method
            
        else:  # No Handling
            config['handle_imbalance'] = False
            config['use_class_weights'] = False
    
    st.session_state.preprocessing_config = config
    
    # Show summary
    st.subheader("Configuration Summary")
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
    st.markdown('<div class="section-header">Train Models</div>', unsafe_allow_html=True)
    
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
        rows_removed = summary['rows_removed']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", summary['final_shape'][0])
        with col2:
            st.metric("Features", summary['final_shape'][1])
        with col3:
            # Handle both removal and addition of rows
            if rows_removed > 0:
                st.metric("Rows Removed", rows_removed)
            elif rows_removed < 0:
                st.metric("Rows Added (Resampling)", abs(rows_removed))
            else:
                st.metric("Rows Changed", 0)
    
    processed_data = st.session_state.processed_data
    
    # Also show summary again if already processed (for returning to this step)
    if processed_data:
        st.info("ℹ️ Preprocessing already completed. Showing summary:")
        summary = processed_data['summary']
        rows_removed = summary['rows_removed']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", summary['final_shape'][0])
        with col2:
            st.metric("Features", summary['final_shape'][1])
        with col3:
            if rows_removed > 0:
                st.metric("Rows Removed", rows_removed)
            elif rows_removed < 0:
                st.metric("Rows Added (Resampling)", abs(rows_removed))
            else:
                st.metric("Rows Changed", 0)
    
    processed_data = st.session_state.processed_data
    
    # Train models
    if st.session_state.model_results is None:
        config = st.session_state.preprocessing_config
        class_weight_method = config.get('class_weight_method') if config.get('use_class_weights') else None
        
        st.info("🤖 Training all models... This may take a while.")
        
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Get model count for progress tracking
        model_configs = models.get_model_configs(class_weight=class_weight_method)
        total_models = len(model_configs)
        
        status_placeholder.text("Starting model training...")
        
        # Train models with progress updates
        model_results = {}
        for idx, (model_name, config) in enumerate(model_configs.items(), 1):
            status_placeholder.text(f"Training {model_name} ({idx}/{total_models})...")
            progress_bar.progress(idx / total_models)
            
            try:
                result = models.train_single_model(
                    config['model'],
                    processed_data['X_train'],
                    processed_data['y_train'],
                    processed_data['X_test'],
                    processed_data['y_test']
                )
                result['description'] = config['description']
                model_results[model_name] = result
            except Exception as e:
                model_results[model_name] = {
                    'trained': False,
                    'error': str(e),
                    'description': config['description']
                }
        
        status_placeholder.empty()
        progress_bar.empty()
        st.session_state.model_results = model_results
        
        st.success("✅ All models trained successfully!")
    
    model_results = st.session_state.model_results
    
    # Show training summary
    st.subheader("Training Summary")
    summary_table = models.create_model_summary_table(model_results)
    st.dataframe(summary_table, use_container_width=True)
    
    # Quick evaluation
    st.subheader("Quick Evaluation")
    with st.spinner("📊 Evaluating models..."):
        eval_results = evaluation.evaluate_all_models(
            model_results,
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        st.session_state.evaluation_results = eval_results
        st.session_state.initial_evaluation_results = eval_results  # NEW: Store initial metrics separately
        
        logger.info("STEP 6 - INITIAL TRAINING EVALUATION (stored separately):")
        for model_name, result in list(eval_results.items())[:2]:
            if 'test_metrics' in result:
                logger.info(f"  {model_name}: F1={result['test_metrics']['f1_score']:.4f}, Acc={result['test_metrics']['accuracy']:.4f}")
    
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
    st.markdown('<div class="section-header">Hyperparameter Optimization</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">🔧 Optimize model hyperparameters using cross-validation</div>', unsafe_allow_html=True)
    
    # Optimization settings
    col1, col2, col3 = st.columns(3)
    with col1:
        opt_method = st.radio(
            "Search Method", 
            ['grid', 'random'],
            format_func=lambda x: 'GridSearchCV' if x == 'grid' else 'RandomizedSearchCV',
            help="**GridSearchCV**: Tests all parameter combinations. Thorough but slow. Best for small parameter spaces.\n\n**RandomizedSearchCV**: Tests random parameter combinations. Faster and often finds good solutions. Best for large parameter spaces."
        )
    with col2:
        cv_folds = st.slider(
            "CV Folds", 
            3, 10, 5,
            help="Number of cross-validation splits. Higher = more reliable but slower. 5 is standard. Use 3 for small datasets, 10 for large datasets."
        )
    with col3:
        if opt_method == 'random':
            n_iter = st.slider(
                "Iterations", 
                10, 100, 50,
                help="Number of random parameter combinations to try. More = better chance of finding optimal parameters but takes longer. 50 is a good balance."
            )
        else:
            n_iter = 50
    
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        model_results = st.session_state.model_results
        processed_data = st.session_state.processed_data
        config = st.session_state.preprocessing_config
        class_weight_method = config.get('class_weight_method') if config.get('use_class_weights') else None
        model_configs = models.get_model_configs(class_weight=class_weight_method)
        
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
            
            logger.info("="*60)
            logger.info("OPTIMIZATION COMPLETED - Starting automatic retraining")
            
            # Automatically retrain with best parameters
            status_text.text("Retraining models with optimized parameters...")
            retrained_results = optimization.retrain_with_best_params(
                opt_results,
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            st.session_state.model_results = retrained_results
            
            logger.info("RETRAINED MODEL_RESULTS - Sample:")
            for model_name, result in list(retrained_results.items())[:2]:
                if result.get('trained'):
                    logger.info(f"  {model_name}: trained={result.get('trained')}, has_model={'model' in result}")
            
            # Re-evaluate with optimized models to update evaluation_results
            status_text.text("Re-evaluating optimized models...")
            eval_results = evaluation.evaluate_all_models(
                retrained_results,
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            st.session_state.evaluation_results = eval_results
            
            logger.info("UPDATED EVALUATION_RESULTS - Sample metrics:")
            for model_name, result in list(eval_results.items())[:2]:
                if 'test_metrics' in result:
                    logger.info(f"  {model_name}: F1={result['test_metrics']['f1_score']:.4f}, Acc={result['test_metrics']['accuracy']:.4f}")
            
            logger.info("="*60)
            
            status_text.empty()
        
        st.success("✅ Optimization completed and models retrained with best parameters!")
    
    # Show results
    if st.session_state.optimization_results:
        opt_results = st.session_state.optimization_results
        
        st.subheader("Optimization Results")
        summary_table = optimization.create_optimization_summary(opt_results)
        st.dataframe(summary_table, use_container_width=True)
        
        st.subheader("Best Parameters")
        best_params = optimization.get_best_params_summary(opt_results)
        for model_name, params in best_params.items():
            with st.expander(f"{model_name}"):
                st.json(params)
        
        # Models are automatically retrained with best parameters after optimization
        st.info("ℹ️ Models have been automatically retrained with optimized hyperparameters.")
    
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
    st.markdown('<div class="section-header">Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    
    logger.info("="*60)
    logger.info("STEP 8: EVALUATION START")
    logger.info(f"Optimization results exist: {st.session_state.optimization_results is not None}")
    logger.info(f"Evaluation results exist: {st.session_state.evaluation_results is not None}")
    
    eval_results = st.session_state.evaluation_results
    processed_data = st.session_state.processed_data
    
    # Log current evaluation metrics before re-evaluation
    if eval_results:
        logger.info("BEFORE RE-EVALUATION - Sample metrics:")
        for model_name, result in list(eval_results.items())[:2]:
            if 'test_metrics' in result:
                logger.info(f"  {model_name}: F1={result['test_metrics']['f1_score']:.4f}, Acc={result['test_metrics']['accuracy']:.4f}")
    
    # Re-evaluate if optimization was done
    if st.session_state.optimization_results:
        st.info("🔄 Optimization was done - re-evaluating models with optimized parameters...")
        logger.info("Re-evaluating models with optimized parameters...")
        
        with st.spinner("📊 Re-evaluating optimized models..."):
            eval_results = evaluation.evaluate_all_models(
                st.session_state.model_results,
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            st.session_state.evaluation_results = eval_results
            st.success("✅ Re-evaluation completed with optimized models")
        
        # Log updated evaluation metrics
        logger.info("AFTER RE-EVALUATION - Sample metrics:")
        for model_name, result in list(eval_results.items())[:2]:
            if 'test_metrics' in result:
                logger.info(f"  {model_name}: F1={result['test_metrics']['f1_score']:.4f}, Acc={result['test_metrics']['accuracy']:.4f}")
    else:
        st.info("ℹ️ Using evaluation results from initial training (no optimization)")
        logger.info("No optimization done - using initial evaluation results")
    
    logger.info("="*60)
    
    # Comparison table
    st.subheader("Performance Comparison")
    comparison_table = evaluation.create_comparison_table(eval_results)
    st.dataframe(comparison_table, use_container_width=True)
    
    # Metrics visualization
    st.subheader("Metrics Comparison")
    fig = evaluation.plot_metrics_comparison(eval_results)
    if fig:
        st.pyplot(fig)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    fig = evaluation.plot_all_confusion_matrices(eval_results)
    if fig:
        st.pyplot(fig)
    
    # ROC curves (if binary classification)
    y_test = processed_data['y_test']
    if len(np.unique(y_test)) == 2:
        st.subheader("ROC Curves")
        fig = evaluation.plot_all_roc_curves(eval_results, y_test)
        if fig:
            st.pyplot(fig)
    
    # Best models - show two winners
    st.subheader("Best Performing Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Best by F1-Score")
        best_f1_name, best_f1_result = evaluation.get_best_model(eval_results, metric='f1_score')
        
        if best_f1_name:
            st.success(f"**{best_f1_name}**")
            
            metrics = best_f1_result['test_metrics']
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with subcol2:
                st.metric("Recall", f"{metrics['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}", delta="Winner")
    
    with col2:
        st.markdown("### Best by Accuracy")
        best_acc_name, best_acc_result = evaluation.get_best_model(eval_results, metric='accuracy')
        
        if best_acc_name:
            st.success(f"**{best_acc_name}**")
            
            metrics = best_acc_result['test_metrics']
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}", delta="Winner")
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with subcol2:
                st.metric("Recall", f"{metrics['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    
    # Note if they're the same model
    if best_f1_name == best_acc_name:
        st.info(f"ℹ️ **{best_f1_name}** wins on both metrics!")
    else:
        st.warning("⚠️ Different models excel at different metrics. Choose based on your problem requirements.")
    
    # Download comparison
    st.subheader("Download Results")
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
    st.markdown('<div class="section-header">Generate Report</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">📄 Generate a comprehensive report with all findings</div>', unsafe_allow_html=True)
    
    logger.info("="*60)
    logger.info("STEP 9: REPORT GENERATION START")
    logger.info(f"Optimization results exist: {st.session_state.optimization_results is not None}")
    logger.info(f"Initial evaluation results exist: {st.session_state.initial_evaluation_results is not None}")
    logger.info(f"Current evaluation results exist: {st.session_state.evaluation_results is not None}")
    
    # Compare initial vs current evaluation results
    if st.session_state.initial_evaluation_results and st.session_state.evaluation_results:
        logger.info("COMPARING INITIAL vs CURRENT EVALUATION RESULTS:")
        for model_name in list(st.session_state.initial_evaluation_results.keys())[:2]:
            initial = st.session_state.initial_evaluation_results[model_name]
            current = st.session_state.evaluation_results[model_name]
            if 'test_metrics' in initial and 'test_metrics' in current:
                logger.info(f"  {model_name}:")
                logger.info(f"    INITIAL (default params): F1={initial['test_metrics']['f1_score']:.4f}, Acc={initial['test_metrics']['accuracy']:.4f}")
                logger.info(f"    CURRENT (optimized params): F1={current['test_metrics']['f1_score']:.4f}, Acc={current['test_metrics']['accuracy']:.4f}")
    
    # Log evaluation metrics being used for report
    if st.session_state.evaluation_results:
        logger.info("EVALUATION RESULTS USED IN REPORT:")
        for model_name, result in list(st.session_state.evaluation_results.items())[:3]:
            if 'test_metrics' in result:
                logger.info(f"  {model_name}: F1={result['test_metrics']['f1_score']:.4f}, "
                          f"Acc={result['test_metrics']['accuracy']:.4f}, "
                          f"Time={result.get('training_time', 'N/A')}")
    
    # Log optimization info
    if st.session_state.optimization_results:
        logger.info("OPTIMIZATION RESULTS:")
        for model_name, result in list(st.session_state.optimization_results.items())[:3]:
            if result.get('optimized'):
                logger.info(f"  {model_name}: CV_Score={result.get('best_score', 'N/A'):.4f}, "
                          f"Params={result.get('best_params', {})}")
    else:
        logger.warning("No optimization results found!")
    
    logger.info("="*60)
    
    # Collect all data
    dataset_info = utils.get_basic_stats(st.session_state.df)
    feature_types = utils.get_feature_types(st.session_state.df)
    eda_summary = eda.compute_eda_summary(
        st.session_state.df,
        feature_types['numeric'],
        feature_types['categorical']
    )
    
    # Get both best models
    best_f1_name, _ = evaluation.get_best_model(
        st.session_state.evaluation_results,
        metric='f1_score'
    )
    best_acc_name, _ = evaluation.get_best_model(
        st.session_state.evaluation_results,
        metric='accuracy'
    )
    
    # Generate report
    with st.spinner("📝 Generating report..."):
        markdown_report = report_generator.generate_markdown_report(
            dataset_info=dataset_info,
            eda_summary=eda_summary,
            diagnostics=st.session_state.diagnostics,
            preprocessing_summary=st.session_state.processed_data['summary'],
            initial_evaluation_results=st.session_state.initial_evaluation_results,  # NEW: Pass initial results
            evaluation_results=st.session_state.evaluation_results,
            optimization_results=st.session_state.optimization_results,
            best_model_name=best_f1_name,  # For backward compatibility
            best_f1_model=best_f1_name,  # NEW: Best by F1
            best_acc_model=best_acc_name  # NEW: Best by Accuracy
        )
    
    st.success("✅ Report generated successfully!")
    
    # Preview
    with st.expander("Preview Report", expanded=True):
        st.markdown(markdown_report)
    
    # Download options
    st.subheader("Download Report")
    
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
