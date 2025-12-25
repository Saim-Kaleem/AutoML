# 🤖 AutoML Classification System

**End-to-end Automated Machine Learning for Supervised Classification**

A comprehensive Streamlit web application that automates the entire machine learning workflow for classification tasks, from data exploration to model deployment.

---

## Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Models Supported](#-models-supported)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## Features

### **Automated Data Analysis**
- Comprehensive exploratory data analysis (EDA)
- Automatic feature type detection (numeric/categorical)
- Manual datatype conversion (int, float, string, category)
- Statistical summaries and visualizations
- Correlation analysis with custom color-themed heatmaps
- Missing value detection and visualization

### **Intelligent Issue Detection**
- Missing value identification (global and per-feature)
- **Configurable outlier detection** (IQR or Z-score methods with user selection)
- **High-cardinality detection** (configurable threshold, default 20)
- **Automatic exclusion** of high-cardinality columns from encoding
- Near-constant feature detection (>95% dominance)
- Class imbalance analysis with dynamic thresholds
- Duplicate row detection
- **Target variable validation** (automatic categorical conversion, cardinality checks)

### **User-Controlled Preprocessing**
- **Manual Feature Selection:** Keep all, select to keep, or select to drop
- **Constant Features:** Automatic removal of truly constant features
- **Near-Constant Features:** Optional removal with user control
- **Missing Values:** Multiple imputation strategies (mean, median, mode, constant)
- **Outliers:** Remove or keep based on user decision
- **Encoding:** One-Hot or Ordinal encoding with high-cardinality exclusion
- **Scaling:** StandardScaler or MinMaxScaler
- **Train-Test Split:** Configurable with stratification
- **Class Imbalance:** 
  - **Class Weights** (balanced, balanced_subsample) for 3 models
  - **Resampling** (SMOTE, ADASYN, over/undersampling)
- **Comprehensive Help Text:** Tooltips explaining every technical term and choice

### **Multiple Classification Models**
1. Logistic Regression (with class weights support)
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Naive Bayes (Gaussian)
5. Random Forest (with class weights support)
6. Support Vector Machine (SVM) (with class weights support)
7. Rule-Based Classifier (baseline)

**Note:** Rule-Based Classifier is not optimized (no hyperparameters to tune)

### **Hyperparameter Optimization**
- GridSearchCV for exhaustive search
- RandomizedSearchCV for faster optimization
- Cross-validation support (configurable folds)
- Automatic best parameter selection

### **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for all models (color-themed)
- ROC-AUC curves for binary classification (color-themed)
- Side-by-side model comparison with visualizations
- Dual Winner Announcement: Best model by F1-score AND best by accuracy
- Training time tracking
- Detailed classification reports
- Separate tracking of initial (default params) vs optimized performance

### **Auto-Generated Reports**
- **Markdown** format
- **HTML** format with styling
- **PDF** format (comprehensive)
- Includes all findings, decisions, and results
- Shows both best models (by F1-score and accuracy)
- Dual performance sections: Initial results vs optimized results
- Downloadable and shareable
- Terminal logging for debugging and transparency

---

## Technology Stack

### Core Technologies
- **Python 3.9+**
- **Streamlit** - Web UI framework
- **scikit-learn** - Machine learning algorithms
- **pandas** & **numpy** - Data manipulation
- **matplotlib** & **seaborn** - Visualizations
- **imbalanced-learn** - Handling class imbalance
- **reportlab** - PDF generation

### Deployment
- **Streamlit Cloud** compatible
- No database required
- In-memory data processing
- Fully stateless architecture

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/automl-classification.git
cd automl-classification
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## Usage

### Step-by-Step Workflow

#### **1, Upload Dataset**
- Upload a CSV file (max 100 MB)
- Must have at least 2 columns and 10 rows
- View data preview, types, and basic statistics
- Optional: Manually convert column datatypes (int, float, string, category)

#### **2. Exploratory Data Analysis**
- Automatic generation of:
  - Histograms for numeric features
  - Bar plots for categorical features
  - Boxplots for outlier visualization
  - Correlation heatmap
  - Missing value summary

#### **3. Issue Detection**
- Configure detection methods:
  - Choose outlier method (IQR or Z-score)
  - Set high-cardinality threshold (default: 20, range: 5-100)
- Review detected issues:
  - Missing values per column
  - Outliers in numeric features
  - High-cardinality categorical columns
  - Near-constant features
  - Duplicate rows

#### **4. Select Target Variable**
- Choose the column to predict
- Automatic validation: Ensures target is categorical with ≤20 unique classes
- Automatic conversion: Numeric targets converted to categorical if valid
- View target distribution
- Analyze class balance with dynamic thresholds

#### **5. Configure Preprocessing**
Make decisions for:
- Train-test split ratio with stratification option
- Manual feature selection (keep all/select to keep/select to drop)
- Constant features: Automatic removal of truly constant, optional removal of near-constant
- Missing value imputation strategies (with help tooltips)
- Outlier handling (keep/remove)
- Categorical encoding method (with high-cardinality exclusion option)
- Feature scaling method
- Class imbalance handling:
  - No handling
  - Class weights (balanced/balanced_subsample)
  - Resampling (SMOTE/ADASYN/over/undersample)
- **All options include helpful tooltips explaining technical terms**

#### **6. Train Models**
- Automatic preprocessing execution
- Parallel training of all 7 models
- Quick performance preview

#### **7. Optimize Hyperparameters** (Optional)
- Choose GridSearchCV or RandomizedSearchCV
- Configure cross-validation folds
- Optimize all models automatically
- Retrain with best parameters

#### **8. Evaluate & Compare**
- View comprehensive metrics table
- Compare models with color-themed bar charts
- Analyze confusion matrices (custom color scheme)
- View ROC curves for binary classification (app-themed colors)
- Identify TWO best models: One by F1-score, one by accuracy
- Side-by-side display of both winners with all metrics
- Download comparison CSV

#### **9. Generate Report**
- Auto-generate comprehensive report
- Preview in browser
- Download as Markdown, HTML, or PDF

---

## Project Structure

```
automl_app/
│
├── app.py                     # Streamlit UI + workflow controller
├── eda.py                     # EDA computations & visualizations
├── issue_detection.py         # Data quality issue detection
├── preprocessing.py           # Preprocessing pipeline
├── models.py                  # Model definitions & training
├── optimization.py            # Hyperparameter optimization
├── evaluation.py              # Model evaluation & metrics
├── report_generator.py        # Report generation (PDF/HTML/MD)
├── utils.py                   # Shared utility functions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Workflow

```
┌─────────────────┐
│  Upload Dataset │
└────────┬────────┘
         ↓
┌─────────────────┐
│   EDA Analysis  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Issue Detection │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Target Selection│
└────────┬────────┘
         ↓
┌─────────────────┐
│  Preprocessing  │
│   (fit on train)│
└────────┬────────┘
         ↓
┌─────────────────┐
│  Train Models   │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Optimize HP   │
│   (optional)    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Evaluate All   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Generate Report │
└─────────────────┘
```

---

## Models Supported

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| **Logistic Regression** | Linear | C, solver, penalty |
| **K-Nearest Neighbors** | Instance-based | n_neighbors, weights, metric |
| **Decision Tree** | Tree-based | max_depth, min_samples_split, criterion |
| **Naive Bayes** | Probabilistic | var_smoothing |
| **Random Forest** | Ensemble | n_estimators, max_depth, max_features |
| **SVM** | Kernel-based | C, kernel, gamma |
| **Rule-Based** | Baseline | (majority class predictor) |

---

## Deployment

### Deployed to Streamlit Cloud

Visit [this link](https://share.streamlit.io)

### Environment Variables

No environment variables required. All processing is done in-memory.

### Resource Requirements

- **Memory:** 2-4 GB recommended
- **CPU:** Multi-core beneficial for parallel training
- **Storage:** No persistent storage needed

---

## Example Datasets

The system works with any classification dataset. Try these:

### Built-in Dataset Ideas
- Iris (multiclass classification)
- Titanic (binary classification)
- Wine Quality (multiclass)
- Credit Card Fraud (imbalanced binary)
- Customer Churn (binary)

### Dataset Requirements
- Format: CSV
- Size: < 100 MB
- Rows: ≥ 10
- Columns: ≥ 2 (features + target)
- Target: Can be numeric or categorical

---

## Key Design Principles

1. **User-Controlled Automation**: AI suggests, user approves
2. **Transparency**: All decisions and results are visible with logging
3. **Reproducibility**: Same inputs → same outputs
4. **Educational**: Clear explanations via help tooltips for every technical term
5. **Production-Ready**: Follows ML best practices
6. **Accessibility**: Beginner-friendly with expert-level options
7. **Visual Consistency**: Color-themed charts matching app UI 

---

## Data Privacy

- All processing happens **in-browser** or on Streamlit Cloud
- No data is stored to disk
- No external API calls
- Session-based state management
- Data cleared on browser refresh

---

## Troubleshooting

### Common Issues

**"Module not found" error**
```bash
pip install -r requirements.txt
```

**"Dataset too large" error**
- Reduce dataset size to < 100 MB
- Remove unnecessary columns

**"Model training taking too long"**
- Use RandomizedSearchCV instead of GridSearchCV
- Reduce n_iter parameter
- Use smaller dataset

**"PDF generation failed"**
- Markdown and HTML reports are always available
- Install reportlab: `pip install reportlab`

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## Authors

**Saim Kaleem**
- GitHub: [Saim-Kaleem](https://github.com/Saim-Kaleem)

**Muhammad Ahad Waqas**
- Github: [Ahad-Waqas](https://github.com/Ahad-Waqas)

---

## Acknowledgments

- scikit-learn team for excellent ML library
- Streamlit for making ML apps accessible
- NUST for the project opportunity

---

## Future Enhancements

- [ ] Support for regression tasks
- [ ] Ensemble model creation
- [ ] Feature engineering automation
- [ ] Time series support
- [ ] Model interpretability (SHAP, LIME)
- [ ] A/B testing framework
- [ ] Model versioning
- [ ] API endpoint generation

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Contact the author

---

**Built with ❤️ for automated machine learning**

*This is a semester project demonstrating end-to-end ML automation without using AutoML libraries.*
