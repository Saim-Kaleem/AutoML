# 🚀 Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 📝 Sample Workflow

### Using the Iris Dataset Example

1. **Prepare a test dataset** (or download Iris dataset)
   ```python
   from sklearn.datasets import load_iris
   import pandas as pd
   
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['target'] = iris.target
   df.to_csv('iris_dataset.csv', index=False)
   ```

2. **Upload the CSV** in the application

3. **Review EDA** - See distributions, correlations, and missing values

4. **Check Issues** - The system will flag any data quality problems

5. **Select Target** - Choose 'target' as the prediction variable

6. **Configure Preprocessing**:
   - Train-test split: 20%
   - Encoding: One-Hot
   - Scaling: StandardScaler
   - No missing values in Iris, so skip imputation

7. **Train Models** - All 7 models will train automatically

8. **Optimize** (optional):
   - Method: GridSearchCV
   - CV Folds: 5

9. **Evaluate** - Compare all models and see which performs best

10. **Generate Report** - Download comprehensive analysis

---

## 🎯 Expected Results

For the Iris dataset, you should see:
- **Random Forest**: ~95-97% accuracy
- **SVM**: ~94-96% accuracy  
- **Logistic Regression**: ~92-95% accuracy

---

## ⚠️ Important Notes

### Data Requirements
- ✅ CSV format
- ✅ At least 10 rows
- ✅ At least 2 columns (features + target)
- ✅ Maximum 100 MB

### Preprocessing Best Practices
- Always use **stratified split** for classification
- Use **StandardScaler** for most algorithms (especially SVM, Logistic Regression)
- Apply **SMOTE** only when class imbalance is severe (ratio > 3:1)
- **One-Hot encoding** is preferred for tree-based models

### Performance Tips
- **Large datasets (>50K rows)**: Use RandomizedSearchCV instead of GridSearchCV
- **Many features (>100)**: Consider feature selection first
- **Imbalanced data**: Try different resampling methods

---

## 🔧 Troubleshooting

### Application won't start
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Out of memory errors
- Reduce dataset size
- Use smaller test_size (e.g., 0.1 instead of 0.2)
- Skip hyperparameter optimization

### Slow training
- Use fewer hyperparameter combinations
- Switch to RandomizedSearchCV
- Reduce n_iter parameter

---

## 📊 Understanding Results

### Metrics Explained

**Accuracy**: Overall correctness (best for balanced datasets)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: How many predicted positives are actually positive
```
Precision = TP / (TP + FP)
```

**Recall**: How many actual positives were identified
```
Recall = TP / (TP + FN)
```

**F1-Score**: Harmonic mean of precision and recall (best for imbalanced data)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### When to Use Each Metric

- **Balanced dataset**: Use **Accuracy**
- **Imbalanced dataset**: Use **F1-Score** or **Recall**
- **Cost of false positives high**: Use **Precision**
- **Cost of false negatives high**: Use **Recall**

---

## 🎓 Educational Resources

### Understanding the Models

**Logistic Regression**
- Best for: Linear relationships, binary classification
- Fast training, interpretable coefficients

**K-Nearest Neighbors**
- Best for: Small to medium datasets
- No training phase, sensitive to scaling

**Decision Tree**
- Best for: Non-linear patterns, mixed data types
- Interpretable rules, prone to overfitting

**Naive Bayes**
- Best for: Text classification, quick baseline
- Assumes feature independence

**Random Forest**
- Best for: Most general-purpose tasks
- Handles non-linearity, reduces overfitting

**SVM**
- Best for: High-dimensional data, clear margins
- Effective with kernel trick

**Rule-Based**
- Baseline comparison (majority class predictor)

---

## 💡 Pro Tips

1. **Always start with EDA** - Understand your data before modeling

2. **Check class balance** - Severe imbalance needs special handling

3. **Feature scaling matters** - Especially for distance-based algorithms

4. **Cross-validation is key** - Don't trust single train-test split

5. **Start simple** - Logistic Regression often performs surprisingly well

6. **Monitor training time** - Complex models need more compute

7. **Validate on new data** - Test set performance is what matters

8. **Document decisions** - The generated report helps with reproducibility

---

## 📞 Need Help?

- 📖 Read the full [README.md](README.md)
- 🐛 Check [GitHub Issues](https://github.com/yourusername/automl-classification/issues)
- 💬 Ask questions in Discussions

---

**Happy AutoML-ing! 🎉**
