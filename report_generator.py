import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import io


def generate_markdown_report(dataset_info: Dict[str, Any],
                            eda_summary: Dict[str, Any],
                            diagnostics: Dict[str, Any],
                            preprocessing_summary: Dict[str, Any],
                            evaluation_results: Dict[str, Dict[str, Any]],
                            optimization_results: Optional[Dict[str, Dict[str, Any]]] = None,
                            best_model_name: str = None) -> str:
    """
    Generate comprehensive Markdown report.
    
    Args:
        dataset_info: Basic dataset information
        eda_summary: EDA findings
        diagnostics: Issue detection results
        preprocessing_summary: Preprocessing decisions and results
        evaluation_results: Model evaluation results
        optimization_results: Hyperparameter optimization results (optional)
        best_model_name: Name of the best performing model
        
    Returns:
        Markdown report as string
    """
    report = []
    
    # Header
    report.append("# AutoML Classification Report")
    report.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # Dataset Overview
    report.append("## 1. Dataset Overview\n")
    report.append(f"- **Number of Rows:** {dataset_info.get('n_rows', 'N/A')}")
    report.append(f"- **Number of Columns:** {dataset_info.get('n_columns', 'N/A')}")
    report.append(f"- **Numeric Features:** {dataset_info.get('n_numeric', 'N/A')}")
    report.append(f"- **Categorical Features:** {dataset_info.get('n_categorical', 'N/A')}")
    report.append(f"- **Memory Usage:** {dataset_info.get('memory_mb', 0):.2f} MB")
    report.append(f"- **Duplicate Rows:** {dataset_info.get('duplicate_rows', 0)}")
    report.append(f"- **Total Missing Values:** {dataset_info.get('total_missing', 0)} ({dataset_info.get('missing_percentage', 0):.2f}%)\n")
    
    # EDA Findings
    report.append("## 2. Exploratory Data Analysis\n")
    
    report.append("### 2.1 Numeric Features Summary\n")
    if eda_summary.get('numeric_summary'):
        report.append("Key statistics for numeric features have been computed.\n")
    else:
        report.append("No numeric features found.\n")
    
    report.append("### 2.2 Categorical Features Summary\n")
    if eda_summary.get('categorical_summary'):
        for col, stats in eda_summary['categorical_summary'].items():
            report.append(f"- **{col}:** {stats['unique_values']} unique values, "
                        f"most frequent = '{stats['top_value']}' ({stats['top_value_freq']} occurrences)")
        report.append("")
    else:
        report.append("No categorical features found.\n")
    
    report.append("### 2.3 Feature Skewness\n")
    if eda_summary.get('skewness'):
        report.append("| Feature | Skewness |")
        report.append("|---------|----------|")
        for col, skew in list(eda_summary['skewness'].items())[:10]:
            report.append(f"| {col} | {skew:.4f} |")
        report.append("")
    
    # Detected Issues
    report.append("## 3. Data Quality Issues Detected\n")
    
    # Missing values
    if diagnostics['missing_values']['has_missing']:
        report.append("### 3.1 Missing Values\n")
        report.append(f"- **Total Missing:** {diagnostics['missing_values']['total_missing']} "
                     f"({diagnostics['missing_values']['global_percentage']:.2f}%)")
        report.append("\n**Columns with Missing Values:**\n")
        for col, info in diagnostics['missing_values']['columns_with_missing'].items():
            report.append(f"- {col}: {info['count']} ({info['percentage']:.2f}%)")
        report.append("")
    else:
        report.append("### 3.1 Missing Values\n\nNo missing values detected.\n")
    
    # Outliers
    if diagnostics['outliers']['columns_with_outliers']:
        report.append("### 3.2 Outliers\n")
        report.append(f"- **Total Outlier Instances:** {diagnostics['outliers']['total_outlier_instances']}")
        report.append("\n**Columns with Outliers:**\n")
        for col in diagnostics['outliers']['columns_with_outliers'][:10]:
            info = diagnostics['outliers']['outliers_per_column'][col]
            report.append(f"- {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        report.append("")
    else:
        report.append("### 3.2 Outliers\n\nNo significant outliers detected.\n")
    
    # High cardinality
    if diagnostics['high_cardinality']:
        report.append("### 3.3 High Cardinality Features\n")
        for item in diagnostics['high_cardinality']:
            report.append(f"- {item['column']}: {item['unique_values']} unique values "
                        f"({item['percentage_unique']:.2f}%)")
        report.append("")
    else:
        report.append("### 3.3 High Cardinality Features\n\nNo high cardinality features detected.\n")
    
    # Class imbalance
    if diagnostics.get('class_imbalance'):
        report.append("### 3.4 Class Imbalance\n")
        imbalance = diagnostics['class_imbalance']
        report.append(f"- **Number of Classes:** {imbalance['n_classes']}")
        report.append(f"- **Imbalanced:** {'Yes' if imbalance['is_imbalanced'] else 'No'}")
        report.append(f"- **Majority Class:** {imbalance['majority_class']} ({imbalance['majority_percentage']:.2f}%)")
        report.append(f"- **Minority Class:** {imbalance['minority_class']} ({imbalance['minority_percentage']:.2f}%)")
        report.append(f"- **Imbalance Ratio:** {imbalance['imbalance_ratio']:.2f}\n")
    
    # Preprocessing
    report.append("## 4. Preprocessing Pipeline\n")
    
    report.append("### 4.1 Data Split\n")
    orig_shape = preprocessing_summary.get('original_shape', (0, 0))
    final_shape = preprocessing_summary.get('final_shape', (0, 0))
    report.append(f"- **Original Training Samples:** {orig_shape[0]}")
    report.append(f"- **Original Features:** {orig_shape[1]}")
    report.append(f"- **Final Training Samples:** {final_shape[0]}")
    report.append(f"- **Final Features:** {final_shape[1]}")
    report.append(f"- **Rows Removed:** {preprocessing_summary.get('rows_removed', 0)}")
    report.append(f"- **Features Created:** {preprocessing_summary.get('features_created', 0)}\n")
    
    config = preprocessing_summary.get('preprocessing_config', {})
    
    if config.get('missing_values'):
        report.append("### 4.2 Missing Value Imputation\n")
        mv_config = config['missing_values']
        report.append(f"- **Numeric Strategy:** {mv_config.get('numeric_strategy', 'N/A')}")
        report.append(f"- **Categorical Strategy:** {mv_config.get('categorical_strategy', 'N/A')}\n")
    
    if config.get('encoding'):
        report.append("### 4.3 Categorical Encoding\n")
        enc_config = config['encoding']
        report.append(f"- **Encoding Type:** {enc_config.get('type', 'N/A')}")
        report.append(f"- **Features After Encoding:** {enc_config.get('n_features_after', 'N/A')}\n")
    
    if config.get('scaling'):
        report.append("### 4.4 Feature Scaling\n")
        scale_config = config['scaling']
        report.append(f"- **Scaling Method:** {scale_config.get('type', 'N/A')}\n")
    
    # Model Training and Evaluation
    report.append("## 5. Model Training and Evaluation\n")
    
    report.append("### 5.1 Models Trained\n")
    report.append("The following models were trained:\n")
    for i, model_name in enumerate(evaluation_results.keys(), 1):
        report.append(f"{i}. {model_name}")
    report.append("")
    
    report.append("### 5.2 Performance Comparison\n")
    report.append("| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |")
    report.append("|-------|----------|-----------|--------|----------|-------------------|")
    
    for model_name, result in evaluation_results.items():
        if 'test_metrics' in result:
            metrics = result['test_metrics']
            report.append(f"| {model_name} | {metrics['accuracy']:.4f} | "
                        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                        f"{metrics['f1_score']:.4f} | {result['training_time']:.4f} |")
    report.append("")
    
    # Hyperparameter Optimization
    if optimization_results:
        report.append("## 6. Hyperparameter Optimization\n")
        
        report.append("### 6.1 Optimization Summary\n")
        report.append("| Model | Method | Best CV Score | Combinations Tested | Time (s) |")
        report.append("|-------|--------|---------------|---------------------|----------|")
        
        for model_name, result in optimization_results.items():
            if result.get('optimized', False):
                report.append(f"| {model_name} | {result['method']} | "
                            f"{result['best_score']:.4f} | {result['n_combinations']} | "
                            f"{result['optimization_time']:.2f} |")
        report.append("")
        
        report.append("### 6.2 Best Parameters\n")
        for model_name, result in optimization_results.items():
            if result.get('optimized', False):
                report.append(f"\n**{model_name}:**")
                for param, value in result['best_params'].items():
                    report.append(f"- {param}: {value}")
        report.append("")
    
    # Best Model
    if best_model_name:
        report.append("## 7. Best Model\n")
        report.append(f"**Model:** {best_model_name}\n")
        
        if best_model_name in evaluation_results:
            best_result = evaluation_results[best_model_name]
            if 'test_metrics' in best_result:
                metrics = best_result['test_metrics']
                report.append("**Test Set Performance:**")
                report.append(f"- Accuracy: {metrics['accuracy']:.4f}")
                report.append(f"- Precision: {metrics['precision']:.4f}")
                report.append(f"- Recall: {metrics['recall']:.4f}")
                report.append(f"- F1-Score: {metrics['f1_score']:.4f}")
                report.append(f"- Training Time: {best_result['training_time']:.4f}s\n")
        
        report.append("**Justification:**")
        report.append(f"The {best_model_name} achieved the best F1-score on the test set, "
                     "indicating the best balance between precision and recall. "
                     "This model is recommended for deployment.\n")
    
    # Conclusion
    report.append("## 8. Conclusion\n")
    report.append("This automated machine learning pipeline successfully:")
    report.append("- Analyzed the dataset and detected data quality issues")
    report.append("- Applied appropriate preprocessing techniques")
    report.append("- Trained and evaluated multiple classification models")
    if optimization_results:
        report.append("- Optimized hyperparameters for improved performance")
    report.append(f"- Identified {best_model_name} as the best performing model\n")
    
    report.append("**Recommendations:**")
    report.append("- Consider feature engineering to potentially improve performance")
    report.append("- Monitor model performance on new data")
    report.append("- Retrain periodically with updated data")
    report.append("- Validate results with domain experts\n")
    
    report.append("---\n")
    report.append("*Report generated by AutoML Classification System*")
    
    return "\n".join(report)


def generate_html_report(markdown_content: str) -> str:
    """
    Convert Markdown report to HTML.
    
    Args:
        markdown_content: Markdown report content
        
    Returns:
        HTML report as string
    """
    # Simple markdown to HTML conversion
    html_lines = []
    
    html_lines.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Classification Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        ul {
            line-height: 1.8;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        .highlight {
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }
    </style>
</head>
<body>
    <div class="container">
""")
    
    # Convert markdown to HTML (basic conversion)
    lines = markdown_content.split('\n')
    in_table = False
    
    for line in lines:
        # Headers
        if line.startswith('# '):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith('## '):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith('### '):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        # Tables
        elif line.startswith('|'):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            if '---' in line:
                continue
            elif in_table and cells:
                if html_lines[-1] == "<table>":
                    html_lines.append("<thead><tr>")
                    for cell in cells:
                        html_lines.append(f"<th>{cell}</th>")
                    html_lines.append("</tr></thead><tbody>")
                else:
                    html_lines.append("<tr>")
                    for cell in cells:
                        html_lines.append(f"<td>{cell}</td>")
                    html_lines.append("</tr>")
        else:
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False
            
            # Horizontal rule
            if line.strip() == '---':
                html_lines.append("<hr>")
            # Lists
            elif line.startswith('- '):
                if not html_lines[-1].startswith('<ul>'):
                    html_lines.append("<ul>")
                html_lines.append(f"<li>{line[2:]}</li>")
            elif html_lines[-1].startswith('<li>') and not line.startswith('- '):
                html_lines.append("</ul>")
                html_lines.append(f"<p>{line}</p>")
            # Bold
            elif '**' in line:
                line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
                html_lines.append(f"<p>{line}</p>")
            # Regular paragraph
            elif line.strip():
                html_lines.append(f"<p>{line}</p>")
            else:
                html_lines.append("<br>")
    
    if in_table:
        html_lines.append("</tbody></table>")
    
    html_lines.append("""
    </div>
</body>
</html>
""")
    
    return "\n".join(html_lines)


def generate_pdf_report_content(markdown_content: str) -> bytes:
    """
    Generate PDF report from markdown content using reportlab.
    
    Args:
        markdown_content: Markdown report content
        
    Returns:
        PDF content as bytes
    """
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12
    )
    
    # Parse markdown and build PDF
    lines = markdown_content.split('\n')
    
    for line in lines:
        if line.startswith('# '):
            story.append(Paragraph(line[2:], title_style))
        elif line.startswith('## '):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(line[3:], heading_style))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading3']))
        elif line.startswith('- '):
            story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
        elif line.strip() and not line.startswith('|') and not line.strip() == '---':
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content


def create_downloadable_report(report_type: str, content: str) -> bytes:
    """
    Create downloadable report in specified format.
    
    Args:
        report_type: Type of report ('markdown', 'html', 'pdf')
        content: Report content
        
    Returns:
        Report as bytes
    """
    if report_type == 'markdown':
        return content.encode('utf-8')
    elif report_type == 'html':
        html_content = generate_html_report(content)
        return html_content.encode('utf-8')
    elif report_type == 'pdf':
        return generate_pdf_report_content(content)
    else:
        raise ValueError(f"Unsupported report type: {report_type}")
