# Automated Supervised Machine Learning App

## Overview
This Streamlit application provides an end-to-end machine learning workflow that allows users to upload, preprocess, visualize, and train machine learning models with minimal coding required.

## Features

### Data Upload
- Supports CSV and Excel file formats
- Easy file upload through sidebar

### Data Preprocessing
- Drop unwanted columns
- Handle missing values using:
  - Mean imputation
  - Median imputation
  - Mode imputation
- Categorical data encoding:
  - Label Encoding
  - One-Hot Encoding
  - Combined Label and One-Hot Encoding

### Data Visualization
Two-stage visualization with interactive plots:
- Scatter Plot
- Histogram Plot
- Box Plot
- Bar Plot
- Pie Plot

### Machine Learning Model Training
Automatic model selection for:
- Classification tasks
- Regression tasks

### Key Capabilities
- Automatic model comparison
- Model performance reports
- Various model visualization options
- Supports both numerical and categorical target variables

## Prerequisites

### Python Libraries
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- pycaret

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AhmedMOM3/automated-supervised-ml.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run automated_supervised_ml.py
```

## Usage Steps
1. Upload your dataset (CSV or Excel)
2. Preprocess data by:
   - Dropping columns
   - Handling missing values
   - Encoding categorical variables
3. Explore data visualizations
4. Select target variable
5. Train models and compare performance

## Model Visualization Options
### Classification
- Area Under the Curve
- Discrimination Threshold
- Precision Recall Curve
- Class Prediction Error
- Classification Report
- Decision Boundary
- Learning Curve
- Manifold Learning
- Calibration Curve
- Validation Curve
- Dimension Learning

### Regression
- Residuals Plot
- Prediction Error Plot
- Cooks Distance Plot
- Recursive Feature Selection
- Learning Curve
- Validation Curve
- Manifold Learning
- Decision Tree

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request.

## License
[Specify the license here]

## Author
[Ahmed MOM3](https://github.com/AhmedMOM3)

## Acknowledgments
- Streamlit
- PyCaret
- Scikit-learn
- Plotly
