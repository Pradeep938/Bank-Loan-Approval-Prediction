Bank Loan Approval Prediction Using Machine Learning

Week 4 – Major Project (Capstone Project)

Project Overview

This project builds a complete machine learning pipeline to predict whether a bank loan application will be approved or rejected.

The project uses the Loan Prediction Dataset (Loan Prediction.csv) and compares Logistic Regression, Decision Tree, Random Forest, and XGBoost classification models.

Objectives

Clean and explore the loan dataset

Handle missing values

Encode categorical variables

Scale numerical features

Train multiple classification models

Compare model performance

Evaluate accuracy, precision, recall, F1-score and ROC-AUC

Create a confusion matrix and ROC curve

Analyze important features

Save the trained model for future predictions

Dataset

The dataset contains 614 loan applications and 13 original columns.

Important fields include:

Gender

Married

Dependents

Education

Self_Employed

ApplicantIncome

CoapplicantIncome

LoanAmount

Loan_Amount_Term

Credit_History

Property_Area

Loan_Status

Loan_ID is treated as an identifier and removed before modeling.

Machine Learning Workflow

Loan Dataset
     ↓
Data Understanding
     ↓
EDA & Visualization
     ↓
Missing Value Treatment
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ├── Logistic Regression
     ├── Decision Tree
     ├── Random Forest
     └── XGBoost
     ↓
Model Evaluation
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1-Score
     └── ROC-AUC
     ↓
Confusion Matrix + ROC Curve
     ↓
Best Model
     ↓
Saved Model

Project Structure

Bank-Loan-Approval-Prediction/
│
├── data/
│   └── Loan Prediction.csv
│
├── notebooks/
│   └── bank_loan_approval_prediction.ipynb
│
├── models/
│   ├── loan_approval_model.pkl
│   └── model_comparison.csv
│
├── visualizations/
│   ├── loan_status_distribution.png
│   ├── numerical_distributions.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve_comparison.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt

How to Run

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd Bank-Loan-Approval-Prediction

2. Install dependencies

pip install -r requirements.txt

3. Start Jupyter Notebook

jupyter notebook

Open:

notebooks/bank_loan_approval_prediction.ipynb

Run all cells from top to bottom.

Evaluation Metrics

The project evaluates:

Accuracy: Overall percentage of correct predictions.

Precision: Among predicted approvals, how many were actually approved.

Recall: Among actual approvals, how many were correctly identified.

F1-score: Harmonic mean of precision and recall.

ROC-AUC: Measures the model's ability to distinguish approved and rejected applications across classification thresholds.

Technologies

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Joblib

Jupyter Notebook

Academic Note

This project is intended for educational and portfolio purposes. Real-world lending decisions require additional financial, regulatory, fairness, and risk-management considerations.
