# Titanic-Supervised-Learning-02450

Supervised learning analysis of the Titanic dataset for the DTU course **02450 – Introduction to Machine Learning and Data Mining**.  
This project builds upon the exploratory work from Report 1 and applies regression and classification models, including hyperparameter tuning and statistical comparison using two-level cross-validation.  
All experiments are implemented in Python through both scripts and Jupyter notebooks.

---

## Project Context

**Course:** 02450 – Introduction to Machine Learning and Data Mining  
**Institution:** Technical University of Denmark (DTU)  
**Academic Period:** 2024  
**Deliverables:** Report (PDF) + source code + notebooks  

This project analyzes the Titanic dataset using supervised learning methods.  
The dataset contains demographic, socio-economic, and travel information for passengers aboard the RMS Titanic.

The dataset used in this project is **titanic.csv**, obtained from Kaggle.  
**Note:** The dataset is *not included* in this repository due to course rules and file size considerations. It must be placed locally in the `data/` folder.

---

## Project Structure

The project is divided into two main sections:

### **1. Regression**
- Predicts fare price (`Fare`) using demographic and travel attributes.
- Methods applied:
  - Linear regression (with standardization)
  - Ridge regression with regularization parameter λ
  - Two-level cross-validation for model selection
  - Artificial Neural Network (ANN)
  - Statistical comparison of ANN vs. ridge regression vs. baseline
- Key topics:
  - Regularization
  - Bias–variance tradeoff
  - Nested cross-validation
  - Interpretation of regression coefficients

### **2. Classification**
- Predicts passenger survival (`Survived`) as a binary classification task.
- Models evaluated:
  - Logistic regression (λ-tuned)
  - Artificial Neural Network (ANN)
  - Decision Tree
  - k-Nearest Neighbors (k-NN)
  - Naive Bayes
  - Baseline classifier
- Evaluation:
  - Two-level cross-validation for fair comparison
  - Error rate & ROC-AUC
  - McNemar’s test for statistical significance
- Key topics:
  - Regularized logistic regression
  - Decision boundaries
  - Nonlinear models (ANN)
  - Model interpretability
