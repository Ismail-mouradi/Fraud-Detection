# 📌 Credit Card Fraud Detection  
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()  
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)]()  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()  

An end-to-end machine learning project for detecting fraudulent credit card transactions.  
The goal is to build a complete fraud detection workflow including:

- Data preprocessing  
- Feature engineering  
- Handling severe class imbalance (SMOTE)  
- Model training (Logistic Regression, Random Forest, XGBoost)  
- Threshold tuning  
- Model comparison using fraud-focused metrics (PR-AUC, recall, F1)

---

## 📂 Repository Structure

```text
Fraud-Detection/
│
├── notebooks/
│   └── 01_credit_card_fraud.ipynb
│
├── src/
│   └── .keep
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 📊 Dataset

**Source:** Kaggle — Credit Card Fraud Detection  
**Rows:** 284,807  
**Fraud cases:** 492 (0.17%) → extremely imbalanced  

Features:
- `V1`…`V28` → anonymized PCA components  
- `Amount` → transaction value  
- `Time` → seconds elapsed from a reference point  
- `Class` → target variable (0 = legitimate, 1 = fraud)

---

## 🔧 Machine Learning Pipeline

### **1️⃣ Data Cleaning**
- Checked for missing values (none found)  
- Removed duplicates  
- Standardized numerical features where needed  

### **2️⃣ Feature Engineering**
Created additional features:
- Hour of transaction  
- Part of day (morning, afternoon, evening, night)  
- Log-transformed transaction amount  
- Dropped `Time` after extracting useful patterns  

### **3️⃣ Handling Class Imbalance**
Used **SMOTE** to oversample minority class after splitting.

### **4️⃣ Train/Test Split**
- 80/20 split  
- Stratified to preserve fraud ratio  

### **5️⃣ Models Used**
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### **6️⃣ Threshold Tuning**
Optimized XGBoost decision threshold using the **Precision-Recall Curve** to balance recall and precision.

---

## 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|----------|-----------|--------|-----|----------|---------|
| Logistic Regression | 0.9738 | 0.053 | 0.874 | 0.100 | 0.9587 | 0.6849 |
| Random Forest | 0.9995 | 0.913 | 0.768 | 0.834 | 0.9687 | 0.8205 |
| XGBoost | 0.9988 | 0.611 | 0.811 | 0.697 | 0.9780 | 0.8083 |
| **XGBoost (Tuned)** | **0.9995** | **0.924** | **0.768** | **0.839** | **0.9780** | **0.8083** |

### ⭐ **Best overall model:**  
**XGBoost with threshold tuning** → highest precision while maintaining strong recall.

---

## 📉 Confusion Matrices  
The notebook includes heatmaps for:

- Logistic Regression  
- Random Forest  
- XGBoost  
- XGBoost (Tuned Threshold)  

These visuals help understand trade-offs between false positives and false negatives.

---

## 🧠 What I Learned

- How to handle extremely imbalanced datasets  
- Correct use of SMOTE (after train/test split)  
- Why accuracy is misleading in fraud detection  
- How to tune classification thresholds  
- How to compare models using PR-AUC and F1  
- Structuring a full ML pipeline end-to-end  
- Building a clean, reproducible GitHub project  

---
