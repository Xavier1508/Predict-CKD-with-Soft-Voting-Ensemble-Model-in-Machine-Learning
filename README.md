# Early Prediction of Chronic Kidney Disease (CKD)

## 📖 Project Description
This project is a web-based application designed to assist in the **early detection of Chronic Kidney Disease (CKD)**.  
It leverages a **Soft Voting and Stacking Ensemble Machine Learning model** to analyze various medical parameters and predict the probability of CKD risk.

The main goal of this project is to provide a **fast, accessible, and informative screening tool** for both healthcare professionals and individuals, based on laboratory data.  
The project can be run with **two different interfaces**: **Streamlit (modern and interactive)** and **Flask (traditional web server)**.

---

## ⚙️ Key Features
- **High Prediction Accuracy:**  
  Utilizes a Stacking Ensemble model that combines **Random Forest, XGBoost, and LightGBM** to achieve robust accuracy.

- **Interactive Interface:**  
  Simple and intuitive form for patient data input.

- **Comprehensive Results:**  
  Displays prediction results (**High Risk / Low Risk**) along with a confidence probability score.

- **Dual Deployment Options:**  
  Includes both **Flask backend** and **Streamlit frontend** for flexibility.

- **Automated Preprocessing:**  
  Automatically handles missing values and performs feature scaling before prediction.

---

## 🧠 Model & Technologies

### Core Model
**StackingClassifier (Ensemble)** composed of:
- **Base Models:** RandomForest, XGBoost, LightGBM  
- **Meta-Model:** Logistic Regression  

### Languages and Libraries
- **Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, XGBoost, LightGBM, Imbalanced-learn  

### Deployment Options
- **Streamlit:** Modern and interactive web interface (`streamlit_app.py`)  
- **Flask:** Traditional web API and HTML/CSS rendering (`app.py`)  