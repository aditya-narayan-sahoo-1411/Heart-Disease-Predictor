# ❤️ Heart Disease Prediction System

A Machine Learning web application that predicts the likelihood of heart disease based on patient clinical data. The application is built using **Python**, **Scikit-Learn**, and **Streamlit**, providing an interactive interface for real-time predictions.

## 📖 Project Overview

Heart disease is one of the leading causes of death worldwide. Early identification of high-risk patients can support timely medical evaluation and treatment.

In this project, multiple Machine Learning classification algorithms were trained and compared. After evaluating their performance using **Accuracy** and **F1 Score**, the **K-Nearest Neighbors (KNN)** classifier was selected as the final model and deployed through a Streamlit web application.

## ✨ Features

* Interactive web interface built with **Streamlit**
* Predicts the likelihood of heart disease using Machine Learning
* User-friendly input form with sliders and dropdown menus
* Real-time prediction results with confidence score
* Comparison of multiple Machine Learning algorithms
* Data preprocessing including missing value handling and one-hot encoding
* Feature scaling using StandardScaler
* Trained model saved using Joblib for deployment
* Responsive and easy-to-use interface

## 🛠️ Tech Stack

**Programming Language**

* Python

**Machine Learning**

* Scikit-Learn
* K-Nearest Neighbors (KNN)

**Data Analysis**

* NumPy
* Pandas

**Data Visualization**

* Matplotlib
* Seaborn

**Deployment**

* Streamlit

**Model Serialization**

* Joblib

**Development Environment**

* VS Code
* Jupyter Notebook

## 📊 Model Performance

The following Machine Learning models were trained and evaluated on the Heart Disease dataset.

| Model                          | Accuracy   | F1 Score   |
| ------------------------------ | ---------- | ---------- |
| Logistic Regression            | 87.50%     | 88.78%     |
| **K-Nearest Neighbors (KNN)**  | **88.59%** | **89.86%** |
| Naive Bayes                    | 86.96%     | 87.88%     |
| Decision Tree                  | 75.00%     | 76.04%     |
| SVM (RBF Kernel)               | 86.41%     | 88.04%     |

### Selected Model

**K-Nearest Neighbors (KNN)** was selected as the final model because it achieved the highest overall performance on the test dataset.


## 📁 Project Structure

```text
Heart-Disease-Predictor/
│
├── app.py                  # Streamlit web application
├── prediction.py           # Prediction logic
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
│
├── data/
│   └── heart.csv           # Dataset
│
├── models/
│   ├── heart_model.pkl     # Trained KNN model
│   ├── scaler.pkl          # StandardScaler object
│   └── columns.pkl         # Feature column names
│
└── notebook/
    └── HeartDisease_Analysis.ipynb
```


## ▶️ Usage

1. Enter the patient's clinical information.
2. Click the **Predict** button.
3. View the prediction result and confidence score.
4. Use the prediction as a decision-support tool only. It should not replace professional medical advice.
