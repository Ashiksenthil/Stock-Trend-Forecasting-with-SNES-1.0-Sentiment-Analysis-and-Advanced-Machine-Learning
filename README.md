# **Stock Trend Forecasting Using Sentiment Analysis and Machine Learning**

This project focuses on predicting stock price movements using **financial market data** and **sentiment analysis from news articles**. By leveraging the **SNES 1.0 dataset**, various **machine learning models** are used to improve prediction accuracy for stock trends. This research combines technical indicators, market sentiment, and advanced machine learning techniques to create a powerful stock forecasting system.

---

## **Project Goals**

- **Predict stock price directions (up or down)** using market data and sentiment scores.
- **Incorporate sentiment analysis** to capture market emotions and trends.
- **Evaluate multiple machine learning models** to determine the best predictor.
- **Balance data classes** using SMOTE (Synthetic Minority Oversampling Technique).
- **Perform robust validation** with time series cross-validation.

---

## **Key Features**

- **Real-World Dataset**: Uses the SNES 1.0 dataset with stock prices, trading volumes, and news sentiment.
- **Advanced Feature Engineering**:  
  - Moving averages and lagged sentiment scores to capture market trends.
  - Relative Strength Index (RSI) as a technical indicator.
- **Machine Learning Models**:  
  - **Random Forest** for feature importance ranking.  
  - **Linear Regression** as a baseline.  
  - **XGBoost** and **LightGBM** for efficient gradient boosting.  
- **SMOTE for Class Imbalance**:  
  - Enhances model performance by balancing the number of uptrend and downtrend samples.
- **Comprehensive Model Evaluation**:  
  - Accuracy, precision, recall, F1-score, and ROC-AUC.

---

## **Dataset Information**

- **Source**: SNES 1.0 Dataset (available on Kaggle).
- **Timeframe**: November 2020 to July 2022.
- **Features Included**:
  - **Stock Prices**: Closing prices, trading volumes.
  - **Sentiment Data**: Derived from financial news articles.
  - **Technical Indicators**: Moving averages (5-day, 10-day), RSI.

---

## **System Workflow**

1. **Data Preprocessing**:
   - Clean data to handle missing values.
   - Normalize numerical features for consistency.
   - Apply **feature engineering** to create new insights.
2. **Model Selection**:
   - Use four models: Random Forest, Linear Regression, XGBoost, and LightGBM.
   - Fine-tune models with hyperparameter optimization.
3. **Class Imbalance Handling**:
   - Apply **SMOTE** to balance data for better prediction.
4. **Model Evaluation**:
   - Use **time series cross-validation** to maintain temporal data order.
5. **Feature Importance**:
   - Identify the key drivers of stock price movements.

![System Workflow Diagram](path/to/system-workflow-image.png)

---

## **Implementation Guide**

### **Prerequisites**

- Install the following libraries:
  ```bash
  pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib

