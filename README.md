# **Stock Trend Forecasting Using Sentiment Analysis and Advanced Machine Learning**

This project aims to forecast short-term stock price movements using **financial data and sentiment analysis** derived from news articles. By leveraging the **SNES 1.0 dataset**, we combine traditional market indicators with news sentiment scores, employing **state-of-the-art machine learning models** for improved stock trend prediction. This work addresses challenges in financial forecasting by applying advanced feature engineering and robust evaluation methods.

---

## **Motivation**

Financial markets are influenced by numerous factors, including investor sentiment, economic indicators, and geopolitical events. Traditional forecasting methods often fall short in capturing the complexity of market dynamics. This project integrates **sentiment analysis** with **machine learning models** to predict stock price movements, offering insights for investors, traders, and financial analysts. The goal is to explore whether **sentiment scores** can enhance prediction accuracy and how different machine learning models perform in this context.

---

## **Project Objectives**

1. **Predict stock price direction (upward or downward)** using both market data and sentiment analysis.
2. **Incorporate feature engineering** to represent temporal patterns and market dynamics.
3. **Handle class imbalance** with SMOTE to ensure robust predictions.
4. **Evaluate and compare machine learning models**: Random Forest, Linear Regression, XGBoost, and LightGBM.
5. **Analyze feature importance** to identify key drivers of stock price movements.

---

## **Key Features of the Project**

- **Sentiment Analysis**: Uses pre-calculated sentiment scores from financial news articles as input features.
- **Machine Learning Algorithms**:
  - **Random Forest**: Provides a benchmark and insights into feature importance.
  - **Linear Regression**: A simple baseline model to compare against advanced techniques.
  - **XGBoost**: Known for its gradient boosting and handling of sparse data.
  - **LightGBM**: Optimized for efficiency and high performance.
- **Feature Engineering**:
  - **Moving Averages (MA)**: Tracks trends over 5-day and 10-day windows.
  - **Relative Strength Index (RSI)**: Measures momentum in price changes.
  - **Lag Features**: Incorporate past values to capture temporal dependencies.
- **Class Imbalance Solution**: Implements **SMOTE (Synthetic Minority Oversampling Technique)** to address skewed class distributions.
- **Model Validation**: Employs **time series cross-validation** to respect data ordering and prevent lookahead bias.

---

## **Data Used**

### **Dataset: SNES 1.0**
- **Source**: Kaggle (Stock-NewsEventsSentiment)
- **Timeframe**: November 2020 – July 2022
- **Features**:
  - **Closing Prices and Trading Volume**: Standard financial metrics.
  - **Sentiment Scores**: Derived from financial news articles for S&P 500 companies.
  - **Technical Indicators**: Moving averages, RSI.
  - **Lagged Sentiment Features**: Sentiment scores from previous days.

![Dataset Overview](path/to/dataset-image.png)

---

## **System Architecture**

This project follows a structured pipeline for stock trend forecasting:

1. **Data Preprocessing**:
   - Clean and normalize data to ensure consistency.
   - Apply one-hot encoding for categorical variables.
   - Remove outliers and handle missing values.

2. **Feature Engineering**:
   - **Technical Indicators**: Calculate moving averages, RSI, and trading volume changes.
   - **Lag Features**: Create lagged versions of closing prices and sentiment scores.

3. **Class Imbalance Handling**:
   - Use **SMOTE** to generate synthetic samples for the minority class (upward price movements).

4. **Model Training**:
   - Train and compare four models: Random Forest, Linear Regression, XGBoost, and LightGBM.
   - Use **hyperparameter tuning** to optimize each model’s performance.

5. **Model Evaluation**:
   - Evaluate using metrics: **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.
   - Implement **time series cross-validation** for realistic performance assessment.

6. **Feature Importance Analysis**:
   - Determine the most influential factors in stock price movement predictions.

![System Workflow](path/to/system-workflow-image.png)

---

## **Installation Guide**

### **Prerequisites**

- Install Python 3.x and the following libraries:
  ```bash
  pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn


