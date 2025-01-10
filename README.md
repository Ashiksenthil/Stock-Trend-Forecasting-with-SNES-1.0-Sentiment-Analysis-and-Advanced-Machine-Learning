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



---

## **System Architecture**

![Project Architecture]![image](https://github.com/user-attachments/assets/177ca0dc-2693-4859-8934-67bb10c6293f)

# **Stock Trend Forecasting with Sentiment Analysis and Advanced Machine Learning**

---

## **Data Preprocessing**

Data preprocessing ensures the data is clean, consistent, and ready for machine learning models. The steps include:

- **Data Cleaning**:  
  Remove invalid, inconsistent, or missing entries. For example, if closing prices are missing, impute them using the previous day’s data or remove rows depending on the extent of missing values.

- **Normalization**:  
  Scale numeric features like stock prices using **StandardScaler** or **MinMaxScaler** to ensure uniformity, which helps models like Linear Regression perform better.

- **Outlier Removal**:  
  Detect and handle abnormal values using statistical techniques such as removing data beyond 3 standard deviations or applying domain-specific knowledge.

- **Categorical Encoding**:  
  Convert categorical variables (like industry sectors) into numeric form using **one-hot encoding**, creating binary columns for each category.

---

## **Feature Engineering**

Creating informative features improves model predictions by representing underlying patterns in the data.

- **Technical Indicators**:  
  - **Moving Averages (MA)**: Compute 5-day and 10-day averages to smooth price fluctuations and highlight trends.
  - **Relative Strength Index (RSI)**: Measures momentum and identifies overbought or oversold conditions.
  - **Trading Volume Changes**: Capture shifts in trading activity that may indicate price movements.

- **Lag Features**:  
  - Create lagged versions of variables such as closing prices and sentiment scores to model temporal dependencies.  
  Example: `Lag1_Closing_Price` represents the previous day's closing price.

---

## **Class Imbalance Handling**

Financial data often have more downward than upward price movements. To prevent biased predictions:

- **SMOTE (Synthetic Minority Oversampling Technique)**:  
  Generates synthetic samples for the minority class (upward trends) by interpolating between nearby examples, balancing the dataset for better model performance.
![Before Smote](https://github.com/user-attachments/assets/70d0df8a-55bd-43cd-9055-a109d25cba6e) ![After Smote](https://github.com/user-attachments/assets/dfb7fddb-0353-49d4-ae30-007fa8c67bfc)
---

## **Model Training**

Four machine learning models are used:

- **Random Forest**:  
  An ensemble model combining multiple decision trees to reduce variance and avoid overfitting. It provides insights into feature importance.

- **Linear Regression**:  
  A simple baseline model assuming a linear relationship between features and output, useful for comparison.

- **XGBoost (Extreme Gradient Boosting)**:  
  Known for handling non-linear relationships, XGBoost uses gradient boosting with regularization to prevent overfitting.

- **LightGBM (Light Gradient Boosting Machine)**:  
  Optimized for large datasets, LightGBM uses a leaf-wise tree growth strategy for faster training and higher accuracy.

- **Hyperparameter Tuning**:  
  Parameters like learning rate, maximum depth, and number of trees are optimized using **GridSearchCV** or **Random Search**.

---

## **Model Evaluation**

Models are assessed using various metrics to determine their predictive power:

- **Accuracy**:  
  Measures overall correctness, but it can be misleading with imbalanced data.

- **Precision**:  
  Indicates the proportion of true upward trend predictions among all predicted upward trends.  
  Formula:  
  `Precision = True Positives / (True Positives + False Positives)`

- **Recall (Sensitivity)**:  
  Measures the ability to correctly identify upward trends.  
  Formula:  
  `Recall = True Positives / (True Positives + False Negatives)`

- **F1-Score**:  
  The harmonic mean of precision and recall, balancing both metrics.

- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:  
  Evaluates the model's ability to distinguish between upward and downward trends across thresholds. A higher AUC indicates better performance.

- **Time Series Cross-Validation**:  
  Ensures temporal consistency by training on past data and testing on future data to mimic real-world conditions.

---

## **Feature Importance Analysis**

Understanding which features drive predictions helps improve models and interpret results.

- **Random Forest Feature Importance**:  
  Ranks features by their contribution to reducing impurity in decision trees.

- **Permutation Importance**:  
  Measures how much shuffling a feature's values affects model accuracy, indicating its significance.

- **SHAP (SHapley Additive exPlanations) Values**:  
  Explains the impact and direction of each feature on individual predictions.

### **Most Influential Features**:
- **Recent Closing Prices**: Directly linked to future price changes.
- **Sentiment Scores**: Represent market mood, influencing price direction.
- **RSI and Trading Volume**: Indicators of momentum and trading activity.

---

This README provides detailed, step-by-step guidance for understanding and implementing your stock trend forecasting project. Replace placeholders with actual paths or images to complete the file!




---

## **Installation Guide**

### **Prerequisites**

- Install Python 3.x and the following libraries:
  ```bash
  pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn


