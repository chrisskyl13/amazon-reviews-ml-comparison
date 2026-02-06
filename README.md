# Amazon Reviews Classification: Traditional ML vs Gradient Boosting

This repository explores sentiment analysis using classical machine learning approaches, focusing on feature engineering with n-grams and TF-IDF weighting.

##  Methodology & Models
The project evaluates the impact of different text representation techniques:
* **CountVectorizer**: Comparing 1-grams vs. 2-grams (Bigrams).
* **TF-IDF**: Using sublinear TF scaling to handle varied review lengths.
* **XGBoost Classifier**: A powerful gradient boosting implementation for high-dimensional text data.
* **Logistic Regression**: Used as a baseline and for coefficient analysis.
* **Add detailed n-gram logic and results

##  Key Results
* **The Power of Context**: Switching from 1-grams to 2-grams increased the AUC from **0.912** to **0.947**. This allows the model to distinguish between "working" and "not working".
* **Feature Importance**: Analysis of model coefficients reveals which specific phrases (e.g., "very disappointed", "works great") drive the sentiment prediction.
* **TF-IDF Visualization**: Includes a heatmap to visualize how the model "sees" important bigrams in actual reviews.

##  Setup
1. Install dependencies: `pip install pandas numpy scikit-learn xgboost seaborn matplotlib kagglehub`
2. The script automatically downloads the dataset using the Kaggle API.
