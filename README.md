This project aims to predict apartment prices in Brussels area using data collected from real estate website.
Data collection has been performed using web scrappers based on selenium framework.
In order to accelerate data collection and free up local computer cpu, the scrapper has been running on google cloud instances (GCP).
The data processing, cleaning, feature enginnering is performed using machine learning sklearn library and pandas.
The model performs well using XGBoost Regressor without hyperparameter tuning.
Hyperparameters had been tuned using bayesian optimization GPyOpt Library (https://github.com/SheffieldML/GPyOpt).
The final model achieves RMSE of 60k.