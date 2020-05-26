This project aims to predict apartment prices in Brussels area using data collected from real estate website.`<addr>`
Data collection has been performed using web scrappers based on selenium framework.`<addr>`
In order to accelerate data collection and free up local computer cpu, the scrapper has been running on google cloud instances (GCP).`<addr>`
The data processing, cleaning, feature enginnering is performed using machine learning sklearn library and pandas.`<addr>`
The model performs well using XGBoost Regressor without hyperparameter tuning.`<addr>`
Hyperparameters had been tuned using bayesian optimization GPyOpt Library [GPyOpt](https://github.com/SheffieldML/GPyOpt).`<addr>`
The final model achieves RMSE of 60k.
