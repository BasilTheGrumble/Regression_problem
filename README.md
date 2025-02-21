# Educational Project on Solving a Regression Problem
## Project Description
The goal of my project is to predict car prices based on data collected from the [Kaggle](https://www.kaggle.com) platform. The dataset can be found at the following link: [Australian Vehicle Prices](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices). The data contains information about various car characteristics, such as engine type, number of seats, year of manufacture, fuel type, and other parameters. I will explore three machine learning models (Lasso Regression, Random Forest, Gradient Boosting) and compare their performance to select the optimal one.

## Project Structure
1. **`Filtering.py`**:  
   - In this file, I created a complete data preprocessing pipeline.  
   - I included custom transformers for processing categorical and numerical features, such as filtering car types, extracting engine information, categorizing the year of manufacture, and grouping fuel types.  
   - To automate the preprocessing process, I used `ColumnTransformer` and `Pipeline`.
2. **`Modeling.py`**:  
   - This file contains the code I wrote to train and compare three models: Lasso Regression, Random Forest, and Gradient Boosting.  
   - For each model, I performed hyperparameter tuning using `GridSearchCV` and evaluated performance using cross-validation.  
   - To compare the models, I used metrics such as MAE, RMSE, R², and CV_RMSE.
3. **`Australian_Vehicle_Prices.csv`**:  
   - This is the original dataset containing data on car prices and their characteristics.
---
## Results
I trained and evaluated three models based on the following metrics:

| Model               | MAE       | RMSE      | R²    | CV_RMSE  |
|---------------------|-----------|-----------|-------|----------|
| **Lasso Regression**| 14842.95  | 26925.56  | 0.37  | 31163.63 |
| **Random Forest**   | 10113.83  | 21581.72  | 0.60  | 24231.99 |
| **Gradient Boosting**| 12092.23 | 23561.40  | 0.52  | 26425.29 |

### Analysis of Results:
1. **Random Forest** shows the best results across all metrics:
   - The smallest error (MAE and RMSE).
   - The highest coefficient of determination (R² = 0.60), indicating better explanation of data variance.
   - The smallest error on cross-validation (CV_RMSE), which indicates high model stability.
2. **Gradient Boosting** also demonstrates good results but lags behind Random Forest in all metrics.
3. **Lasso Regression** shows the worst results, which may be due to the linear nature of the model and the complexity of the data.
---
## Conclusions
Based on the conducted analysis, I made the following conclusions:
1. **Random Forest** is the most suitable model for predicting car prices in this case. It provides the best balance between prediction accuracy and stability.
2. Hyperparameter tuning and the use of cross-validation allowed me to significantly improve the performance of all models.
3. Data preprocessing (filtering, categorization, filling missing values) played a key role in improving model quality.
---
