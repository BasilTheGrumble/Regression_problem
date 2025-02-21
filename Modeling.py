from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from Filtering import train_df_preprocessed, test_df_preprocessed


X_train = train_df_preprocessed.drop(columns=['Price'])
Y_train = train_df_preprocessed['Price']
X_test = test_df_preprocessed.drop(columns=['Price'])
Y_test = test_df_preprocessed['Price']


def evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    metrics = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
        "CV_RMSE": cv_rmse
    }

    return metrics


lasso_param_grid = {
    'alpha': [0.01, 0.1]
}
lasso = GridSearchCV(Lasso(max_iter=1000000), lasso_param_grid, cv=5, scoring='neg_mean_squared_error')


rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='neg_mean_squared_error')


gb_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.001, 0.01],
    'max_depth': [5, 10]
}
gb = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=5, scoring='neg_mean_squared_error')


results = [evaluate_model(lasso, X_train, Y_train, X_test, Y_test, "Lasso Regression"),
           evaluate_model(rf, X_train, Y_train, X_test, Y_test, "Random Forest"),
           evaluate_model(gb, X_train, Y_train, X_test, Y_test, "Gradient Boosting")]


for result in results:
    print(f"Model: {result['Model']}")
    print(f"  - MAE: {result['MAE']:.2f}")
    print(f"  - RMSE: {result['RMSE']:.2f}")
    print(f"  - R²: {result['R²']:.2f}")
    print(f"  - CV_RMSE: {result['CV_RMSE']:.2f}")
    print()