import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def ridge_hyperparam_tuning(X_train, y_train):
    """
    Conduct hyperparameter tuning for ridge regression.
    Parameters:
        X_train (list): Training predictor values
        y_train (list): Training response values
    Returns:
        params: A dictionary of parameters
    """
    # define the hyperparameters
    alpha = [i for i in range(1, 16, 2)]
    model = Ridge()
    params = {"alpha": alpha}

    # perform tuning with GridSearch
    grid_search_result = GridSearchCV(
        model, params, scoring="neg_mean_squared_error", cv=5)
    grid_search_result.fit(X_train, y_train)

    # return the best parameters from tuning
    return grid_search_result.best_params_


def ridge_regression(X_train, X_val, y_train, y_val, params):
    """
    Perform ridge regression.
    Parameters:
        X_train (list): Training predictor values
        X_val (list): Validation predictor values
        y_train (list): Training response values
        X_val (list): Validation response values
    Returns:
        reg: the ridge regression model
    """

    # declare model
    alpha_val = params["alpha"]
    reg = Ridge(alpha=alpha_val)

    # fit training data to model
    reg.fit(X_train, y_train)

    # predict
    y_pred = reg.predict(X_val)

    # evaluate model score
    print(f"R2 score: {reg.score(X_val, y_val)}")
    print(f"MSE value: {mean_squared_error(y_val, y_pred)}")
    return reg


def dt_hyperparam_tuning(X_train, y_train):
    """
    Conduct hyperparameter tuning for decision tree regressor.
    Parameters:
        X_train (list): Training predictor values
        y_train (list): Training response values
    Returns:
        params: A dictionary of parameters
    """

    # define the parameters to be tuned
    criterions = [
        "squared_error", "absolute_error", "friedman_mse"]
    max_depths = [i for i in range(1, 5)]
    min_splits = [i for i in range(2, 10)]
    model = DecisionTreeRegressor()
    params = {
        "criterion": criterions,
        "max_depth": max_depths,
        "min_samples_split": min_splits}

    # perform tuning with GridSearch
    grid_search_result = GridSearchCV(
        model, params, scoring="neg_mean_squared_error", cv=5)
    grid_search_result.fit(X_train, y_train)

    # return the best parameters from tuning
    return grid_search_result.best_params_


def decision_tree_regressor(X_train, X_val, y_train, y_val):
    """
    Perform decision tree regression.
    Parameters:
        X_train (list): Training predictor values
        X_val (list): Validation predictor values
        y_train (list): Training response values
        X_val (list): Validation response values
        params (dict): Dictionary of best parameters
    Returns:
        model: the decision tree regression model
    """

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"R2 score: {model.score(X_val, y_val)}")
    print(f"MSE value: {mean_squared_error(y_val, y_pred)}")
    return model


# declare constants here
TRAIN_SPLIT = 0.7
SPLIT = 0.5
LABEL = "pu_hourly"

# import the training, testing and evaluation datasets
sub_tax_recs_22 = pd.read_csv("data/curated/2022_records.csv")
truth_labels_22 = list(sub_tax_recs_22[LABEL])

sub_tax_recs_23 = pd.read_csv("data/curated/2023_records.csv")
truth_labels_23 = list(sub_tax_recs_23[LABEL])

# split the datasets into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    sub_tax_recs_22, truth_labels_22, train_size=TRAIN_SPLIT, random_state=42)

# split the test set into 2 equal parts
X_1, X_2, y_1, y_2 = train_test_split(
    sub_tax_recs_23, truth_labels_23, train_size=TRAIN_SPLIT, random_state=42)

# run hyperparameter tuning and fitting for ridge regression
params = ridge_hyperparam_tuning(X_train, y_train)
model = ridge_regression(X_train, X_val, y_train, y_val, params)
y_pred = model.predict(X_1)
print(f"Ridge Regression Test R2 score: {model.score(X_1, y_1)}")
print(f"Ridge Regression Test MSE value: {mean_squared_error(y_1, y_pred)}")
print(f"Ridge Regression coefficients and intercept: \
    {model.coef_}, {model.intercept_}")

# run hyperparameter tuning and fitting for decision tree regression
# params = dt_hyperparam_tuning(X_train, y_train)
model = decision_tree_regressor(X_train, X_val, y_train, y_val)
y_pred = model.predict(X_1)
print(f"Decision Tree Regressor Test R2 score: {model.score(X_1, y_1)}")
print(f"Decision Tree Regressor Test MSE value: \
    {mean_squared_error(y_1, y_pred)}")
