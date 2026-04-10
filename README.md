# Project Walmart

## Objective

The objective of this project is to predict the Weekly_Sales for the Walmart stores. 

## Dataset

In order to do so, we are using a dataset containing information about the customers:
Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Date


## Study
### Preprocessing

**Date column**
The Date column in its initial state would be considered as text by the model, and must be modified to be interpreted as a numerical value and contribute to the model efficiency.
It is thus transformed in a datetime format and new columns are created to express the year, month, day and day of the week, before dropping the initial Date column. 

**Missing values and Outliers**
- Missing values in Weekly_Sales are dropped (no target, no interest)
- Missing values in Holiday_Flag are filled with 0 (considered as not holidays)
- Missing values in Temperature are filled with the average.

The values at more than 3 variances from the average are considered outliers and their lines are dropped from the dataset. 

The numerical columns are preprocessed with a Standardcaler, and a OneHotEncoder is used on the Store column.

### Machine Learning
**Models**
The prediction was tested using a Linear Regression, Lasso and Ridge. 
The hyperparameter *alpha* was optimised using Optuna, and the complete training was registered on an mlflow server hosted on huggingface.

**Results**