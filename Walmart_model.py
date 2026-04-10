# Import lines
import pandas as pd
import numpy as np
import seaborn as sns

from dotenv import load_dotenv
import os
import mlflow
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# setting Jedha color palette as default
pio.templates["jedha"] = go.layout.Template(
    layout_colorway=["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"]
)
pio.templates.default = "jedha"
pio.renderers.default = "svg" # to be replaced by "iframe" if working on JULIE
from IPython.display import display


# ---- VARIABLES ----

load_dotenv()

## TARGET
target = "Weekly_Sales"

## Training env
# os.environ["MLFLOW_TRACKING_URI"] = "https://sheyko-mlflow-server-ft34.hf.space/"
# EXPERIMENT_NAME = "Walmart_model"
model_name = "Lasso_model"
test_size = 0.25
registered_model_name = "Greedy_weekly"
alias = "challenger"

# MLflow config
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["EXPERIMENT_NAME"])

# --- DATA LOADING AND CLEANING ---

## Data loading
data = pd.read_csv('data/Walmart_Store_sales.csv')

## Data cleaning
### Drop missing values
data = data[data["Weekly_Sales"].notna()]
data["Holiday_Flag"] = data["Holiday_Flag"].fillna(0)
data["Temperature"] = data["Temperature"].fillna(data["Temperature"].mean())
data = data[data["Date"].notna()]

### Decomposing Date into its components for ingestion by the model
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["day"] = data["Date"].dt.day
data["day_in_week"] = data["Date"].dt.day_of_week
data.drop("Date", axis=1, inplace=True)

### Removal of the outliers: Per the convention of this project, the outlying values will be all the values at 3 Sigma from the average
col_list = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]

for col in col_list:
    avg = data[col].mean()
    var = data[col].std()
    data = data[data[col].between(avg-3*var, avg+3*var)]

data["Store"] = data["Store"].astype(int).astype(str)

data.describe()

#------------------------------------------------------
#---------------------- TRAINING ----------------------
#------------------------------------------------------

print("Training in progress....")

X = data.drop(target, axis = 1)
y = data[target]

num_col = X.columns.drop("Store")
cat_col = ["Store"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, #random_state=42,
    stratify=["Store"]
)

input_example = x_train.iloc[:3]

# Pipeline (Scaler + Model)
# pas de ColumnTransformer car seulement des colonnes numériques
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(), num_col),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_col)
        ("model", Lasso())
    ]
)

run_description = (
    f"Target: {target}\n"
    "Estimator: Linear Regression\n"
    "StandardScaler + LinearRegression including regularization Lasso\n"
    "Base run with all data"
)

# MLflow run
with mlflow.start_run(description=run_description):
    # Train
    pipeline.fit(x_train, y_train)

    # Predict
    y_pred = pipeline.predict(x_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    n = len(y_test)
    p = x_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # MLflow signature
    signature = infer_signature(x_train, pipeline.predict(x_train))

    # Log metrics
    mlflow.log_metrics({
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2
    })

    # Log params
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("model", "Lasso")
    mlflow.log_param("test_size", test_size)


    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        name=model_name,
        registered_model_name = registered_model_name,
        input_example=input_example,
        signature=signature,
        # code_paths=["func_feat_eng.py", "Model_func.py"]
    )

#--- Set registered model alias
client = MlflowClient()

model = client.get_registered_model(registered_model_name)
latest_version = model.latest_versions[-1].version

client.set_registered_model_alias(registered_model_name, alias, latest_version)
print(f"Attribution de l'alias '{alias}' à la version {latest_version} du model {registered_model_name}")
print("End of model training")