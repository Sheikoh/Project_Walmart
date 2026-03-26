# Import lines
import pandas as pd
import numpy as np
import seaborn as sns
import optuna
import math

from dotenv import load_dotenv
import os
import mlflow
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from sklearn.preprocessing import StandardScaler
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
data = pd.read_csv('Walmart_Store_sales.csv')

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

data.describe()

X = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

input_example = x_train.iloc[:3]

#------------------------------------------------------
#---------------------- TRAINING ----------------------
#------------------------------------------------------

print("Training in progress....")

# Pipeline (Scaler + Model)
# pas de ColumnTransformer car seulement des colonnes numériques
def create_model(params):

    models_dic = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=params["alpha"]),
        "Lasso": Lasso(alpha=params["alpha"])
    }

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", models_dic[params["model_type"]])
        ]
    )
    return pipeline

run_description = (
    f"Target: {target}\n"
    "Estimator: Linear Regression\n"
    "StandardScaler + LinearRegression including regularization Lasso\n"
    "Base run with all data"
)

import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_with_target(df, target_name, save_path=None):
  """
  Plots the correlation of each variable in the dataframe with the 'demand' column.

  Args:
  - df (pd.DataFrame): DataFrame containing the data, including a 'target' column.
  - target_name (str): Name of the target column to be used.
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - None (Displays the plot on a Jupyter window)
  """

  # Compute correlations between all variables and 'demand'
  correlations = df.corr()[target].drop(target).sort_values()

  # Generate a color palette from red to green
  colors = sns.diverging_palette(10, 130, as_cmap=True)
  color_mapped = correlations.map(colors)

  # Set Seaborn style
  sns.set_style(
      "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
  )  # Light grey background and thicker grid lines

  # Create bar plot
  fig = plt.figure(figsize=(12, 8))
  plt.barh(correlations.index, correlations.values, color=color_mapped)

  # Set labels and title with increased font size
  plt.title(f"Correlation with {target}", fontsize=18)
  plt.xlabel("Correlation Coefficient", fontsize=16)
  plt.ylabel("Variable", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="x")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # prevent matplotlib from displaying the chart every time we call this function
  plt.close(fig)

  return fig


# Test the function
# correlation_plot = plot_correlation_with_demand(data, save_path="correlation_plot.png")


def plot_residuals(y_pred, y_true, save_path=None):
  """
  Plots the residuals of the model predictions against the true values.

  Args:
  - y_pred (pd.Series): The prediction done by the model.
  - y_true (pd.Series): The true values for the validation set.
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - None (Displays the residuals plot on a Jupyter window)
  """

  # Predict using the model
#   preds = model.predict(dvalid)

  # Calculate residuals
  residuals = y_true - y_pred

  # Set Seaborn style
  sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

  # Create scatter plot
  fig = plt.figure(figsize=(12, 8))
  plt.scatter(y_true, residuals, color="blue", alpha=0.5)
  plt.axhline(y=0, color="r", linestyle="-")

  # Set labels, title and other plot properties
  plt.title("Residuals vs True Values", fontsize=18)
  plt.xlabel("True Values", fontsize=16)
  plt.ylabel("Residuals", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="y")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # Show the plot
  plt.close(fig)

  return fig

def plot_correlation(y_pred, y_true, save_path=None):
  """
  Plots the residuals of the model predictions against the true values.

  Args:
  - y_pred (pd.Series): The prediction done by the model.
  - y_true (pd.Series): The true values for the validation set.
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - None (Displays the residuals plot on a Jupyter window)
  """

  # Calculate residuals
#   residuals = y_true - y_pred

  # Set Seaborn style
  sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

  # Create scatter plot
  fig = plt.figure(figsize=(12, 8))
  plt.scatter(y_true, y_pred, color="blue", alpha=0.5)
#   plt.axhline(y=0, color="r", linestyle="-")

  # Set labels, title and other plot properties
  plt.title("Prediction vs True Values", fontsize=18)
  plt.xlabel("True Values", fontsize=16)
  plt.ylabel("Prediction", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="y")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # Show the plot
  plt.close(fig)

  return fig

def plot_history(study, save_path=None):
  """
  Plots the history of the model metrics along the steps.

  Args:
  - Study (Optuna.study): The study containing its own history of metrics
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - None (Displays the residuals plot on a Jupyter window)
  """
  trials = study.get_trials()
  history = [trial.value for trial in trials]
  # Set Seaborn style
  sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

  # Create scatter plot
  fig = plt.figure(figsize=(12, 8))
  plt.scatter(range(len(history)), history, color="blue", alpha=0.5)
#   plt.axhline(y=0, color="r", linestyle="-")

  # Set labels, title and other plot properties
  plt.title("Prediction vs True Values", fontsize=18)
  plt.xlabel("True Values", fontsize=16)
  plt.ylabel("Prediction", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="y")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # Show the plot
  plt.close(fig)

  return fig

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
  """
  Logging callback that will report when a new trial iteration improves upon existing
  best trial values.

  Note: This callback is not intended for use in distributed computing systems such as Spark
  or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
  workers or agents.
  The race conditions with file system state management for distributed trials will render
  inconsistent values with this callback.
  """

  winner = study.user_attrs.get("winner", None)

  if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-8, 5, log=True),
            "model_type": trial.suggest_categorical("model_type", ["Linear", "Ridge", "Lasso"])
            }
        
        pipeline = create_model(params)
        pipeline.fit(x_train, y_train)
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
        mlflow.log_param("model", params["model_type"])
        mlflow.log_param("test_size", test_size)


        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            name=params["model_type"],
            registered_model_name = registered_model_name,
            input_example=input_example,
            signature=signature,
            # code_paths=["func_feat_eng.py", "Model_func.py"]
        )
        return mse

run_name = "graphLog_attempt"

 # Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(run_name=run_name, nested=True):
  # Initialize the Optuna study
  study = optuna.create_study(direction="minimize")

  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
  study.optimize(objective, n_trials=10, callbacks=[champion_callback])

  mlflow.log_params(study.best_params)
  mlflow.log_metric("best_mse", study.best_value)
  mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

  # Log tags
  mlflow.set_tags(
      tags={
          "project": "Walmart Project",
          "optimizer_engine": "optuna",
          "model_family": study.best_params["model_type"],
          "feature_set_version": 1,
      }
  )

  # Log a fit model instance
  pipeline = create_model(study.best_params)
  pipeline.fit(x_train, y_train)
  y_pred = pipeline.predict(x_test)


  # Log the correlation plot
  correlation_plot = plot_correlation_with_target(data,target_name=target, save_path="correlation_plot.png")
  mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")

  # Log the feature importances plot
#   importances = plot_feature_importance(model, booster=study.best_params.get("booster"))
#   mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

  # Log the residuals plot
  residuals = plot_residuals(y_pred, y_test, save_path="residuals.png")
  mlflow.log_figure(figure=residuals, artifact_file="residuals.png")

  # Log the correlation between pred and true values
  prediction = plot_correlation(y_pred, y_test, save_path="prediction.png")
  mlflow.log_figure(figure=prediction, artifact_file="prediction.png")

  #Log the evolution of the metric
  history = plot_history(study, save_path = "history.png")
  mlflow.log_figure(figure=history, artifact_file="history.png")

    # MLflow signature
    
  signature = infer_signature(x_train, pipeline.predict(x_train))


  mlflow.sklearn.log_model(
      pipeline,
      name=study.best_params["model_type"],
      registered_model_name=registered_model_name,
      input_example=input_example,
      signature=signature
  )

  # Get the logged model uri so that we can load it from the artifact store
  model_uri = mlflow.get_artifact_uri(registered_model_name)   

#--- Set registered model alias
client = MlflowClient()

model = client.get_registered_model(registered_model_name)
latest_version = model.latest_versions[-1].version

client.set_registered_model_alias(registered_model_name, alias, latest_version)
print(f"Attribution de l'alias '{alias}' à la version {latest_version} du model {registered_model_name}")
print("End of model training")