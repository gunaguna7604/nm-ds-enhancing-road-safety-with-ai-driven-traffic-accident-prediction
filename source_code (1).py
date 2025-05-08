
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import gradio as gr

# Load dataset
df = pd.read_csv("global_traffic_accidents.csv")

# Drop irrelevant columns
df = df.drop(columns=["Accident ID", "Date", "Time", "Location"])

# Define features and target
X = df.drop(columns=["Casualties"])
y = df["Casualties"]

# Identify categorical and numerical features
categorical_features = ["Weather Condition", "Road Condition", "Cause"]
numerical_features = ["Latitude", "Longitude", "Vehicles Involved"]

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"  # Leave numerical features as-is
)

# Build model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Gradio interface
def predict_casualties(weather, road, cause, lat, lon, vehicles):
    input_df = pd.DataFrame([{
        "Weather Condition": weather,
        "Road Condition": road,
        "Cause": cause,
        "Latitude": lat,
        "Longitude": lon,
        "Vehicles Involved": vehicles
    }])
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

iface = gr.Interface(
    fn=predict_casualties,
    inputs=[
        gr.Dropdown(choices=df["Weather Condition"].unique().tolist(), label="Weather Condition"),
        gr.Dropdown(choices=df["Road Condition"].unique().tolist(), label="Road Condition"),
        gr.Dropdown(choices=df["Cause"].unique().tolist(), label="Cause"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
        gr.Slider(1, 10, step=1, label="Vehicles Involved")
    ],
    outputs="number",
    title="Traffic Accident Casualty Predictor",
    description="Predicts number of casualties based on accident features."
)

if __name__ == "__main__":
    iface.launch()
