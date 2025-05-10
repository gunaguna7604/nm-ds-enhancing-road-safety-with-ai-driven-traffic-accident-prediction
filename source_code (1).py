import pandas as pd
import numpy as np
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("global_traffic_accidents.csv")

# Create Severity label based on Casualties
df["Severity"] = pd.cut(
    df["Casualties"],
    bins=[-1, 2, 5, float("inf")],
    labels=["Low", "Medium", "High"]
)

# Select features
features = ["Weather Condition", "Road Condition", "Vehicles Involved"]
X = df[features]
y = df["Severity"]

# Encode categorical variables
X = pd.get_dummies(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Define prediction function
def predict_severity(weather, road, vehicles):
    input_df = pd.DataFrame([[weather, road, vehicles]], columns=["Weather Condition", "Road Condition", "Vehicles Involved"])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_df)[0]
    severity = le.inverse_transform([prediction])[0]
    return f"Predicted Accident Severity: {severity}"

# Gradio app
# Convert choices to strings to avoid the ValueError
interface = gr.Interface(
    fn=predict_severity,
    inputs=[
        gr.Dropdown(choices=[str(c) for c in df["Weather Condition"].unique()], label="Weather Condition"), # Convert to string
        gr.Dropdown(choices=[str(c) for c in df["Road Condition"].unique()], label="Road Condition"), # Convert to string
        gr.Number(label="Vehicles Involved")
    ],
    outputs="text",
    title="AI Traffic Accident Severity Predictor"
)

if __name__ == "__main__":
    interface.launch()
