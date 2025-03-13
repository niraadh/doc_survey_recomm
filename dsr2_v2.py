import io
import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
file_path = "dummy_npi_data.csv"  # Change path if needed

if not os.path.exists(file_path):
    st.error(f"The file '{file_path}' was not found. Please check the path and try again.")
    st.stop()

df = pd.read_csv(file_path)

# Convert Login and Logout Time to datetime
df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce')
df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce')

# Drop rows with invalid timestamps
df = df.dropna(subset=['Login Time', 'Logout Time'])

# Extract hour and minute from Login and Logout time
df['Login Hour'] = df['Login Time'].dt.hour
df['Login Minute'] = df['Login Time'].dt.minute
df['Logout Hour'] = df['Logout Time'].dt.hour
df['Logout Minute'] = df['Logout Time'].dt.minute

# Normalize survey attempts and usage time for probability scoring
if df['Count of Survey Attempts'].max() > 0:
    df['Survey Score'] = df['Count of Survey Attempts'] / df['Count of Survey Attempts'].max()
else:
    df['Survey Score'] = 0

if df['Usage Time (mins)'].max() > 0:
    df['Usage Score'] = df['Usage Time (mins)'] / df['Usage Time (mins)'].max()
else:
    df['Usage Score'] = 0

# Compute an initial probability score
df['Initial Probability'] = (df['Survey Score'] + df['Usage Score']) / 2

# Define features and target variable
features = ['Login Hour', 'Login Minute', 'Logout Hour', 'Logout Minute', 'Survey Score', 'Usage Score']
target = 'Initial Probability'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f'Model MAE: {mae}')

# Function to get the most probable doctors
def get_most_probable_doctors(input_hour, input_minute):
    """
    Given an input hour and minute, predict the probability of doctors being available
    and return the sorted list of all doctors.
    """
    input_data = df.copy()
    
    input_data['Is Within Time'] = (
        ((input_hour > input_data['Login Hour']) | 
        ((input_hour == input_data['Login Hour']) & (input_minute >= input_data['Login Minute']))) &
        ((input_hour < input_data['Logout Hour']) | 
        ((input_hour == input_data['Logout Hour']) & (input_minute <= input_data['Logout Minute'])))
    ).astype(int)
    
    X_input = input_data[['Login Hour', 'Login Minute', 'Logout Hour', 'Logout Minute', 'Survey Score', 'Usage Score']]
    input_data['Predicted Probability'] = model.predict(X_input)
    
    input_data['Final Probability'] = input_data['Predicted Probability'] * (1 + 0.5 * input_data['Is Within Time'])
    
    sorted_doctors = input_data[['NPI', 'Speciality', 'Final Probability']].sort_values(by='Final Probability', ascending=False)
    
    return sorted_doctors

# Streamlit UI
st.title("Doctor Availability Predictor")
input_hour = st.slider("Select an hour", 0, 23, 12)
input_minute = st.slider("Select minutes", 0, 59, 0)

if st.button("Predict"):
    results = get_most_probable_doctors(input_hour, input_minute)
    st.write("Doctors Sorted by Probability:")
    st.dataframe(results)
    
    # Provide option to download results as CSV or Excel
    csv = results.to_csv(index=False).encode('utf-8')
    
    # Fix the issue with Excel download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        results.to_excel(writer, index=False)
    excel_data = excel_buffer.getvalue()
    
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="doctor_predictions.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="Download as Excel",
        data=excel_data,
        file_name="doctor_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
