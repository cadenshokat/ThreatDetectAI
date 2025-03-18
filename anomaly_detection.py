import random
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import logging
import argparse
import joblib

# Configure logging
logging.basicConfig(filename='anomaly_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Simulated data: Initial set of normal network traffic
normal_data = {
    'bytes_sent': [random.randint(50, 500) for _ in range(10)],  # Normal range for bytes_sent
    'errors': [random.randint(1, 5) for _ in range(10)]  # Normal range for errors
}

# Initialize Isolation Forest model (anomaly detection)
model_if = IsolationForest(contamination=0.33)


# Function to generate and predict real-time data
def generate_real_time_data():
    new_data = {
        'bytes_sent': random.randint(50, 500),
        'errors': random.randint(1, 5)
    }
    return new_data


# Function to simulate the real-time data process
def process_real_time_data():
    data = pd.DataFrame(normal_data)
    model_if.fit(data[['bytes_sent', 'errors']])  # Fit the model with the initial data

    print("Starting real-time anomaly detection...")

    # Real-time data simulation loop
    for _ in range(50):  # Simulate 50 incoming data points
        new_data = generate_real_time_data()  # Generate new data
        print(f"New data: {new_data}")

        # Convert the data to a DataFrame
        new_df = pd.DataFrame([new_data])

        # Predict anomalies
        prediction = model_if.predict(new_df[['bytes_sent', 'errors']])

        # Display the prediction
        print("Anomaly detected!" if prediction == -1 else "No anomaly detected.")

        # Log anomaly detection
        log_anomaly(new_data, 'Isolation Forest')

        # Visualize real-time data (optional)
        plt.scatter(new_df['bytes_sent'], new_df['errors'], color='red' if prediction == -1 else 'blue')
        plt.xlabel('Bytes Sent')
        plt.ylabel('Errors')
        plt.title('Real-Time Network Traffic Anomaly Detection')
        plt.pause(0.5)  # Pause to allow plotting updates

        # Wait before generating the next data point
        time.sleep(1)


# Log anomaly to the log file
def log_anomaly(data_point, model_name):
    logging.info(f"Anomaly detected by {model_name}: {data_point}")


# Multi-Model Anomaly Detection Function
def run_multi_model_anomaly_detection(data):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Model 1: Isolation Forest
    model_if.fit(scaled_data)
    data['IF_anomaly'] = model_if.predict(scaled_data)

    # Model 2: K-Means Clustering
    kmeans = KMeans(n_clusters=2)
    data['KMeans_anomaly'] = kmeans.fit_predict(scaled_data)

    # Model 3: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    dbscan = LocalOutlierFactor(n_neighbors=2, contamination=0.33)
    data['DBSCAN_anomaly'] = dbscan.fit_predict(scaled_data)

    # Save models (Optional)
    joblib.dump(model_if, 'models/isolation_forest_model.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(dbscan, 'models/dbscan_model.pkl')

    # Visualize the results of all models
    plt.scatter(data['bytes_sent'], data['errors'], c=data['IF_anomaly'], cmap='coolwarm', label='Isolation Forest')
    plt.scatter(data['bytes_sent'], data['errors'], c=data['KMeans_anomaly'], cmap='viridis', marker='x',
                label='KMeans')
    plt.scatter(data['bytes_sent'], data['errors'], c=data['DBSCAN_anomaly'], cmap='plasma', marker='*', label='DBSCAN')
    plt.legend()
    plt.title('Anomaly Detection Using Multiple Models')
    plt.xlabel('Bytes Sent')
    plt.ylabel('Errors')
    plt.show()


# Storing anomalies over time
history = []


# Simulate adding anomalies to the history
def add_to_history(data_point, model_name):
    history.append({'timestamp': pd.Timestamp.now(), 'data': data_point, 'model': model_name})


# After each anomaly detection
add_to_history({'bytes_sent': 1000, 'errors': 30}, 'Isolation Forest')
add_to_history({'bytes_sent': 500, 'errors': 10}, 'DBSCAN')

# Display history
history_df = pd.DataFrame(history)
print(history_df)

# Set up argument parsing for the CLI
parser = argparse.ArgumentParser(description='AI-based Cybersecurity Anomaly Detection')

# Add argument to run anomaly detection
parser.add_argument('--model', choices=['if', 'kmeans', 'dbscan'], default='if',
                    help='Choose model for anomaly detection')
args = parser.parse_args()

# Run the appropriate model based on user input
if args.model == 'if':
    print("Running Isolation Forest...")
    run_multi_model_anomaly_detection(pd.DataFrame(normal_data))
elif args.model == 'kmeans':
    print("Running K-Means Clustering...")
    run_multi_model_anomaly_detection(pd.DataFrame(normal_data))
elif args.model == 'dbscan':
    print("Running DBSCAN...")
    run_multi_model_anomaly_detection(pd.DataFrame(normal_data))

