# ThreatDetectAI: AI-Based Cybersecurity Anomaly Detection

## Project Overview

ThreatDetectAI is an AI-driven cybersecurity project designed to detect anomalies in network traffic data. It leverages multiple machine learning models—**Isolation Forest**, **KMeans**, and **DBSCAN**—to identify potential threats in real-time and logs these events for further analysis.

## Key Features

- **Real-Time Data Simulation:** 
  - Simulates incoming network traffic by generating random data at regular intervals.
  - Continuously monitors and evaluates data for anomalies.

- **Multi-Model Anomaly Detection:** 
  - Implements **Isolation Forest** for tree-based anomaly detection.
  - Uses **KMeans Clustering** to identify outliers via clustering.
  - Applies **DBSCAN** to detect anomalies based on data density.
  
- **Logging & Historical Tracking:** 
  - Automatically logs each detected anomaly with a timestamp in `anomaly_log.txt`.
  - Optionally saves a historical record of anomalies in `history_data.csv` for long-term analysis.

- **Command-Line Interface (CLI):**
  - Easily select which model to run via CLI arguments.
  - Run the script with different models to compare performance and outputs.

- **Model Persistence:**
  - Trained models can be saved (using Pickle) in the `models/` directory for future reuse without retraining.

## Running on a Virtual Machine (VM)

This project is optimized for deployment on a VM. Below are the steps to run ThreatDetectAI on your Azure (or other cloud provider) VM.

### 1. Prepare Your VM

- **Deploy Your VM:**
  - Ensure your VM is up and running (with SSH access enabled).
  - For Linux-based VMs, use SSH from your terminal:
    ```bash
    ssh your_username@your_vm_public_ip
    ```

- **Set Up Your Environment:**
  - Navigate to your project directory on the VM.
  - Activate your virtual environment (if using one):
    ```bash
    source .venv/bin/activate
    ```
  - Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Running the Anomaly Detection Script

ThreatDetectAI is designed to run directly on your VM. Depending on your chosen model, execute one of the following commands from your VM's terminal:

- **Run with Isolation Forest (default):**
  ```bash
  python anomaly_detection.py --model if

- **Run with KMeans Clustering:**
  ```bash
  python anomaly_detection.py --model kmeans

- **Run with DBSCAN:**
  ```bash
  python anomaly_detection.py --model dbscan

 ### 3. Monitoring and Analysis

 - Log Files:
   - Check `anomaly_log.txt` to review all logged anomalies with timestamps.
  
 - Historical Data:
   - If enabled, review `history_data.csv` for a summary of anomalies over time.
  
 - Visualization:
   - The script uses Matplotlib to plot network traffic data, where anomalies are color-coded for easy identification.



   
