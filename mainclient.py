import socket
import json
import os
import pandas as pd
from datetime import datetime

def save_to_csv(data_list, file_name="Gestures/received_data.csv"):
    """
    Save the received data to a CSV file.
    """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Save or append to the CSV file
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False, sep=';')
    else:
        df.to_csv(file_name, index=False, sep=';')
    print(f"Data saved to {file_name}")

def load_csv(file_name):
    """
    Load data from a CSV file, handling empty or missing files.
    """
    if os.path.exists(file_name):
        try:
            return pd.read_csv(file_name, sep=';').to_dict(orient='records')
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_name} is empty. Initializing empty list.")
            return []
    return []  # If file doesn't exist, return an empty list

def preprocess_data(raw_data):
    """
    Parse and clean the raw data for machine learning.
    """
    parsed_data = []
    for entry in raw_data:
        try:
            parsed_data.extend([json.loads(e) for e in entry.split('\n') if e.strip()])
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid data entry: {entry[:50]}...")
    return parsed_data

def start_client(server_ip="10.24.20.226", port=5000):
    """
    Start the client to receive and process data from the server.
    """
    file_name = f"Gestures/received_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print("Connected to server.")

    # Load existing data or initialize an empty list
    data_list = load_csv(file_name)

    try:
        while True:
            data = client_socket.recv(4096).decode("utf-8")
            if data:
                print("Received data:", data[:80], "...")
                # Preprocess and append received data
                new_data = preprocess_data([data])
                data_list.extend(new_data)
                save_to_csv(data_list, file_name)
    except KeyboardInterrupt:
        print("Client shutting down.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    start_client()
