import socket
import os
import pandas as pd
from datetime import datetime

def save_to_csv(data_list, file_name="Gestures/received_data.csv"):
    """
    Save the received data to a CSV file.
    """
    df = pd.DataFrame(data_list, columns=[
        "Frame Number", "Timestamp", "Right Hand X", "Right Hand Y", "Right Hand Z", "Left Hand X", "Left Hand Y", "Left Hand Z"
    ])
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False, sep=';')
    else:
        df.to_csv(file_name, index=False, sep=';')
    print(f"Data saved to {file_name}")

def preprocess_data(raw_data):
    """
    Parse and clean the raw data from CSV format.
    """
    parsed_data = []
    for entry in raw_data:
        try:
            lines = entry.strip().split('\n')
            for line in lines:
                values = line.split(';')
                if len(values) == 8:  # Ensure correct number of columns
                    parsed_data.append(values)
                else:
                    print(f"Warning: Skipping invalid data entry: {line[:50]}...")
        except Exception as e:
            print(f"Error processing data: {e}")
    return parsed_data

def start_client(server_ip="10.24.20.218", port=5000):
    """
    Start the client to receive and process data from the server.
    """
    file_name = f"Gestures/received_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print("Connected to server.")

    try:
        while True:
            data = client_socket.recv(4096).decode("utf-8")
            if data:
                print("Received data:", data[:180], "...")
                new_data = preprocess_data([data])
                if new_data:
                    save_to_csv(new_data, file_name)
    except KeyboardInterrupt:
        print("Client shutting down.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    start_client()