import socket
import json
import os

def save_to_json(data_list, file_name="received_data.json"):
    # Save the received data to a JSON file
    with open(file_name, "w") as json_file:
        json.dump(data_list, json_file, indent=4)
    print(f"Data saved to {file_name}")

def load_data(file_name):
    # Load data from a JSON file, handling empty or invalid files
    if os.path.exists(file_name):
        try:
            with open(file_name, "r") as json_file:
                return json.load(json_file)
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: {file_name} is empty or contains invalid JSON. Initializing empty list.")
            return []
    return []  # If file doesn't exist, return an empty list

def start_client(server_ip="127.0.0.1", port=5000, file_name="received_data.json"):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print("Connected to server.")

    # Load data from file or initialize an empty list
    data_list = load_data(file_name)

    try:
        while True:
            data = client_socket.recv(4096).decode("utf-8")
            if data:
                print("Received data:", data)
                data_list.append(data)  # Append received data to the list
                save_to_json(data_list, file_name)  # Save updated list to JSON
    except KeyboardInterrupt:
        print("Client shutting down.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    start_client()
