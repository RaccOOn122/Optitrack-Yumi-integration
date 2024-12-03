import sys
import time
import socket
import json
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData


def my_parse_args(arg_list, args_dict):
    """
    Parse command-line arguments to configure client/server addresses and multicast/unicast mode.
    """
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict


def format_data(data_dict):
    """
    Format the raw MoCap data into a more specific structure:
    - Model Name
    - Marker Count
    - Individual Markers
    - Rotation for Model D3
    """
    try:
        formatted_data = {
            "model_name": None,  # Will be filled with model name
            "marker_count": None,  # Will be filled with marker count
            "markers": [],  # List to hold markers
            "rotation": None,  # Rotation data for model
        }

        # Iterate through marker sets and extract relevant data
        for marker_set in data_dict.get("marker_sets", []):
            if marker_set.get("name") == "D3":
                formatted_data["model_name"] = marker_set.get("name")
                formatted_data["marker_count"] = marker_set.get("marker_count")
                
                # Assuming 3 markers for model D3
                for i in range(min(3, len(marker_set.get("markers", [])))):
                    formatted_data["markers"].append(marker_set["markers"][i])
                
                # Add rotation if available
                formatted_data["rotation"] = marker_set.get("rotation")

        return formatted_data

    except Exception as e:
        print(f"Error formatting data: {e}")
        return {}



def receive_new_frame(data_dict):
    """
    Callback function triggered for each new frame of MoCap data.
    Formats the data and sends it over the network if the socket is connected.
    """
    global client_socket

    # print(f"Raw frame data: {data_dict}")  # Debug statement


    formatted = format_data(data_dict)

    # print(f"formatted frame data: {formatted}")  # Debug statement

    if client_socket:
        try:
            json_data = json.dumps(data_dict)
            client_socket.sendall(json_data.encode("utf-8"))
        except Exception as e:
            print(f"Error sending data: {e}")
    else:
        print("No client connected. Data not sent.")


def receive_rigid_body_frame(new_id, position, rotation):
    """
    Callback function triggered for each rigid body in the MoCap data.
    Currently unused but available for future expansion.
    """
    pass


def print_configuration(natnet_client):
    """
    Print the current NatNet client configuration.
    """
    print("Connection Configuration:")
    print("  Client:          %s" % natnet_client.local_ip_address)
    print("  Server:          %s" % natnet_client.server_ip_address)
    print("  Command Port:    %d" % natnet_client.command_port)
    print("  Data Port:       %d" % natnet_client.data_port)


# Global variable for socket connection
client_socket = None

if __name__ == "__main__":
    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = True

    # Parse arguments from command line
    optionsDict = my_parse_args(sys.argv, optionsDict)

    # Start the TCP socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((optionsDict["clientAddress"], 5000))
    server_socket.listen(1)
    print(f"Socket server listening on {optionsDict['clientAddress']}:5000")

    try:
        client_socket, address = server_socket.accept()
        print(f"Connection established with {address}")
    except KeyboardInterrupt:
        print("Socket server shutting down...")
        server_socket.close()
        sys.exit(0)

    # Set up the NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Assign callback functions
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start the NatNet client
    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        server_socket.close()
        sys.exit(1)

    # Check connection status
    time.sleep(1)
    if not streaming_client.connected():
        print("ERROR: Could not connect properly. Ensure Motive streaming is enabled.")
        server_socket.close()
        sys.exit(2)

    print_configuration(streaming_client)
    print("\nNatNet Client successfully connected. Listening for data...")

    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if client_socket:
            client_socket.close()
        server_socket.close()
