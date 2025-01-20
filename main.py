import sys
import time
import socket
import json
from NatNetClient import NatNetClient

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
    Format the raw MoCap data into a structured dictionary.
    Each rigid body data is represented with frame number and timestamp.
    Replace rigid body IDs with their names.
    """
    try:
        # Example mapping from IDs to names (update this dynamically if needed)
        rigid_body_id_to_name = {
            66: "HandLeft",
            67: "HandRight",
            68: "Head",
        }

        frame_number = data_dict.get("frame_number", None)
        timestamp = data_dict.get("timestamp", None)
        rigid_bodies = data_dict.get("rigid_bodies", [])

        formatted_rows = []
        for rb in rigid_bodies:
            if rb.get("tracking_valid", False):  # Only process valid tracked bodies
                rigid_body_id = rb.get("id", None)
                rigid_body_name = rigid_body_id_to_name.get(rigid_body_id, f"Unknown (ID {rigid_body_id})")
                
                formatted_rows.append({
                    "Frame Number": frame_number,
                    "Timestamp": timestamp,
                    "Rigid Body Name": rigid_body_name,
                    "Position X": rb.get("position", [None, None, None])[0],
                    "Position Y": rb.get("position", [None, None, None])[1],
                    "Position Z": rb.get("position", [None, None, None])[2],
                    "Rotation X": rb.get("rotation", [None, None, None, None])[0],
                    "Rotation Y": rb.get("rotation", [None, None, None, None])[1],
                    "Rotation Z": rb.get("rotation", [None, None, None, None])[2],
                    "Rotation W": rb.get("rotation", [None, None, None, None])[3],
                })
        return formatted_rows

    except Exception as e:
        print(f"Error formatting data: {e}")
        return []

    
def receive_new_frame(data_dict):
    """
    Callback function triggered for each new frame of MoCap data.
    Formats the data and streams it to the client as JSON.
    """
    global client_socket

    # Format the data
    formatted_rows = format_data(data_dict)

    if client_socket:
        try:
            # Send each row as a separate JSON object
            for row in formatted_rows:
                json_data = json.dumps(row)
                client_socket.sendall(json_data.encode("utf-8") + b'\n')  # Add newline for easy parsing
        except Exception as e:
            print(f"Error sending data: {e}")
    else:
        print("No client connected. Data not sent.")


# Main script
if __name__ == "__main__":
    optionsDict = {}
    optionsDict["clientAddress"] = "0.0.0.0"
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

    # Start the NatNet client
    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        server_socket.close()
        sys.exit(1)

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
