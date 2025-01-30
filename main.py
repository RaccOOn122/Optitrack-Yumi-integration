import sys
import time
import socket
import csv
from NatNetClient import NatNetClient

# Global variables
last_sent_frame = None
reference_marker = [0, 0, 0]  # Reference marker for relative position calculation


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
    Format the raw MoCap data into structured output with relative positioning.
    """
    global reference_marker
    try:
        rigid_body_id_to_name = {
            63: "HandLeft",
            64: "HandRight",
            62: "Head",
        }

        frame_number = data_dict.get("frame_number", None)
        timestamp = data_dict.get("timestamp", None)
        rigid_bodies = data_dict.get("rigid_bodies", [])

        # Set reference marker to the position of the "Head" (ID 62)
        for rb in rigid_bodies:
            if rb.get("id") == 62 and rb.get("tracking_valid", False):
                reference_marker = rb.get("position", [0, 0, 0])
                break

        left_hand = None
        right_hand = None

        for rb in rigid_bodies:
            if rb.get("tracking_valid", False):
                rigid_body_id = rb.get("id", None)
                if rigid_body_id == 63:  # Left Hand
                    left_hand = rb.get("position", [0, 0, 0])
                elif rigid_body_id == 64:  # Right Hand
                    right_hand = rb.get("position", [0, 0, 0])

        if left_hand and right_hand:
            return [[
                frame_number,
                timestamp,
                round(right_hand[0] - reference_marker[0], 4),
                round(right_hand[1] - reference_marker[1], 4),
                round(right_hand[2] - reference_marker[2], 4),
                round(left_hand[0] - reference_marker[0], 4),
                round(left_hand[1] - reference_marker[1], 4),
                round(left_hand[2] - reference_marker[2], 4)
            ]]
        else:
            return []
    except Exception as e:
        print(f"Error formatting data: {e}")
        return []


def receive_new_frame(data_dict):
    """
    Callback function triggered for each new frame of MoCap data.
    Formats the data and streams it to the client as CSV.
    """
    global client_socket, last_sent_frame

    formatted_rows = format_data(data_dict)

    if formatted_rows:
        frame_number = formatted_rows[0][0]
        if frame_number == last_sent_frame:
            return  # Skip duplicate frames
        last_sent_frame = frame_number

    if client_socket:
        try:
            # Convert list of rows into CSV format
            csv_data = "\n".join([";".join(map(str, row)) for row in formatted_rows]) + "\n"
            client_socket.sendall(csv_data.encode("utf-8"))
        except Exception as e:
            print(f"Error sending data: {e}")
    else:
        print("No client connected. Data not sent.")


if __name__ == "__main__":
    optionsDict = {}
    optionsDict["clientAddress"] = "0.0.0.0"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = True

    optionsDict = my_parse_args(sys.argv, optionsDict)

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

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    streaming_client.new_frame_listener = receive_new_frame

    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        server_socket.close()
        sys.exit(1)

    print("\nNatNet Client successfully connected. Listening for data...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if client_socket:
            client_socket.close()
        server_socket.close()
