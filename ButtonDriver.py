import serial
import pyautogui

def perform_action_based_on_input(data):
    """
    Perform an action based on the received serial data.
    
    Parameters:
        data (str): The received data from the serial port.
    """
    if data == "5":
        print("Simulating 'Enter' key press.")
        pyautogui.press("enter")
    elif data == "1":
        print("Simulating 'Ctrl+S'.")
        pyautogui.hotkey("ctrl", "s")
    else:
        print(f"No action mapped for: {data}")


def read_serial_and_act(port, baudrate=9600, timeout=1):
    """
    Reads data from a USB serial port and performs actions based on the received input.
    
    Parameters:
        port (str): The USB port name (e.g., 'COM3').
        baudrate (int): The communication speed (default is 9600).
        timeout (float): Timeout for reading in seconds (default is 1 second).
    """
    try:
        # Open the serial port
        with serial.Serial(port, baudrate, timeout=timeout) as ser:
            print(f"Connected to {port} at {baudrate} baud.")
            
            while True:
                # Read a line from the serial port
                data = ser.readline().decode('utf-8').strip()
                if data:
                    print(f"Received: {data}")
                    perform_action_based_on_input(data)
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")

# Replace with the correct port for your device
read_serial_and_act(port='COM13')  # For Windows
