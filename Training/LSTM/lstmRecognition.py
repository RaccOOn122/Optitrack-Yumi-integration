import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def load_model_and_encoders(model_path="output1/best_gesture_lstm.keras", 
                            label_encoder_path="output1/gesture_label_encoder.npy", 
                            scaler_path="output1/gesture_scaler.npy"):
    """
    Load the trained LSTM model, label encoder, and scaler.
    """
    model = tf.keras.models.load_model(model_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    scaler = np.load(scaler_path, allow_pickle=True).item()
    
    return model, label_encoder, scaler

def preprocess_input_data(data, seq_length=150, feature_columns=['Right_X', 'Right_Y', 'Right_Z', 'Left_X', 'Left_Y', 'Left_Z'], scaler=None):
    """
    Preprocess a single gesture input from CSV file or live stream.
    """
    if isinstance(data, str) and os.path.exists(data):
        data = pd.read_csv(data, sep=';')
    elif isinstance(data, pd.DataFrame):
        data = data.copy()
    else:
        raise ValueError("Input data must be a CSV file path or a Pandas DataFrame")
    
    if scaler:
        data[feature_columns] = scaler.transform(data[feature_columns])
    
    # Create a sequence and pad if necessary
    input_sequence = [data[feature_columns].values]
    input_sequence = pad_sequences(input_sequence, maxlen=seq_length, dtype='float32', padding='post', truncating='post')
    
    return np.array(input_sequence)

def recognize_gesture(input_data):
    """
    Recognize the gesture from input data and return the gesture name with confidence score.
    """
    model, label_encoder, scaler = load_model_and_encoders()
    processed_data = preprocess_input_data(input_data, scaler=scaler)
    
    # Predict gesture
    predictions = model.predict(processed_data)
    predicted_label_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
    
    return predicted_label, confidence

if __name__ == "__main__":
    test_file = "../Gestures/Test/salut/received_data_20250131_165627.csv"  # Example test file
    recognized_gesture, confidence_score = recognize_gesture(test_file)
    print(f"Recognized Gesture: {recognized_gesture} (Confidence: {confidence_score:.2f})")
