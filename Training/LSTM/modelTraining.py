import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show error messages (suppress warnings)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(train_file, test_file):
    """
    Load and preprocess the training and test data.
    """
    # Load datasets
    train_df = pd.read_csv(train_file, sep=';')
    test_df = pd.read_csv(test_file, sep=';')
    
    # Normalize the features
    scaler = MinMaxScaler()
    feature_columns = ['Relative X', 'Relative Y', 'Relative Z']
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    
    # Encode gesture labels
    label_encoder = LabelEncoder()
    train_df['Gesture'] = label_encoder.fit_transform(train_df['Gesture'])
    test_df['Gesture'] = label_encoder.transform(test_df['Gesture'])
    
    # Convert data to sequences
    def create_sequences(df, seq_length=30):
        sequences, labels = [], []
        grouped = df.groupby('Rigid Body Name')
        for _, group in grouped:
            group = group.sort_values(by='Frame Number')
            for i in range(len(group) - seq_length):
                sequences.append(group[feature_columns].iloc[i:i + seq_length].values)
                labels.append(group['Gesture'].iloc[i + seq_length])
        return np.array(sequences), np.array(labels)
    
    X_train, y_train = create_sequences(train_df)
    X_test, y_test = create_sequences(test_df)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))
    
    return X_train, y_train, X_test, y_test, label_encoder

def build_lstm_model(input_shape, num_classes):
    """
    Build an LSTM model for gesture classification.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # File paths
    train_file = "../Post-Processing/train_gesture_data.csv"
    test_file = "../Post-Processing/test_gesture_data.csv"
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test, label_encoder = load_and_preprocess_data(train_file, test_file)
    
    # Define model parameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    num_classes = y_train.shape[1]
    
    # Build and train the model
    model = build_lstm_model(input_shape, num_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Save the model and label encoder
    model.save("gesture_lstm_model.h5")
    np.save("gesture_label_encoder.npy", label_encoder.classes_)
    print("Model and label encoder saved.")
