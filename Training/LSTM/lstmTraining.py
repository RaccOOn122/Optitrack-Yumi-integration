import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(train_file, test_file, seq_length=30):
    """
    Load and preprocess the training and test data.
    """
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Define feature columns
    feature_columns = ['Right_X', 'Right_Y', 'Right_Z', 'Left_X', 'Left_Y', 'Left_Z']

    # Normalize the features
    scaler = MinMaxScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    # Encode gesture labels
    label_encoder = LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    test_df['Label'] = label_encoder.transform(test_df['Label'])

    # Convert data to sequences
    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[feature_columns].iloc[i:i + seq_length].values)
            labels.append(data['Label'].iloc[i + seq_length])
        return np.array(sequences), np.array(labels)

    X_train, y_train = create_sequences(train_df, seq_length)
    X_test, y_test = create_sequences(test_df, seq_length)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

    return X_train, y_train, X_test, y_test, label_encoder, scaler

def build_lstm_model(input_shape, num_classes):
    """
    Build an optimized LSTM model for gesture classification.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001), input_shape=input_shape),
        Dropout(0.4),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # File paths
    train_file = "../Gestures/Train/train_gesture_data.csv"
    test_file = "../Gestures/Test/test_gesture_data.csv"

    # Load and preprocess data
    seq_length = 150  # Define sequence length
    X_train, y_train, X_test, y_test, label_encoder, scaler = load_and_preprocess_data(train_file, test_file, seq_length)

    # Define model parameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    num_classes = y_train.shape[1]

    # Build the optimized model
    model = build_lstm_model(input_shape, num_classes)

    # Callbacks to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    checkpoint = ModelCheckpoint("best_gesture_lstm.keras", monitor='val_loss', save_best_only=True)

    # Train with callbacks
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=50, batch_size=64, callbacks=[early_stopping, reduce_lr, checkpoint])

    # Evaluate the best model
    best_model = tf.keras.models.load_model("best_gesture_lstm.keras")
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Best Model Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save label encoder and scaler
    np.save("gesture_label_encoder.npy", label_encoder.classes_)
    np.save("gesture_scaler.npy", scaler)
    print("Best model, label encoder, and scaler saved.")
