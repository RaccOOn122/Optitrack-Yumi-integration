import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import itertools


def load_and_preprocess_data(train_file, test_file, seq_length, stride):
    """
    Load and preprocess the training and test data using a sliding window approach.
    """
    train_df = pd.read_csv(train_file, sep=',')
    test_df = pd.read_csv(test_file, sep=',')
    
    # Strip column names and select feature columns
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    feature_columns = ['Right_X', 'Right_Y', 'Right_Z', 'Left_X', 'Left_Y', 'Left_Z']

    # Normalize the features
    scaler = MinMaxScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    # Encode gesture labels
    label_encoder = LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    test_df['Label'] = label_encoder.transform(test_df['Label'])

    # Convert data to sequences using a sliding window
    def create_sequences(data, seq_length, stride):
        sequences, labels = [], []
        for i in range(0, len(data) - seq_length, stride):
            sequences.append(data[feature_columns].iloc[i:i + seq_length].values)
            labels.append(data['Label'].iloc[i + seq_length - 1])
        return np.array(sequences), np.array(labels)

    X_train, y_train = create_sequences(train_df, seq_length, stride)
    X_test, y_test = create_sequences(test_df, seq_length, stride)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

    return X_train, y_train, X_test, y_test, label_encoder, scaler


def build_lstm_model(input_shape, num_classes, units1, units2, units3, dropout_rate, l2_reg):
    """
    Build an optimized LSTM model for gesture classification.
    """
    model = Sequential([
        LSTM(units1, return_sequences=True, kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(units2, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(units3, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_file = "Training/Gestures/Train/train_gesture_data.csv"
    test_file = "Training/Gestures/Test/test_gesture_data.csv"

    # Define hyperparameter combinations
    seq_lengths = [80, 100, 120, 140,  160]  # 2 values
    strides = [10, 20, 30, 40]  # 2 values
    units_options = [(256, 128, 64)]  # 2 values
    dropout_rates = [0.3, 0.4]  # 2 values
    l2_regs = [0.001, 0.002]  # 2 values

    param_combinations = list(itertools.product(seq_lengths, strides, units_options, dropout_rates, l2_regs))
    best_accuracy = 0
    best_model_path = ""

    for seq_length, stride, (units1, units2, units3), dropout_rate, l2_reg in param_combinations:
        print(f"\nTraining with seq_length={seq_length}, stride={stride}, units={units1, units2, units3}, dropout={dropout_rate}, l2={l2_reg}")

        # Load and preprocess data
        X_train, y_train, X_test, y_test, label_encoder, scaler = load_and_preprocess_data(train_file, test_file, seq_length, stride)
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]

        # Build and train model
        model = build_lstm_model(input_shape, num_classes, units1, units2, units3, dropout_rate, l2_reg)

        # Define model name
        model_name = f"lstm_seq{seq_length}_stride{stride}_units{units1}-{units2}-{units3}_dropout{dropout_rate}_l2{l2_reg}.keras"
        checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)

        # Train model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
                             checkpoint])

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model {model_name} - Test Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_name

    print(f"\nBest model: {best_model_path} with accuracy: {best_accuracy:.4f}")
