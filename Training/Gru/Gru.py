import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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

def build_gru_model(input_shape, num_classes):
    """
    Build an optimized GRU model for gesture classification.
    """
    model = Sequential([
        GRU(128, return_sequences=True, kernel_regularizer=l2(0.001), input_shape=input_shape),
        Dropout(0.4),
        GRU(64, return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        GRU(32, kernel_regularizer=l2(0.001)),
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
    train_file = "../Post-Processing/train_gesture_data.csv"
    test_file = "../Post-Processing/test_gesture_data.csv"

    # Load and preprocess data
    X_train, y_train, X_test, y_test, label_encoder = load_and_preprocess_data(train_file, test_file)

    # Define model parameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    num_classes = y_train.shape[1]

    # Build the optimized model
    model = build_gru_model(input_shape, num_classes)

    # Callbacks to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    checkpoint = ModelCheckpoint("best_gesture_gru.keras", monitor='val_loss', save_best_only=True)

    # Train with callbacks
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=50, batch_size=64, callbacks=[early_stopping, reduce_lr, checkpoint])

    # Evaluate the best model
    best_model = tf.keras.models.load_model("best_gesture_gru.keras")
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Best Model Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save label encoder
    np.save("gesture_label_encoder.npy", label_encoder.classes_)
    print("Best GRU model and label encoder saved.")
