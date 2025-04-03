import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import itertools


def load_and_preprocess_data(train_file, test_file, seq_length, stride):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    feature_columns = ['Right_X', 'Right_Y', 'Right_Z', 'Left_X', 'Left_Y', 'Left_Z']
    
    scaler = MinMaxScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    
    label_encoder = LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    test_df['Label'] = label_encoder.transform(test_df['Label'])
    
    def create_sequences(data, seq_length, stride):
        sequences, labels = [], []
        for i in range(0, len(data) - seq_length, stride):
            sequences.append(data[feature_columns].iloc[i:i + seq_length].values)
            labels.append(data['Label'].iloc[i + seq_length - 1])
        return np.array(sequences), np.array(labels)
    
    X_train, y_train = create_sequences(train_df, seq_length, stride)
    X_test, y_test = create_sequences(test_df, seq_length, stride)
    
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))
    
    return X_train, y_train, X_test, y_test, label_encoder, scaler


def build_gru_model(input_shape, num_classes, units1, units2, units3, dropout_rate, l2_reg):
    model = Sequential([
        GRU(units1, return_sequences=True, kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        GRU(units2, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        GRU(units3, kernel_regularizer=l2(l2_reg)),
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
    
    seq_lengths = [80, 100, 120, 140, 160]
    strides = [10, 20, 30, 40]
    units_options = [(256, 128, 64), (128, 64, 32)]
    dropout_rates = [0.3, 0.4, 0.5]
    l2_regs = [ 0.001, 0.002]

    param_combinations = list(itertools.product(seq_lengths, strides, units_options, dropout_rates, l2_regs))
    best_accuracy = 0
    best_model_path = ""
    
    for seq_length, stride, (units1, units2, units3), dropout_rate, l2_reg in param_combinations:
        
        X_train, y_train, X_test, y_test, label_encoder, scaler = load_and_preprocess_data(train_file,
            test_file, seq_length, stride)
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        model = build_gru_model(input_shape, num_classes, units1, units2, units3, dropout_rate, l2_reg)
        
        model_name = f"gru_seq{seq_length}_stride{stride}_units{units1}-{units2}-{units3}_dropout{dropout_rate}_l2{l2_reg}.keras"
        checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)
        
        print(f"Training with seq_length={seq_length}, stride={stride}, units={units1, units2, units3}, dropout={dropout_rate}, l2={l2_reg}")
        
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
                             checkpoint])
        
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model {model_name} - Test Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_name
    
    print(f"Best model: {best_model_path} with accuracy: {best_accuracy:.4f}")
