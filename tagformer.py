import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ['PYTHONHASHSEED'] = str(9)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DATASET_PATH = "/Users/aroraji/Desktop/DriveDBPaper/predicting-driver-stress-using-deep-learning/scripts/all_drives.csv"

def get_train_test_data(path, test_drives=["Drive15", "Drive16"]):
    data = pd.read_csv(path)
    data = data.dropna()
    X_train = data[~data["Drive"].isin(test_drives)]
    X_test = data[data["Drive"].isin(test_drives)]
    y_train = X_train["Stress_mean"]
    y_test = X_test["Stress_mean"]
    X_train = X_train.drop(["time", "Drive", "Stress_mean"], axis=1)
    X_test = X_test.drop(["time", "Drive", "Stress_mean"], axis=1)
    return X_train, y_train, X_test, y_test

def positional_encoding(positions, d_model):
    angle_rads = np.arange(positions)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class TAGformer(keras.Model):
    def __init__(self, num_nodes, num_features, num_classes, hidden_dim=128, num_heads=8, num_layers=4):
        super(TAGformer, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.graph_conv = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))

        self.positional_encoding = positional_encoding(num_nodes, hidden_dim)

        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(hidden_dim, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.1)
            ])

        self.feature_fusion = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))

        self.mlp = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        x = self.graph_conv(inputs)

        x += self.positional_encoding[:, :self.num_nodes, :]

        for layer in self.transformer_layers:
            attn_output = layer[0](x, x)
            x = layer[1](x + attn_output)
            x = layer[2](x)
            x = layer[3](x)

        x = self.feature_fusion(x)

        x = tf.reduce_mean(x, axis=1)

        outputs = self.mlp(x)
        return outputs

def preprocess_data_for_tagformer(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)

    # Map stress levels (1, 3, 5) to (0, 1, 2)
    y_train = y_train.map({1: 0, 3: 1, 5: 2})
    y_test = y_test.map({1: 0, 3: 1, 5: 2})

    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)

    return X_train, X_test, y_train, y_test

def train_and_evaluate_tagformer(X_train, X_test, y_train, y_test):
    num_nodes = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes = y_train.shape[1]

    model = TAGformer(num_nodes=num_nodes, num_features=num_features, num_classes=num_classes)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def lr_scheduler(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[lr_callback], verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

def main():
    X_train, y_train, X_test, y_test = get_train_test_data(DATASET_PATH)

    X_train, X_test, y_train, y_test = preprocess_data_for_tagformer(X_train, X_test, y_train, y_test)

    train_and_evaluate_tagformer(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()