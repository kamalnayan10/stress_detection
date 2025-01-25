import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW

SEED = 9
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

DATASET_PATH = "/Users/aroraji/Desktop/DriveDBPaper/predicting-driver-stress-using-deep-learning/scripts/all_drives.csv"
MODEL_SAVE_PATH = "tcn_self_attention_model.h5"
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 3  # Stress levels: 0 (low), 1 (medium), 2 (high)

###############################################################################
# 1. Data Loading & Preprocessing
###############################################################################
def load_and_preprocess_data(csv_path, test_size=0.2):
    """
    Load and preprocess the dataset.
    """
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    X = df.drop(columns=["time", "Drive", "Stress_mean"])
    y = df["Stress_mean"]

    y = y.map({1.0: 0, 3.0: 1, 5.0: 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)

    y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    return X_train, X_test, y_train, y_test

###############################################################################
# 2. TCN + Self-Attention Model
###############################################################################
class TemporalBlock(layers.Layer):
    """
    Temporal Convolutional Block with dilation and residual connection.
    """
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.dropout2 = layers.Dropout(dropout_rate)
        self.residual = layers.Conv1D(filters=filters, kernel_size=1)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        res = self.residual(inputs)
        outputs = self.norm(x + res)
        return outputs


class TCNWithAttention(keras.Model):
    """
    TCN + Self-Attention model for stress recognition.
    """
    def __init__(self, num_classes, num_filters=64, kernel_size=3, num_layers=4, num_heads=4, dropout_rate=0.2):
        super(TCNWithAttention, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Temporal Convolutional Network (TCN)
        self.tcn_layers = [
            TemporalBlock(
                filters=num_filters,
                kernel_size=kernel_size,
                dilation_rate=2 ** i,
                dropout_rate=dropout_rate
            )
            for i in range(num_layers)
        ]

        # Self-Attention Layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_filters)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

        # MLP Classifier
        self.mlp = keras.Sequential([
            layers.Dense(num_filters, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        x = inputs
        for layer in self.tcn_layers:
            x = layer(x)

        attn_output = self.attention(x, x)
        x = self.norm(x + attn_output)

        x = tf.reduce_mean(x, axis=1)

        outputs = self.mlp(x)
        return outputs

###############################################################################
# 3. Model Training & Evaluation
###############################################################################
def train_and_evaluate(model, X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    Train and evaluate the model.
    """
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weights = dict(enumerate(class_weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')

    def cosine_annealing(epoch, lr):
        return float(1e-3 * 0.5 * (1 + np.cos(epoch / 50 * np.pi)))

    lr_scheduler = LearningRateScheduler(cosine_annealing)

    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4, clipvalue=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[model_checkpoint, lr_scheduler],
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return model

###############################################################################
# 4. Save & Load Model
###############################################################################
def save_model(model, path):
    """
    Save the trained model to disk.
    """
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load a trained model from disk.
    """
    model = keras.models.load_model(path)
    print(f"Model loaded from {path}")
    return model

###############################################################################
# 5. Main Function
###############################################################################
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATASET_PATH)

    model = TCNWithAttention(num_classes=NUM_CLASSES)

    trained_model = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    save_model(trained_model, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()