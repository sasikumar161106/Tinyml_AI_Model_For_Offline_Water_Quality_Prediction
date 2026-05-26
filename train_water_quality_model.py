# ----------------------------------------------------------
# train_water_quality_model.py
# Trains an AI model to predict water quality from dataset
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Load the dataset
df = pd.read_csv("water_quality_data.csv")
print("Dataset loaded ✅")
print(df.head())

# 2️⃣ Prepare input (X) and output (y)
X = df[['pH', 'TDS', 'Turbidity', 'WaterTemp', 'AirTemp', 'Humidity', 'Rain']].values
y_raw = df['Label'].values

# Encode labels as 0/1/2
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Labels encoded:", list(le.classes_))

# 3️⃣ Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4️⃣ Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Build a tiny neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 6️⃣ Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# 7️⃣ Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")

# 8️⃣ Convert to TensorFlow Lite format (.tflite) with Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimization for smaller model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    # Provide a few examples for the converter to calibrate quantization
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [tf.cast(input_value, tf.float32)]
        
converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()
open("water_quality_model.tflite", "wb").write(tflite_model)
print("\n💾 Model saved as 'water_quality_model.tflite' (Quantized)")

# 9️⃣ Save Scaler Parameters
np.savez("scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
print("💾 Scaler parameters saved as 'scaler.npz'")
