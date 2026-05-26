# TinyML AI Model For Offline Water Quality Prediction

This repository provides a complete pipeline to generate synthetic water quality data, train an artificial neural network using TensorFlow, and convert the resulting model into a C++ format for TinyML applications on microcontrollers. 

## Features
- **Synthetic Data Generation**: Creates realistic data mimicking sensor outputs for pH, TDS, Turbidity, Water Temperature, Air Temperature, Humidity, and Rain. Labels the water quality as `Good`, `Moderate`, or `Poor`.
- **Tiny Neural Network**: A small-footprint Keras Sequential model that is trained to predict the water quality label from the 7 input parameters.
- **TFLite Conversion**: Converts the trained model into a TensorFlow Lite format (`.tflite`) for efficient edge inference.
- **C++ Header Export**: Exports the `.tflite` model into a byte array in a `.cpp` file for immediate compilation and deployment on microcontrollers like Arduino, ESP32, or STM32.

## Project Structure
- `generate_water_quality_data.py`: Script to generate the synthetic `water_quality_data.csv` dataset.
- `train_water_quality_model.py`: Script to load the dataset, train the AI model, and save it as `water_quality_model.tflite`.
- `convert_tflite_to_cpp.py`: Script to convert the `.tflite` model to a C++ array format (`model_data.cpp`).
- `water_quality_data.csv`: The generated synthetic dataset.
- `water_quality_model.tflite`: The exported TensorFlow Lite model.
- `model_data.cpp`: The final C++ array file for offline prediction on MCUs.

## Usage

### 1. Requirements
Install the necessary Python dependencies:
```bash
pip install pandas numpy tensorflow scikit-learn
```

### 2. Generate Data
Run the following script to create a synthetic dataset (`water_quality_data.csv`):
```bash
python generate_water_quality_data.py
```

### 3. Train the Model
Run the training script. This script trains a neural network on the dataset and exports a `water_quality_model.tflite` file:
```bash
python train_water_quality_model.py
```

### 4. Convert for Microcontrollers
Convert the TensorFlow Lite model into a C++ file (`model_data.cpp`):
```bash
python convert_tflite_to_cpp.py
```

The resulting `model_data.cpp` contains a `g_water_quality_model` byte array and its length, which can be directly included in your C/C++ projects for embedded systems.
