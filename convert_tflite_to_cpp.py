# -------------------------------------------------------
# convert_tflite_to_cpp.py
# Converts water_quality_model.tflite to model_data.cpp
# -------------------------------------------------------
import numpy as np

# File names
input_file = "water_quality_model.tflite"
scaler_file = "scaler.npz"
output_file = "model_data.cpp"

# Read the binary model file
with open(input_file, "rb") as f:
    data = f.read()

# Convert to C array format
hex_array = ', '.join(f'0x{b:02x}' for b in data)

# Read the scaler parameters
try:
    scaler_data = np.load(scaler_file)
    scaler_mean_str = ', '.join(f"{x}f" for x in scaler_data['mean'])
    scaler_scale_str = ', '.join(f"{x}f" for x in scaler_data['scale'])
    
    scaler_code = f"""
// Scaler parameters for preprocessing raw sensor data
// Formula: scaled_value = (raw_value - mean) / scale
const float g_scaler_mean[] = {{{scaler_mean_str}}};
const float g_scaler_scale[] = {{{scaler_scale_str}}};
"""
except FileNotFoundError:
    scaler_code = "// Scaler parameters not found."

# Generate .cpp content
cpp_code = f"""
#include <cstdint>
{scaler_code}

// TinyML Model
const unsigned char g_water_quality_model[] = {{
{hex_array}
}};
const int g_water_quality_model_len = {len(data)};
"""

# Save as model_data.cpp
with open(output_file, "w") as f:
    f.write(cpp_code)

print(f"✅ Converted {input_file} → {output_file}")
print(f"📦 Model size: {len(data)} bytes")
