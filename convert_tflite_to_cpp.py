# -------------------------------------------------------
# convert_tflite_to_cpp.py
# Converts water_quality_model.tflite to model_data.cpp
# -------------------------------------------------------
import numpy as np

# File names
input_file = "water_quality_model.tflite"
output_file = "model_data.cpp"

# Read the binary model file
with open(input_file, "rb") as f:
    data = f.read()

# Convert to C array format
hex_array = ', '.join(f'0x{b:02x}' for b in data)

# Generate .cpp content
cpp_code = f"""
#include <cstdint>
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
