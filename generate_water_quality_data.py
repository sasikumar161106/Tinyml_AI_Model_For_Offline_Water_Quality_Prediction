# -------------------------------------------------------------
# generate_water_quality_data.py
# Synthetic dataset for Offline AI Water Quality Prediction
# -------------------------------------------------------------
import pandas as pd
import numpy as np

# number of samples
n = 100

# random seed for reproducibility
np.random.seed(42)

# generate random realistic sensor values
pH = np.random.uniform(4.5, 9.0, n).round(2)
TDS = np.random.uniform(100, 1000, n).round(0)
Turbidity = np.random.uniform(1, 100, n).round(1)
WaterTemp = np.random.uniform(20, 35, n).round(1)
AirTemp = np.random.uniform(25, 38, n).round(1)
Humidity = np.random.uniform(40, 90, n).round(1)
Rain = np.random.choice([0, 1], size=n, p=[0.6, 0.4])  # 60% dry, 40% rainy

# function to assign label based on conditions
def label_row(ph, tds, turb):
    if (6.5 <= ph <= 8.5) and (tds < 300) and (turb < 5):
        return "Good"
    elif (5.5 <= ph <= 9.0) and (tds < 600) and (turb < 50):
        return "Moderate"
    else:
        return "Poor"

Label = [label_row(p, t, tu) for p, t, tu in zip(pH, TDS, Turbidity)]

# combine all into dataframe
df = pd.DataFrame({
    "pH": pH,
    "TDS": TDS,
    "Turbidity": Turbidity,
    "WaterTemp": WaterTemp,
    "AirTemp": AirTemp,
    "Humidity": Humidity,
    "Rain": Rain,
    "Label": Label
})

# preview top rows
print(df.head())

# save to CSV
df.to_csv("water_quality_data.csv", index=False)
print("\n✅ Dataset saved successfully as 'water_quality_data.csv'")
