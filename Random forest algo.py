import pyttsx3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = {
    'Runs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Voltage': [1.5, 1.6, 1.7, 1.8, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
    'Current': [0.09, 0.33, 0.73, 1.6, 1.73, 2.43, 2.6, 3.75, 4.29, 4.5, 5.76],
    'Hydrogen Production Rate': [0.65, 2.5, 5.5, 12, 15, 20, 25, 30, 35, 55, 60]
}

df = pd.DataFrame(data)

# Calculate power consumed
df['Power Consumed'] = df['Voltage'] * df['Current']

# Split the data into features and target variables
X = df[['Voltage', 'Current']]
y = df['Hydrogen Production Rate']

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Train the model
rf_regressor.fit(X, y)

# Generate all combinations of voltage and current from the dataset
existing_combinations = df[['Voltage', 'Current']]

# Predict hydrogen production rates for all combinations
predictions = rf_regressor.predict(existing_combinations)

# Print all combinations, power consumed, and predicted hydrogen production rates
print("Existing combinations, power consumed, and predicted hydrogen production rates:")
for i, row in existing_combinations.iterrows():
    power_consumed = row['Voltage'] * row['Current']
    print(f"Voltage: {row['Voltage']}, Current: {row['Current']}, Power Consumed: {power_consumed:.2f} -> Predicted Hydrogen Production Rate: {predictions[i]}")

# Find the index of the highest predicted hydrogen production rate
best_index = np.argmax(predictions)
best_combination = existing_combinations.iloc[best_index]
best_prediction = predictions[best_index]

# Print the best combination, power consumed, and its predicted hydrogen production rate
best_power_consumed = best_combination['Voltage'] * best_combination['Current']
print("\nBest predicted result:")
print(f"Best Combination - Voltage: {best_combination['Voltage']}, Current: {best_combination['Current']}, Power Consumed: {best_power_consumed:.2f}")
print(f"Predicted Hydrogen Production Rate: {best_prediction}")

# Speak the result
engine = pyttsx3.init()
engine.say(f"The best combination is Voltage: {best_combination['Voltage']}V, Current: {best_combination['Current']}A, Power Consumed: {best_combination['Voltage']*best_combination['Current']:.2f}, Predicted Hydrogen Production Rate: {best_prediction}")
engine.runAndWait()

# Plotting the data
fig = plt.figure(figsize=(12, 6))

# Plot 2D graph
plt.subplot(1, 2, 1)
plt.scatter(df['Voltage'], df['Current'], c=df['Hydrogen Production Rate'], cmap='viridis')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Hydrogen Production Rate vs Voltage & Current')
plt.colorbar(label='Hydrogen Production Rate')

# Plot 3D graph
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(df['Voltage'], df['Current'], df['Hydrogen Production Rate'], c=df['Hydrogen Production Rate'], cmap='viridis')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (A)')
ax.set_zlabel('Hydrogen Production Rate (ml/min)')
ax.set_title('Hydrogen Production Rate vs Voltage & Current')
plt.show()
