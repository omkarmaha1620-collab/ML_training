import pandas as pd
import matplotlib.pyplot as plt

# Load LSTM predictions
FILE = "LSTM/lstm_predictions.csv"

df = pd.read_csv(FILE)

# Convert time column
df["target_time"] = pd.to_datetime(df["target_time"])

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    df["target_time"],
    df["actual_VHM0"],
    label="Actual VHM0"
)

plt.plot(
    df["target_time"],
    df["predicted_VHM0"],
    label="LSTM Predicted VHM0"
)

plt.xlabel("Time")
plt.ylabel("VHM0")
plt.title("LSTM: Actual vs Predicted VHM0")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
plt.savefig("LSTM/lstm_actual_vs_predicted.png", dpi=300)

plt.show()

print("Graph saved successfully!")