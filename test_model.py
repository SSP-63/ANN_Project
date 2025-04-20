import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("risk_model.h5")

# Create all zeros input (all "Never" responses)
all_zeros = np.zeros((1, 10))

# Make prediction
prediction = model.predict(all_zeros)
risk = np.argmax(prediction)

print("Prediction probabilities:", prediction)
print(f"Predicted risk level: {risk}")
print("Risk levels: 0=Low, 1=Moderate, 2=High")