# file: train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv('addiction_risk_data.csv')
X = df.drop(columns=['RiskLevel']).values
y = df['RiskLevel'].values

# Convert labels to one-hot encoding
y_encoded = to_categorical(y, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Build model
model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='relu'))  # Increased neurons
model.add(Dropout(0.3))  # Dropout to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))  # 3 classes

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)  # Increased epochs

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2f}")

# Save model
model.save('risk_model.h5')
print("✅ Improved model saved as risk_model.h5")
