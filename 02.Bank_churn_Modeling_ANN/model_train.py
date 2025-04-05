import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

print("Starting model training process...")

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Churn_Modelling.csv')

# Drop unnecessary columns
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Create preprocessing objects
scaler = StandardScaler()
geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

# Apply encodings
print("Preprocessing data...")
df['Geography'] = geo_encoder.fit_transform(df['Geography'])
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

# Create mappings for later use
geo_mapping = dict(zip(geo_encoder.classes_, geo_encoder.transform(geo_encoder.classes_)))
gender_mapping = dict(zip(gender_encoder.classes_, gender_encoder.transform(gender_encoder.classes_)))

# Feature and target separation
X = df.drop('Exited', axis=1)
y = to_categorical(df['Exited'])

# Store feature names
feature_names = list(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
print("Building neural network...")
model = Sequential()
model.add(Dense(16, kernel_initializer='normal', activation='relu', input_shape=(10,)))
model.add(Dropout(rate=0.1))
model.add(BatchNormalization())
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dropout(rate=0.1))
model.add(BatchNormalization())
model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
history = model.fit(
    X_train, 
    y_train, 
    validation_data=(X_test, y_test), 
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
print("Saving model...")
model.save("churn_model.h5")

# Save preprocessing objects together
preprocessor = {
    'scaler': scaler,
    'geo_encoder': geo_encoder,
    'gender_encoder': gender_encoder,
    'geo_mapping': geo_mapping,
    'gender_mapping': gender_mapping,
    'feature_names': feature_names
}

with open('churn_preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Generate performance visualization
print("Generating performance visualization...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_performance.png')

print("Training completed!")
print(f"Files saved: churn_model.h5, churn_preprocessor.pkl, model_performance.png")