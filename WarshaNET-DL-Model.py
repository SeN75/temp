import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

# Load data from JSON file
with open('/kaggle/input/cars117/cars117.json', 'r') as f:
    data = json.load(f)

# Extract all unique parts from the JSON data
all_parts = set()
for entry in data.values():
    if 'replaced_parts' in entry:
        all_parts.update(entry['replaced_parts'])
all_parts = sorted(list(all_parts))

# Prepare X and Y data
X_data = []
Y_data = []

for entry in data.values():
    if 'one_hot_vector' in entry and 'replaced_parts' in entry:
        X_data.append(entry['one_hot_vector'])
        Y = [1 if part in entry['replaced_parts'] else 0 for part in all_parts]
        Y_data.append(Y)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Print some information about the data
print(f"Number of samples: {len(X_data)}")
print(f"Input vector size: {len(X_data[0])}")
print(f"Number of unique parts: {len(all_parts)}")

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train.flatten())
class_weight_dict = dict(enumerate(class_weights))

# Define the model
input_dim = X_data.shape[1]
output_dim = len(all_parts)

inputs = Input(shape=(input_dim,))
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(output_dim, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate the model on the test set
Y_pred = model.predict(X_test)
Y_pred_binary = (Y_pred > 0.5).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(Y_test.flatten(), Y_pred_binary.flatten())
precision = precision_score(Y_test.flatten(), Y_pred_binary.flatten(), average='weighted')
recall = recall_score(Y_test.flatten(), Y_pred_binary.flatten(), average='weighted')
f1 = f1_score(Y_test.flatten(), Y_pred_binary.flatten(), average='weighted')

print(f"\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Function to predict parts to be replaced for a new input
def predict_parts_to_replace(model, input_vector, threshold=0.1):
    prediction = model.predict(np.array([input_vector]))
    predicted_parts = [all_parts[i] for i, prob in enumerate(prediction[0]) if prob > threshold]
    return predicted_parts, prediction[0]

# Example usage of the prediction function
print("\nExample prediction for the first test case:")
example_input = X_test[0]
predicted_parts, probabilities = predict_parts_to_replace(model, example_input)
print("Predicted parts to replace:", predicted_parts)
print("Actual parts replaced:", [all_parts[i] for i, val in enumerate(Y_test[0]) if val == 1])

# Print top 5 highest probability parts
top_5_indices = np.argsort(probabilities)[-5:][::-1]
print("\nTop 5 highest probability parts:")
for idx in top_5_indices:
    print(f"{all_parts[idx]}: {probabilities[idx]:.4f}")

# Save the model
model.save('improved_car_damage_prediction_model.h5')
print("\nModel saved as 'improved_car_damage_prediction_model.h5'")