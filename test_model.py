import pickle
import numpy as np

# Load model
with open("model/performance_model.pkl", "rb") as f:
    model, le = pickle.load(f)

print("Enter Employee Details:")

attendance = int(input("Attendance: "))
years = int(input("Years at company: "))
training = int(input("Training hours: "))
rating = int(input("Previous rating: "))
overtime = int(input("Overtime hours: "))

# Create input array
data = np.array([[attendance, years, training, rating, overtime]])

# Predict
prediction = model.predict(data)

# Convert numeric label back to text
result = le.inverse_transform(prediction)

print("\nPredicted Performance:", result[0])