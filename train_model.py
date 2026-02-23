import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/employee_data.csv")
print(data.columns)
# Features and target
X = data.drop("performance", axis=1)
y = data["performance"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoder
with open("model/performance_model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("✅ Model trained and saved successfully!")