import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("Housing.csv")

# Display basic info
print("Initial Data Summary:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# Encode categorical columns manually (added 'prefarea')
categorical_mappings = {
    'mainroad': {'yes': 1, 'no': 0},
    'guestroom': {'yes': 1, 'no': 0},
    'basement': {'yes': 1, 'no': 0},
    'hotwaterheating': {'yes': 1, 'no': 0},
    'airconditioning': {'yes': 1, 'no': 0},
    'prefarea': {'yes': 1, 'no': 0},  # ✅ This was missing before
    'furnishingstatus': {
        'unfurnished': 0,
        'semi-furnished': 1,
        'furnished': 2
    }
}

df.replace(categorical_mappings, inplace=True)

# Confirm encoding
print("\nAfter encoding:\n", df.head())

# Separate features and target
X = df.drop(columns="price")
y = df["price"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print(f"\nModel R² score on test set: {score:.4f}")

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model and scaler saved successfully.")
