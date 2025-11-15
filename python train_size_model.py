import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("size_data.csv")

# Extract features and target
X = df[["Height", "Shoulder_Width", "Chest_Width", "Waist_Width"]]
y = df["Size"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
with open("size_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as size_model.pkl")