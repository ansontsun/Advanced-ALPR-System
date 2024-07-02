from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load your dataset
features = np.load('features.npy')
labels = np.load('labels.npy')

# Print the unique labels and their counts
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"Unique labels: {unique_labels}")
print(f"Counts: {counts}")

# Check if there is more than one unique class
if len(unique_labels) <= 1:
    raise ValueError(f"The number of classes has to be greater than one; got {len(unique_labels)} class")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')