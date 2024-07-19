import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

features_path = 'numpy/original/features.npy'
labels_path = 'numpy/original/labels.npy'

features = np.load(features_path)
labels = np.load(labels_path)

# Check if features and labels are populated
if len(features) == 0 or len(labels) == 0:
    print("No features or labels were extracted. Please check the data and paths.")
else:
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Initialize the SVM classifier
    clf = svm.SVC(kernel='linear')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

    # Calculate the mean and standard deviation for training and testing scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curves
    plt.figure()
    plt.title("Learning Curve (SVM)")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Plot the mean and standard deviation for training and testing scores
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save the trained model
    joblib.dump(clf, 'svm_license_plate_detector.pkl')

print("SVM model trained and saved successfully.")