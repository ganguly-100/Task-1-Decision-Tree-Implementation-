iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to a Pandas DataFrame for easier inspection (optional, but good practice)
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
print("--- First 5 rows of the Iris dataset ---")
print(df.head())
print("\n")


# --- 2. Split Data into Training and Testing Sets ---
# We split the data to train the model on one subset and test its performance on another, unseen subset.
# test_size=0.3 means 30% of the data will be used for testing.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("\n")


# --- 3. Build and Train the Decision Tree Model ---
# We create an instance of the DecisionTreeClassifier.
# `criterion='entropy'` is a measure of information gain used to decide the best feature to split on.
# `max_depth=3` limits the tree's depth to prevent overfitting and make it easier to visualize.
# `random_state` is for reproducibility of the model's 'decisions'.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

# Train the classifier on the training data
print("--- Training the Decision Tree Classifier ---")
clf.fit(X_train, y_train)
print("Model training complete.")
print("\n")


# --- 4. Evaluate the Model ---
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"--- Model Evaluation ---")
print(f"Accuracy on the test set: {accuracy:.2f}")
print("\n")


# --- 5. Visualize the Decision Tree ---
# Scikit-learn can export the tree in a format called .dot.
# Graphviz is a tool for visualizing this .dot format.
# You might need to install it:
# pip install graphviz
# And also install the graphviz software from: https://graphviz.org/download/

print("--- Generating Decision Tree Visualizations ---")

# Visualization 1: Using graphviz library
# This creates a more detailed and customizable visualization.
dot_data = export_graphviz(clf,
                           out_file=None,
                           feature_names=feature_names,
                           class_names=target_names,
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
# This will save the tree as 'iris_decision_tree.pdf' and open a preview
try:
    graph.render("iris_decision_tree", format="png", view=False, cleanup=True)
    print("Saved Graphviz visualization to 'iris_decision_tree.png'")
except graphviz.backend.execute.ExecutableNotFound:
    print("Graphviz executable not found. Please install it and ensure it's in your PATH.")
    print("Skipping Graphviz visualization.")


# Visualization 2: Using Matplotlib (built-in, no extra installation needed)
# This is a simpler, quicker way to see the tree.
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=300)
tree.plot_tree(clf,
               feature_names=feature_names,
               class_names=target_names,
               filled=True,
               rounded=True)

# Save the plot to a file# main.py
# This script demonstrates how to build, train, and visualize a Decision Tree Classifier
# using the scikit-learn library and the classic Iris dataset.

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
import matplotlib.pyplot as plt
from sklearn import tree

# --- 1. Load and Prepare the Dataset ---
# We're using the Iris dataset, a classic dataset in machine learning.
# It contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
# Features: Sepal Length, Sepal Width, Petal Length, Petal Width
# Target: Iris species (Setosa, Versicolour, Virginica)

plt.title("Decision Tree for Iris Classification (Matplotlib)")
plt.savefig('iris_decision_tree_matplotlib.png')
print("Saved Matplotlib visualization to 'iris_decision_tree_matplotlib.png'")
print("\n--- Script Finished ---")

# To run this script:
# 1. Make sure you have pandas, scikit-learn, matplotlib, and graphviz installed:
#    pip install pandas scikit-learn matplotlib graphviz
# 2. For the Graphviz visualization, install the Graphviz software from their official website.
# 3. Save the code as a Python file (e.g., main.py) and run `python main.py` from your terminal.
