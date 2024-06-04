# Iris Flower Classification

This repository contains a Jupyter notebook that performs classification on the Iris flower dataset using Logistic Regression. The notebook demonstrates how to load, explore, and prepare the dataset, train a Logistic Regression model, and evaluate its performance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Project Description](#project-description)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Introduction

The Iris flower dataset is a well-known dataset in the machine learning community. It contains 150 samples of iris flowers, each with four features (sepal length, sepal width, petal length, and petal width) and one of three species (setosa, versicolor, or virginica). This project uses Logistic Regression to classify the species of iris flowers based on their features.

## Installation

To run the Jupyter notebook, you need to have Python and the following libraries installed:

- numpy
- pandas
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas seaborn scikit-learn
```

## Usage

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-github-username/my-notebook-repo.git
   cd my-notebook-repo
   ```

2. **Open the Jupyter notebook:**

   ```bash
   jupyter notebook my_notebook.ipynb
   ```

3. **Run the notebook cells to execute the code.**

## Dataset

The Iris dataset is included in the repository as `IRIS Flower.csv`. The dataset contains the following columns:

- `sepal_length`: Sepal length in cm
- `sepal_width`: Sepal width in cm
- `petal_length`: Petal length in cm
- `petal_width`: Petal width in cm
- `species`: Species of the iris flower (setosa, versicolor, virginica)

## Model Training and Evaluation

The notebook performs the following steps:

1. **Data Loading:**
   - Load the dataset using pandas.

2. **Data Exploration:**
   - Display descriptive statistics and the first few rows of the dataset.

3. **Data Preparation:**
   - Split the dataset into features (X) and target (Y).
   - Split the data into training and test sets using `train_test_split`.

4. **Model Training:**
   - Train a Logistic Regression model using the training data.

5. **Model Evaluation:**
   - Make predictions on the test data.
   - Evaluate the model using accuracy score and classification report.

## Results

The Logistic Regression model is evaluated using the accuracy score and classification report, which provides detailed metrics on precision, recall, and F1-score for each class. The results are printed in the notebook.

## Project Description

### Objectives

1. **Load and explore the dataset** to understand its structure and main characteristics.
2. **Preprocess the data** to prepare it for machine learning algorithms.
3. **Train a Logistic Regression model** using the preprocessed data.
4. **Evaluate the model** to determine its performance and accuracy.
5. **Document the entire process** for clarity and reproducibility.

### Steps Completed

1. **Loading the Dataset**
   - The dataset was loaded into a pandas DataFrame from a CSV file (`IRIS Flower.csv`).
   - Displayed the first few rows and descriptive statistics to understand the data distribution and identify any potential issues.

2. **Exploratory Data Analysis (EDA)**
   - Used pandas' `describe()` method to generate summary statistics.
   - Displayed the first five rows of the dataset using `head()` to verify data loading and gain initial insights.

3. **Data Preparation**
   - Extracted the feature columns (sepal length, sepal width, petal length, and petal width) into `X`.
   - Extracted the target column (species) into `Y`.
   - Split the dataset into training and testing sets using `train_test_split` from scikit-learn, with 80% of the data for training and 20% for testing.

4. **Model Training**
   - Imported `LogisticRegression` from scikit-learn's linear model module.
   - Created an instance of `LogisticRegression` and trained it on the training data (`X_train` and `Y_train`).

5. **Model Evaluation**
   - Made predictions on the test set (`X_test`) using the trained model.
   - Calculated the accuracy of the model using `accuracy_score` from scikit-learn.
   - Generated a classification report using `classification_report` to evaluate the model's precision, recall, and F1-score for each class.

### Code Overview

Here's a brief overview of the code used in this project:

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:\\Users\\sunny\\Untitled Folder\\IRIS Flower.csv', encoding='latin-1')

# Exploratory Data Analysis
print(data.describe())
print(data.head(5))

# Data Preparation
df = data.values
X = df[:, 0:4]
Y = df[:, 4]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the Logistic Regression model
model_LR = LogisticRegression()
model_LR.fit(X_train, Y_train)

# Make predictions
prediction1 = model_LR.predict(X_test)

# Evaluate the model
print(accuracy_score(Y_test, prediction1) * 100)
print(classification_report(Y_test, prediction1))
```

## Future Work

- **Explore more advanced models:** Consider using other machine learning algorithms like Decision Trees, Random Forests, or Support Vector Machines to potentially improve accuracy.
- **Hyperparameter tuning:** Perform grid search or random search to find the optimal hyperparameters for the Logistic Regression model.
- **Feature engineering:** Investigate additional feature transformations or combinations that might improve model performance.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
