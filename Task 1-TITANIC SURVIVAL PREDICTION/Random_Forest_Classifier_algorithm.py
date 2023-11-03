import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_titanic_train_data():
    # Load the titanic_trainset
    titanic_train = pd.read_csv('Task 1-TITANIC SURVIVAL PREDICTION/DATASET/train.csv')

    # Display basic information about the titanic_trainset
    print(titanic_train.info())

    # Display the first few rows of the titanic_trainset
    print(titanic_train.head())

    # Visualize some basic statistics
    print(titanic_train.describe())

    # Visualize missing data
    plt.figure(figsize=(10, 6))
    sns.heatmap(titanic_train.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing data")
    plt.show()

    return titanic_train

def handling_missing_values(titanic_train):
    # Handling missing values
    titanic_train.drop("Name", axis=1, inplace=True)
    titanic_train.drop(["Cabin", "Ticket"], axis=1, inplace=True)
    titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace=True)
    titanic_train["Embarked"].fillna(titanic_train["Embarked"].mode()[0], inplace=True)
    titanic_train = pd.get_dummies(titanic_train, columns=["Sex", "Embarked"], drop_first=True)

    # Create the target variable (Survived)
    X = titanic_train.drop(["Survived", "PassengerId"], axis=1)
    Y = titanic_train["Survived"]

    return X, Y

def split_datasets(X, Y):
    # Split the titanic_train into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return model

# Add the predict_survival function
def predict_survival(model, new_passenger_data):
    # Preprocess the new passenger data
    new_passenger_data = pd.DataFrame(new_passenger_data, index=[0])

    # Make sure the column names match the ones used during preprocessing
    new_passenger_data.rename(columns={'Sex_male': 'Sex_male', 'Embarked_Q': 'Embarked_Q', 'Embarked_S': 'Embarked_S'}, inplace=True)

    # Use the model to predict survival
    prediction = model.predict(new_passenger_data)

    return prediction

# Main function
def main():
    # Load and explore the Titanic train dataset
    titanic_train = load_titanic_train_data()

    # Handle missing values and preprocess the dataset, then split and train the model
    X, Y = handling_missing_values(titanic_train)
    model = split_datasets(X, Y)

    # Define new passenger data as a dictionary
    new_passenger_data = {
        "Pclass": 1,
        "Age": 35,
        "SibSp": 1,
        "Parch": 2,
        "Fare": 100,
        "Sex_male": 1,  # Male passenger
        "Embarked_Q": 0,  # Not from Queenstown
        "Embarked_S": 1,  # From Southampton
    }

    # Use the predict_survival function to predict survival for the new passenger
    prediction = predict_survival(model, new_passenger_data)
    if prediction == 1:
        print("The new passenger is predicted to survive.")
    else:
        print("The new passenger is predicted not to survive")

if __name__ == "__main__":
    main()
