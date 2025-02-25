# Contributors: Egemen Alkan

# Import necessary libraries
import argparse  # For parsing command-line arguments
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.utils import resample  # For handling imbalanced datasets
import os  # For file and directory operations
import boto3  # AWS SDK to interact with S3
from sklearn.ensemble import RandomForestClassifier  # RandomForest model
import joblib  # For saving/loading machine learning models
import json  # For handling JSON input/output

# Function to load data from an S3 bucket
def load_data_from_s3(bucket, key):
    """
    Load CSV data from an S3 bucket.
    
    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The key (file path) of the object in the bucket.
    
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    print(f"Loading data from S3 bucket: {bucket}, key: {key}")
    s3 = boto3.client("s3")  # Create an S3 client
    obj = s3.get_object(Bucket=bucket, Key=key)  # Retrieve the specified object
    data = pd.read_csv(obj["Body"])  # Read the object's body as a CSV file
    return data


# Function to preprocess data
def preprocess_data(X, y, min_samples=2):
    """
    Handle target classes with fewer than `min_samples` samples.
    
    Args:
        X (pd.DataFrame): The features.
        y (pd.Series): The target variable.
        min_samples (int): The minimum number of samples per class.
    
    Returns:
        pd.DataFrame, pd.Series: Preprocessed features and target variable.
    """
    print(f"Filtering out classes with fewer than {min_samples} samples...")
    class_counts = y.value_counts()  # Get counts of each class in the target variable

    # Identify valid classes with at least `min_samples` samples
    valid_classes = class_counts[class_counts >= min_samples].index
    X_filtered = X[y.isin(valid_classes)]  # Filter features for valid classes
    y_filtered = y[y.isin(valid_classes)]  # Filter target variable for valid classes

    # Handle cases where the target has less than two classes
    if y_filtered.nunique() < 2:
        print("Warning: Target variable has insufficient classes. Synthesizing new classes...")
        
        if y_filtered.empty:
            # If no valid classes, synthesize a minimal dataset with two classes
            print("Synthesizing a minimal dataset with two classes.")
            n_samples = len(X)
            half = n_samples // 2
            y_synthetic = pd.Series([0] * half + [1] * (n_samples - half), index=X.index)
            X_synthetic = pd.concat([X] * 2).iloc[:n_samples].reset_index(drop=True)
        else:
            # If only one class exists, synthesize another class
            new_class = y_filtered.max() + 1
            synthetic_indices = X_filtered.sample(frac=0.2, replace=True, random_state=42).index
            X_synthetic = X.loc[synthetic_indices]
            y_synthetic = pd.Series([new_class] * len(X_synthetic), index=X_synthetic.index)

        # Combine original and synthetic data
        X_filtered = pd.concat([X_filtered, X_synthetic], ignore_index=True)
        y_filtered = pd.concat([y_filtered, y_synthetic], ignore_index=True)

    print("Target distribution after processing:")
    print(y_filtered.value_counts())
    return X_filtered, y_filtered


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--key", type=str, required=True, help="S3 key for the training dataset")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for preprocessed data")
    args = parser.parse_args()

    # Load the training data from S3
    data = load_data_from_s3(args.bucket, args.key)

    # Separate features and target variable
    y = data["survived"]  # Target variable
    X = data.drop(columns=["survived", "passengerid"])  # Drop target and irrelevant columns

    # Preprocess categorical features
    print("Converting categorical features to numerical values...")
    X["sex"] = X["sex"].map({"male": 1, "female": 0})  # Encode 'sex': male=1, female=0
    X["embarked"] = X["embarked"].map({"C": 0, "Q": 1, "S": 2})  # Encode 'embarked' categories

    # Handle missing values
    if X.isnull().sum().any():
        print("Warning: Dataset contains missing values.")
        X = X.fillna(X.mean())  # Fill missing values with column means

    # Preprocess data to handle imbalanced or small classes
    X, y = preprocess_data(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Stratify to maintain class distribution
    )

    # Train a Random Forest model
    print("Training a RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Determine model saving directory
    if os.path.exists("/opt/ml/model"):
        # Running in SageMaker
        print("Running in SageMaker environment. Saving model to /opt/ml/model...")
        model_dir = "/opt/ml/model"
    else:
        # Running locally
        print("Running locally. Saving model to current working directory...")
        model_dir = os.path.join(args.output_dir, "model")

    # Save the trained model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)  # Save model to a .joblib file
    print(f"Model saved at {model_path}")

    # Save preprocessed training and testing data
    print(f"Saving preprocessed data to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(args.output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(args.output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(args.output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(args.output_dir, "y_test.csv"), index=False)
    print("Data saved.")


# Function for loading the model in SageMaker
def model_fn(model_dir):
    """Load the trained model for inference."""
    print("DEBUG: Loading model...")
    try:
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        print("DEBUG: Model loaded successfully.")
    except Exception as e:
        print(f"DEBUG: Error loading model: {e}")
        raise e
    return model

# Function for processing input data in SageMaker
def input_fn(serialized_input_data, content_type):
    """Deserialize input data."""
    print("DEBUG: Input function called")
    if content_type == "application/json":
        try:
            input_data = json.loads(serialized_input_data)  # Parse JSON data
            column_names = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
            return pd.DataFrame(input_data, columns=column_names)  # Return as DataFrame
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Function for generating predictions
def predict_fn(input_data, model):
    """Perform prediction using the model."""
    print("DEBUG: predict_fn called")
    print("DEBUG: Input data for prediction:")
    print(input_data.head())
    try:
        predictions = model.predict(input_data)
        print("DEBUG: Predictions:")
        print(predictions)
    except Exception as e:
        print(f"DEBUG: Error during prediction: {e}")
        raise e
    return predictions


# Function for formatting output in SageMaker
def output_fn(prediction, accept):
    """Serialize prediction output."""
    if accept == "application/json":
        return json.dumps(prediction.tolist()), accept  # Return JSON output
    else:
        raise ValueError(f"Unsupported accept type: {accept}")



###############################################################################################################

# !python train.py --bucket ea-titanic-bucket --key titanic_cleaned.csv

