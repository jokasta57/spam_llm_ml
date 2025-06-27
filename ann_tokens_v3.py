'''
I'll create a comprehensive implementation of an Artificial Neural Network (ANN) with two layers that reads from a CSV file, uses an 80% training/20% validation split, Adam optimization, and plots the cross-validation error.
ANN with Two Layers and Cross-ValidationCode 
This code implements a complete ANN solution with the specifications you requested. Here's a breakdown of what it does:
Key Features:

Data Processing:

Loads data from a CSV file
Handles missing values automatically
Encodes categorical variables
Scales numerical features using StandardScaler
Implements an 80% training/20% validation split


ANN Architecture:

Two-layer neural network (configurable nodes per layer)
ReLU activation for hidden layers
Adam optimization with customizable learning rate
Mean Squared Error (MSE) loss function


Cross-Validation:

Implements K-fold cross-validation (default: 5 folds)
Computes MSE and R² metrics for each fold
Calculates average performance across folds
Early stopping to prevent overfitting


Visualization:

Plots training and validation loss over epochs
Visualizes cross-validation error across folds
Charts feature importance based on network weights
Shows loss and MAE trends during training



Usage:
To use this code, you'll need to:

Replace "your_dataset.csv" with the path to your actual CSV file
Set the target_column variable to the name of your target variable
Adjust hyperparameters (learning rate, layer nodes, etc.) as needed

This implementation supports both regression and classification tasks (with minor adjustments to the output layer for classification).
Would you like me to explain any particular part of this implementation in more detail?
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    # Load CSV file
    data = pd.read_csv(file_path)
    print(f"Data loaded with shape: {data.shape}")
    
    # Display the first few rows of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Display basic information about the dataset
    print("\nBasic statistics:")
    print(data.describe())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    # Handle missing values (if any)
    if missing_values.sum() > 0:
        print("Filling missing values...")
        # Fill numeric columns with mean
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mean())
        
        # Fill categorical columns with mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0])
    
    # Handle categorical features (if any)
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print("\nEncoding categorical variables...")
        data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    return data

# Function to split data into features and target for classification
def prepare_data(data, target_column, is_classification=True):
    print(f"\nSplitting data into features and target (target: {target_column})...")
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # For classification, we'll convert target to categorical 
    if is_classification:
        # Get number of unique classes
        num_classes = len(np.unique(y))
        print(f"Number of classes: {num_classes}")
        
        # For binary classification
        if num_classes == 2:
            y_categorical = y  # Keep as is for binary
        else:
            # For multi-class classification, we need to one-hot encode
            y_categorical = to_categorical(y)
    else:
        y_categorical = y  # For regression
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val, scaler, X.columns, num_classes

# Build the ANN model for classification
def build_model(input_dim, num_classes, learning_rate=0.001, layer1_nodes=64, layer2_nodes=64, layer3_nodes=64):
    print(f"\nBuilding ANN with learning rate: {learning_rate}")
    print(f"Layer 1: {layer1_nodes} nodes, Layer 2: {layer2_nodes} nodes")
    
    model = Sequential([
        # First hidden layer
        Dense(layer1_nodes, activation='relu', input_dim=input_dim),
        # Second hidden layer
        Dense(layer2_nodes, activation='relu'),
        # Third hidden layer
        Dense(layer3_nodes, activation='relu'),
        # Output layer - depends on the number of classes
        Dense(1 if num_classes == 2 else num_classes, 
              activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    
    # Compile the model with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    
    # Use appropriate loss function based on number of classes
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

# Perform k-fold cross-validation for classification
def cross_validate(X, y, num_classes, n_splits=5, learning_rate=0.001, 
                  layer1_nodes=64, layer2_nodes=32, epochs=100, batch_size=32):
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    accuracy_scores = []
    history_list = []
    
    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold_no}")
        
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        
        # For binary classification, y might be 1D array
        if num_classes == 2 and len(y.shape) == 1:
            y_train, y_val = y[train_index], y[val_index]
        else:
            # For multi-class with one-hot encoding
            y_train, y_val = y[train_index], y[val_index]
        
        # Build model
        model = build_model(
            input_dim=X.shape[1],
            num_classes=num_classes,
            learning_rate=learning_rate,
            layer1_nodes=layer1_nodes,
            layer2_nodes=layer2_nodes
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model for classification
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        accuracy_scores.append(val_accuracy)
        history_list.append(history)
        
        fold_no += 1
    
    # Calculate average metrics
    avg_accuracy = np.mean(accuracy_scores)
    
    print("\nCross-validation results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    
    return accuracy_scores, history_list

# Plot cross-validation metrics for classification
def plot_cv_results(accuracy_scores, history_list, n_splits):
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy for each fold
    ax1.bar(range(1, n_splits+1), accuracy_scores)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Cross-Validation Fold')
    ax1.set_xticks(range(1, n_splits+1))
    
    # Plot 2: Training & validation accuracy across epochs for each fold
    for i, history in enumerate(history_list):
        ax2.plot(history.history['accuracy'], linestyle='--', alpha=0.7, label=f'Train Acc (Fold {i+1})')
        ax2.plot(history.history['val_accuracy'], alpha=0.7, label=f'Val Acc (Fold {i+1})')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    # Convert probabilities to class predictions if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # For binary problems where output is probability
    if len(y_pred.shape) == 1 and np.max(y_pred) <= 1 and np.min(y_pred) >= 0:
        y_pred_class = (y_pred > 0.5).astype(int)
    else:
        y_pred_class = y_pred
    
    # If y_true is one-hot encoded, convert back to class indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_class, target_names=[str(c) for c in classes]))

# Plot feature importance
def plot_feature_importance(model, feature_names, title="Feature Importance"):
    # For a simple neural network, we'll use the weights from the first layer as a proxy for feature importance
    weights = model.layers[0].get_weights()[0]
    
    # Calculate the absolute mean of weights for each feature
    importance = np.abs(weights).mean(axis=1)
    
    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return importance_df

# Plot ROC curve (for binary classification)
def plot_roc_curve(y_true, y_pred_prob):
    from sklearn.metrics import roc_curve, auc
    
    # If y_true is one-hot encoded, convert back to class indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # For multi-class, we'd need to adjust this further
    # If just a simple binary classification with 0/1 output:
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def main():
    # Replace with your actual CSV file path
    file_path = "mi_base.csv"  
    
    # Replace with your actual target column name
    target_column = "class"  
    
    try:
        # Load and preprocess data
        data = load_data(file_path)
        
        # Determine if this is a classification or regression task
        # For demonstration, we'll assume classification
        is_classification = True
        
        # Prepare data
        X_train, X_val, y_train, y_val, scaler, feature_names, num_classes = prepare_data(
            data, target_column, is_classification=is_classification
        )
        
        # Model hyperparameters
        learning_rate = 0.001
        layer1_nodes = 64
        layer2_nodes = 32
        epochs = 100
        batch_size = 32
        n_splits = 5
        
        # Build and train the model with holdout validation
        print("\nTraining final model on 80% training data...")
        model = build_model(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            learning_rate=learning_rate,
            layer1_nodes=layer1_nodes,
            layer2_nodes=layer2_nodes
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        y_pred = model.predict(X_val)
        
        print(f"\nFinal model evaluation on holdout validation set:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        
        # Plot training history - now including accuracy
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot 2: Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot 3: Accuracy vs Loss
        plt.subplot(1, 3, 3)
        plt.plot(history.history['loss'], history.history['accuracy'], 'o-', label='Training')
        plt.plot(history.history['val_loss'], history.history['val_accuracy'], 'o-', label='Validation')
        plt.title('Accuracy vs Loss')
        plt.xlabel('Loss')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrix
        class_names = range(num_classes)
        plot_confusion_matrix(np.argmax(y_val, axis=1) if num_classes > 2 and len(y_val.shape) > 1 else y_val, 
                              y_pred, class_names)
        
        # For binary classification, plot ROC curve
        print(f"Numero de clases {num_classes}")
        if num_classes == 2:
            plot_roc_curve(y_val, y_pred)
            print("Curvas ROC")
        
        # Perform cross-validation
        X_combined = np.vstack((X_train, X_val))
        
        # Combine y_train and y_val appropriately
        if num_classes == 2 and len(y_train.shape) == 1:
            y_combined = np.concatenate((y_train, y_val))
        else:
            y_combined = np.vstack((y_train, y_val))
        
        accuracy_scores, history_list = cross_validate(
            X_combined, y_combined,
            num_classes=num_classes,
            n_splits=n_splits,
            learning_rate=learning_rate,
            layer1_nodes=layer1_nodes,
            layer2_nodes=layer2_nodes,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Plot cross-validation results - now for accuracy
        plot_cv_results(accuracy_scores, history_list, n_splits)
        
        # Plot feature importance
        importance_df = plot_feature_importance(model, feature_names)
        print("\nFeature importance:")
        print(importance_df)
        
        # Save the model
        model.save("ann_classification_model.h5")
        print("\nModel saved as 'ann_classification_model.h5'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()