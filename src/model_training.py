import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import joblib
import os

class FatigueModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.intercept = None
        self.coefficients = None
        
    def train(self, X_train, y_train, feature_names, C=1.0, max_iter=1000):
        """
        Train logistic regression model
        
        Parameters:
        - X_train: training features
        - y_train: training labels
        - feature_names: list of feature names
        - C: inverse regularization strength (smaller = stronger regularization)
        - max_iter: maximum iterations for convergence
        """
        self.feature_names = feature_names
        
        # Initialize logistic regression with class_weight to handle imbalance
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42,
            solver='lbfgs'
        )
        
        # Train the model
        print("\n" + "="*50)
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print("="*50)
        print(f"Features: {feature_names}")
        print(f"Training samples: {X_train.shape[0]}")
        
        self.model.fit(X_train, y_train)
        
        # Store coefficients
        self.intercept = self.model.intercept_[0]
        self.coefficients = self.model.coef_[0]
        
        print(f"\nModel Coefficients:")
        print(f"Intercept (β₀): {self.intercept:.4f}")
        for name, coef in zip(feature_names, self.coefficients):
            print(f"β_{name}: {coef:.4f}")
        
        return self.model
    
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['No Fatigue', 'Fatigue']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def calculate_fatigue_probability_manual(self, difficulty, time_spent, streak):
        """
        Manually calculate fatigue probability using formula
        
        Parameters:
        - difficulty: 0-100 (higher = harder)
        - time_spent: hours spent on course
        - streak: number of quizzes taken
        
        Returns:
        - fatigue probability (0-1)
        """
        import math
        
        # The features need to be scaled the same way as training data
        # For demonstration, we'll use raw coefficients
        # In production, you'd need to scale the inputs
        
        z = self.intercept
        for i, feature_name in enumerate(self.feature_names):
            if feature_name == 'Difficulty':
                z += self.coefficients[i] * difficulty
            elif feature_name == 'TimeSpentOnCourse':
                z += self.coefficients[i] * time_spent
            elif feature_name == 'NumberOfQuizzesTaken':
                z += self.coefficients[i] * streak
        
        probability = 1 / (1 + math.exp(-z))
        return probability
    
    def save_model(self, filepath='models/fatigue_model.joblib'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'intercept': self.intercept,
            'coefficients': self.coefficients,
            'scaler': None  # You would save the scaler here
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='models/fatigue_model.joblib'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.intercept = model_data['intercept']
        self.coefficients = model_data['coefficients']
        print(f"\nModel loaded from {filepath}")
        return self