import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data_path):
        """Initialize the preprocessor with data path"""
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset"""
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def explore_data(self):
        """Print basic information about the dataset"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        print(f"\nFirst 5 rows:")
        print(self.data.head())
        
        print(f"\nDataset Info:")
        print(self.data.info())
        
        print(f"\nMissing Values:")
        print(self.data.isnull().sum())
        
        print(f"\nBasic Statistics:")
        print(self.data.describe())
        
        return self.data
    
    def create_fatigue_label(self):
        """
        Create fatigue label based on engagement patterns
        Fatigue = 1 if:
        - High session time AND low quiz scores (cognitive overload)
        - OR low completion rate despite high time
        """
        # Calculate difficulty (inverse of QuizScores)
        self.data['Difficulty'] = 100 - self.data['QuizScores']
        
        # Normalize features for threshold calculation
        time_percentile = self.data['TimeSpentOnCourse'].rank(pct=True)
        score_percentile = self.data['QuizScores'].rank(pct=True)
        completion_percentile = self.data['CompletionRate'].rank(pct=True)
        
        # Define fatigue conditions
        condition1 = (time_percentile > 0.7) & (score_percentile < 0.3)  # High time, low scores
        condition2 = (time_percentile > 0.7) & (completion_percentile < 0.3)  # High time, low completion
        
        # Create fatigue label (1 = fatigued, 0 = not fatigued)
        self.data['Fatigue'] = (condition1 | condition2).astype(int)
        
        print(f"\nFatigue Distribution:")
        print(self.data['Fatigue'].value_counts())
        print(f"Fatigue Rate: {self.data['Fatigue'].mean()*100:.2f}%")
        
        return self.data
    
    def prepare_features(self):
        """Select and prepare features for the model"""
        # Define features
        feature_columns = ['Difficulty', 'TimeSpentOnCourse', 'NumberOfQuizzesTaken']
        target_column = 'Fatigue'
        
        # Additional features (optional - uncomment to use more features)
        # feature_columns = ['Difficulty', 'TimeSpentOnCourse', 'NumberOfQuizzesTaken', 
        #                    'NumberOfVideosWatched', 'QuizScores']
        
        X = self.data[feature_columns].copy()
        y = self.data[target_column].copy()
        
        # Scale features (important for logistic regression)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        print(f"\nFeatures prepared: {feature_columns}")
        print(f"X shape: {X_scaled.shape}, y shape: {y.shape}")
        
        return X_scaled, y, feature_columns
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData Split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training fatigue rate: {y_train.mean()*100:.2f}%")
        print(f"Test fatigue rate: {y_test.mean()*100:.2f}%")
        
        return X_train, X_test, y_train, y_test