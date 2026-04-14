import numpy as np
import pandas as pd

class FatiguePredictor:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        
    def predict_single_user(self, difficulty, time_spent, streak, threshold=0.5):
        """
        Predict fatigue for a single user
        
        Parameters:
        - difficulty: 0-100 (higher = more difficult)
        - time_spent: hours spent
        - streak: number of quizzes taken
        - threshold: probability threshold for fatigue classification
        
        Returns:
        - prediction: 0 or 1
        - probability: fatigue probability
        - recommendation: action to take
        """
        # Create feature array
        features = np.array([[difficulty, time_spent, streak]])
        
        # Scale if scaler is provided
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Get probability
        probability = self.model.predict_proba(features)[0, 1]
        
        # Make prediction
        prediction = 1 if probability >= threshold else 0
        
        # Generate recommendation based on probability
        if probability >= 0.7:
            recommendation = "URGENT: User showing high fatigue. Suggest immediate break and reduce difficulty."
        elif probability >= 0.5:
            recommendation = "ALERT: User at risk of fatigue. Consider adjusting session length or difficulty."
        elif probability >= 0.3:
            recommendation = "MONITOR: User showing mild fatigue signs. Continue monitoring."
        else:
            recommendation = "GOOD: User engaged and not fatigued. Maintain current approach."
        
        return {
            'fatigue_probability': probability,
            'prediction': 'Fatigued' if prediction == 1 else 'Not Fatigued',
            'recommendation': recommendation,
            'threshold_used': threshold
        }
    
    def predict_batch(self, X, threshold=0.5):
        """Predict for multiple users"""
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def get_risk_level(self, probability):
        """Categorize fatigue risk level"""
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.5:
            return "Medium Risk"
        elif probability >= 0.3:
            return "Low Risk"
        else:
            return "No Risk"