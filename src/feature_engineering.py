import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        
    def create_streak_feature(self):
        """
        Create a streak feature based on consecutive quiz attempts
        This is a simplified version - in real scenario, you'd need sequential data
        """
        # Using NumberOfQuizzesTaken as a proxy for engagement streak
        # Higher number of quizzes taken indicates better engagement streak
        self.data['EngagementStreak'] = pd.qcut(
            self.data['NumberOfQuizzesTaken'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Convert to numerical
        streak_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        self.data['StreakScore'] = self.data['EngagementStreak'].map(streak_mapping)
        
        return self.data
    
    def create_engagement_efficiency_feature(self):
        """Create engagement efficiency score (quiz scores per hour)"""
        self.data['Efficiency'] = self.data['QuizScores'] / (self.data['TimeSpentOnCourse'] + 0.1)
        return self.data
    
    def create_consistency_feature(self):
        """Create consistency score based on videos watched and quizzes taken"""
        self.data['Consistency'] = self.data['NumberOfVideosWatched'] * self.data['NumberOfQuizzesTaken']
        return self.data
    
    def create_all_features(self):
        """Create all engineered features"""
        self.data = self.create_streak_feature()
        self.data = self.create_engagement_efficiency_feature()
        self.data = self.create_consistency_feature()
        
        print("\nEngineered Features Created:")
        print(f"- StreakScore (based on NumberOfQuizzesTaken)")
        print(f"- Efficiency (QuizScores / TimeSpentOnCourse)")
        print(f"- Consistency (VideosWatched × QuizzesTaken)")
        
        return self.data