import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_plot_style():
    """Set consistent plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def plot_feature_importance(coeffs, feature_names, title='Feature Importance'):
    """Plot feature importance based on coefficients"""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coeffs
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in importance_df['Coefficient']]
    plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title(title)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    return plt

def calculate_fatigue_probability(beta0, beta1, beta2, beta3, x1, x2, x3):
    """
    Calculate fatigue probability using logistic regression formula
    
    Parameters:
    - beta0: intercept
    - beta1: coefficient for difficulty (QuizScores)
    - beta2: coefficient for session time (TimeSpentOnCourse)
    - beta3: coefficient for streak (NumberOfQuizzesTaken)
    - x1: difficulty value (lower score = higher difficulty)
    - x2: session time
    - x3: streak (number of quizzes)
    """
    import math
    
    # Invert QuizScores to represent difficulty (lower score = harder)
    difficulty = 100 - x1
    
    z = beta0 + (beta1 * difficulty) + (beta2 * x2) + (beta3 * x3)
    probability = 1 / (1 + math.exp(-z))
    return probability