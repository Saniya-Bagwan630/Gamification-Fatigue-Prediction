import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import FatigueModel
from src.prediction import FatiguePredictor
from src.utils import set_plot_style, plot_confusion_matrix, plot_feature_importance
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("GAMIFICATION FATIGUE PREDICTION MODEL")
    print("Using Logistic Regression for Real-time Fatigue Detection")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    preprocessor = DataPreprocessor('data/online_course_engagement_data.csv')
    data = preprocessor.load_data()
    data = preprocessor.explore_data()
    data = preprocessor.create_fatigue_label()
    
    # 2. Feature engineering
    print("\n[2/6] Creating engineered features...")
    engineer = FeatureEngineer(data)
    data = engineer.create_all_features()
    
    # 3. Prepare features for modeling
    print("\n[3/6] Preparing features for modeling...")
    X, y, feature_names = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    
    # 4. Train the logistic regression model
    print("\n[4/6] Training logistic regression model...")
    model = FatigueModel()
    model.train(X_train, y_train, feature_names, C=1.0, max_iter=1000)
    
    # 5. Evaluate the model
    print("\n[5/6] Evaluating model performance...")
    evaluation_results = model.evaluate(X_test, y_test)
    
    # 6. Make predictions and demonstrate usage
    print("\n[6/6] Making predictions and demonstrating usage...")
    
    # Create predictor
    predictor = FatiguePredictor(model.model)
    
    # Test different scenarios
    test_scenarios = [
        {"name": "High Risk Student", "difficulty": 85, "time_spent": 90, "streak": 2},
        {"name": "Medium Risk Student", "difficulty": 60, "time_spent": 50, "streak": 5},
        {"name": "Low Risk Student", "difficulty": 30, "time_spent": 20, "streak": 8},
        {"name": "Engaged Student", "difficulty": 20, "time_spent": 15, "streak": 10},
    ]
    
    print("\n" + "="*50)
    print("PREDICTION EXAMPLES")
    print("="*50)
    
    for scenario in test_scenarios:
        result = predictor.predict_single_user(
            difficulty=scenario["difficulty"],
            time_spent=scenario["time_spent"],
            streak=scenario["streak"],
            threshold=0.5
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  - Difficulty: {scenario['difficulty']}/100")
        print(f"  - Time Spent: {scenario['time_spent']} hours")
        print(f"  - Streak: {scenario['streak']} quizzes")
        print(f"  - Fatigue Probability: {result['fatigue_probability']:.2%}")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Recommendation: {result['recommendation']}")
    
    # Calculate feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Plot feature importance
    set_plot_style()
    fig1 = plot_feature_importance(model.coefficients, feature_names, 
                                   'Logistic Regression Coefficients')
    fig1.savefig('outputs/figures/feature_importance.png', dpi=150, bbox_inches='tight')
    print("✓ Feature importance plot saved to outputs/figures/feature_importance.png")
    
    # Plot confusion matrix
    cm = evaluation_results['confusion_matrix']
    fig2 = plot_confusion_matrix(cm, ['No Fatigue', 'Fatigue'], 
                                 'Confusion Matrix - Fatigue Prediction')
    fig2.savefig('outputs/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Confusion matrix saved to outputs/figures/confusion_matrix.png")
    
    # Save the model
    model.save_model('models/fatigue_model.joblib')
    
    # Generate report
    print("\n" + "="*50)
    print("MODEL SUMMARY REPORT")
    print("="*50)
    print(f"""
    Model: Logistic Regression
    Features used: {', '.join(feature_names)}
    
    Model Equation:
    P(Fatigue=1) = 1 / (1 + e^(-z))
    where z = {model.intercept:.4f} + {model.coefficients[0]:.4f}×Difficulty + 
              {model.coefficients[1]:.4f}×TimeSpent + {model.coefficients[2]:.4f}×Streak
    
    Performance Metrics:
    - Accuracy:  {evaluation_results['accuracy']:.4f}
    - Precision: {evaluation_results['precision']:.4f}
    - Recall:    {evaluation_results['recall']:.4f}
    - F1-Score:  {evaluation_results['f1']:.4f}
    - ROC-AUC:   {evaluation_results['roc_auc']:.4f}
    
    Interpretation:
    - Positive coefficient = increases fatigue probability
    - Negative coefficient = decreases fatigue probability
    
    Based on our model:
    - Difficulty (inverse of quiz scores) has coefficient {model.coefficients[0]:.4f}
    - Session Time has coefficient {model.coefficients[1]:.4f}
    - Streak has coefficient {model.coefficients[2]:.4f}
    """)
    
    print("\n" + "="*50)
    print("MODEL READY FOR DEPLOYMENT!")
    print("="*50)
    print("\nYou can now:")
    print("1. Use the model for real-time fatigue prediction")
    print("2. Integrate with learning apps for adaptive responses")
    print("3. Monitor fatigue levels and trigger interventions")
    print("4. Retrain the model with new data for continuous improvement")
    
    return model, preprocessor.scaler

if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # Run main
    model, scaler = main()