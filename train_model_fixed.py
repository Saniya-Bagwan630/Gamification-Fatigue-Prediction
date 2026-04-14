import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models folder
os.makedirs('models', exist_ok=True)

print("="*60)
print("STEP 1: Loading Data")
print("="*60)

# Load the data
df = pd.read_csv('data/online_course_engagement_data.csv')
print(f"✓ Loaded {len(df)} student records")

print("\n" + "="*60)
print("STEP 2: Creating Features")
print("="*60)

# Create difficulty feature (inverse of quiz scores)
df['Difficulty'] = 100 - df['QuizScores']

# Select features for the model
features = ['Difficulty', 'TimeSpentOnCourse', 'NumberOfQuizzesTaken']
X = df[features]

# Better fatigue definition using percentiles
# A student is fatigued if:
# 1. Quiz scores in bottom 30% AND time spent in top 40%
# 2. Quiz scores in bottom 20%
# 3. Completion rate in bottom 20% AND time spent in top 40%

quiz_bottom_30 = df['QuizScores'] < df['QuizScores'].quantile(0.30)
time_top_40 = df['TimeSpentOnCourse'] > df['TimeSpentOnCourse'].quantile(0.60)
quiz_bottom_20 = df['QuizScores'] < df['QuizScores'].quantile(0.20)
completion_bottom_20 = df['CompletionRate'] < df['CompletionRate'].quantile(0.20)

condition1 = quiz_bottom_30 & time_top_40
condition2 = quiz_bottom_20
condition3 = completion_bottom_20 & time_top_40

df['Fatigue'] = (condition1 | condition2 | condition3).astype(int)

print(f"✓ Features: {features}")
print(f"✓ Total students: {len(X)}")
print(f"✓ Fatigued students: {df['Fatigue'].sum()} ({df['Fatigue'].mean()*100:.1f}%)")
print(f"✓ Non-fatigued students: {(len(df)-df['Fatigue'].sum())} ({(1-df['Fatigue'].mean())*100:.1f}%)")

# Show sample of fatigued students
fatigued_sample = df[df['Fatigue'] == 1].head(3)
if len(fatigued_sample) > 0:
    print("\n✓ Sample of fatigued students:")
    for idx, row in fatigued_sample.iterrows():
        print(f"   - Quiz Score: {row['QuizScores']:.1f}, Time: {row['TimeSpentOnCourse']:.1f}h, Completion: {row['CompletionRate']:.1f}%")

y = df['Fatigue']

print("\n" + "="*60)
print("STEP 3: Training the Model")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✓ Training set: {len(X_train)} students")
print(f"✓ Test set: {len(X_test)} students")
print(f"✓ Training fatigue rate: {y_train.mean()*100:.1f}%")
print(f"✓ Test fatigue rate: {y_test.mean()*100:.1f}%")

# Train model with class balancing
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print(f"✓ Model training complete!")

print("\n" + "="*60)
print("STEP 4: Model Performance")
print("="*60)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"✓ Training accuracy: {train_accuracy:.2%}")
print(f"✓ Test accuracy: {test_accuracy:.2%}")

print("\n✓ Model Coefficients:")
print(f"   Intercept: {model.intercept_[0]:.4f}")
for name, coef in zip(features, model.coef_[0]):
    direction = "INCREASES fatigue" if coef > 0 else "DECREASES fatigue"
    print(f"   {name}: {coef:.4f} → {direction}")

print("\n" + "="*60)
print("STEP 5: Testing with Example Students")
print("="*60)

test_students = [
    {'name': 'Fresh Student', 'difficulty': 15, 'time': 5, 'streak': 8},
    {'name': 'Engaged Student', 'difficulty': 30, 'time': 20, 'streak': 10},
    {'name': 'At Risk Student', 'difficulty': 60, 'time': 55, 'streak': 3},
    {'name': 'Fatigued Student', 'difficulty': 75, 'time': 80, 'streak': 1}
]

for student in test_students:
    features_input = [[student['difficulty'], student['time'], student['streak']]]
    probability = model.predict_proba(features_input)[0, 1]
    prediction = "Fatigued" if probability >= 0.5 else "Not Fatigued"
    
    print(f"\n📊 {student['name']}:")
    print(f"   Difficulty: {student['difficulty']}/100")
    print(f"   Session Time: {student['time']} hours")
    print(f"   Streak: {student['streak']} quizzes")
    print(f"   → Fatigue Probability: {probability:.1%}")
    print(f"   → Prediction: {prediction}")

print("\n" + "="*60)
print("STEP 6: Saving the Model")
print("="*60)

joblib.dump(model, 'models/fatigue_model.joblib')
print("✓ Model saved to: models/fatigue_model.joblib")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run the web app with:")
print("   py -3.11 -m streamlit run app.py")