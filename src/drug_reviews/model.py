import os
import numpy as np
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_evaluate(X_train, y_train, X_test, y_test,output_dir):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Gradient Boosting": GradientBoostingClassifier(
                                n_estimators=50,
                                max_depth=3,
                                subsample=0.8,
                                learning_rate=0.1,
                                random_state=42
                            ),
        "Random Forest": RandomForestClassifier(
                                n_estimators=100,
                                max_features="sqrt",
                                max_depth=6, 
                                max_leaf_nodes=6,
                                random_state=42
                            ),     
        }
    
    vectorizer = TfidfVectorizer(max_features=5000)
    smote = SMOTE(random_state=42)
    results = {}

    best_model_name = None
    best_model_pipeline = None
    best_model_accuracy = 0

    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"\n Training: {name}")

        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('chi2', SelectKBest(chi2, k=3000)),  # Select top 3000 features
            ('smote', smote),  
            ('clf', model)
        ])

        start_time = time.time()

        # Cross-validation and fit model on full training data after CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        #cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        pipeline.fit(X_train, y_train)

        # Predictions on train and test data
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        end_time = time.time()

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        elapsed = end_time - start_time

        print(f"Cross-Validation Accuracy (5-fold): {cv_scores}")
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(classification_report(
            y_test, test_pred, target_names=['negative', 'neutral', 'positive'], zero_division=0
        ))

        gap = abs(train_acc - test_acc)
        mean_cv = np.mean(cv_scores)
        
        if test_acc > best_model_accuracy and gap < 0.05:
            best_model_accuracy = test_acc
            best_model_name = name
            best_model_pipeline = pipeline
            
        results[name] = {
            "model": pipeline,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "cv": mean_cv,
        }

        joblib.dump(pipeline, os.path.join(output_dir, f"{name}_sentiment_model.pkl"))
        cm = confusion_matrix(y_test, test_pred, normalize='true')  # row-wise normalization
        print("Confusion Matrix:\n", cm)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
        plt.close()

    print("\n Model Fit Analysis:")
    for model_name, result in results.items():
        gap = abs(result['train_accuracy'] - result['test_accuracy'])
        overfit_status = "✅ Balanced"
        if gap > 0.05:
            overfit_status = "⚠️ Possible overfitting" if result['train_accuracy'] > result['test_accuracy'] else "⚠️ Possible underfitting"
        print(f"{model_name}: Train = {result['train_accuracy']:.4f}, Test = {result['test_accuracy']:.4f}, Gap = {gap:.4f} → {overfit_status}")

    # Save the best model 
    best_model_path = os.path.join(output_dir, f"{best_model_name}_best_model.pkl")
    joblib.dump(best_model_pipeline, best_model_path)
    print(f" Best model saved: {best_model_name}")