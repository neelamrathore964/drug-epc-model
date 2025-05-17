import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')

    acc_gap = acc_train - acc_test
    f1_gap = f1_train - f1_test

    print(f"\n🔍 {name.upper()} Evaluation:")
    print(f"📊 Train Accuracy: {acc_train:.4f}, F1: {f1_train:.4f}")
    print(f"🧪 Test Accuracy:  {acc_test:.4f}, F1: {f1_test:.4f}")
    print(f"📉 Accuracy Gap (Train - Test): {acc_gap:.4f}")
    print(f"📉 F1 Gap (Train - Test): {f1_gap:.4f}")
    print("📄 Classification Report:\n", classification_report(y_test, y_test_pred))

    if acc_train < 0.6 and acc_test < 0.6:
        print("⚠️ Likely underfitting: low performance on both.")
    elif acc_gap > 0.1 or f1_gap > 0.1:
        print("⚠️ Possible overfitting: large gap between train and test.")
    else:
        print("✅ Model appears well-balanced.")

    return {
        "model": model,
        "name": name,
        "acc_train": acc_train,
        "f1_train": f1_train,
        "acc_test": acc_test,
        "f1_test": f1_test
    }

def train_model(X, y, output_dir, do_cv=True, cv_folds=5):
    os.makedirs(output_dir, exist_ok=True)

    # Train-test split before SMOTE (to evaluate performance on real test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
       #"random_forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        #"gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
    }

    results = []
    best_model = None
    best_f1 = 0.0

    for name, base_model in models.items():
        print(f"\n🚀 Training model: {name}")

        # Create SMOTE pipeline
        if name == "logistic_regression":
            pipeline = ImbPipeline([
                #("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
            ])
        else:
            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                ("clf", base_model)
            ])

        #Cross-validation (on full original data with SMOTE)
        if do_cv:
            try:
                print("🔁 Cross-validating...")
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='f1_weighted', n_jobs=-1)
                print(f"📊 CV F1 Scores: {cv_scores}")
                print(f"📈 Mean CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            except Exception as e:
                print(f"⚠️ CV failed for {name}: {e}")

        # Fit on train with SMOTE
        pipeline.fit(X_train, y_train)
        result = evaluate_model(pipeline, X_train, y_train, X_test, y_test, name)
        results.append(result)

        # Save model
        model_path = os.path.join(output_dir, f"{name}_smote.pkl")
        joblib.dump(pipeline, os.path.join(output_dir, f"{name}_epc_rating.pkl"))
        #joblib.dump(pipeline, model_path)
        print(f"💾 Saved model to: {model_path}")

        if result["f1_test"] > best_f1:
            best_f1 = result["f1_test"]
            best_model = result

    if best_model:
        best_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(best_model["model"], best_path)
        print(f"\n🏆 Best model ({best_model['name']}) saved to: {best_path}")
        print(f"🎯 Best Test F1 Score: {best_model['f1_test']:.4f}")

    return best_model["model"]
