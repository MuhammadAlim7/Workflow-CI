import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    mlflow.autolog()
    with mlflow.start_run() as run:
        # Load preprocessed data
        df = pd.read_csv("Teen_Mental_Health_preprocessing.csv")
        X = df.drop(columns=["depression_label"])
        y = df["depression_label"]
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Write run ID to a file so GitHub Actions can read it for Docker build
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        
        print("Model retraining and logging complete. Run ID saved.")
