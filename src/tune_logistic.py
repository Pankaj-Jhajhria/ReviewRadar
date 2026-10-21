
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"models"/"sentiment_baselines"
DATA=ROOT/"data"/"processed"

bundle=joblib.load(BASE/"logistic_regression.joblib")
vectorizer=bundle["vectorizer"]

train=pd.read_parquet(DATA/"train_sample_500000.parquet")
val=pd.read_parquet(DATA/"validation_sample_100000.parquet")

X_train=vectorizer.transform(train["text"].fillna("").astype(str))
X_val=vectorizer.transform(val["text"].fillna("").astype(str))
y_train=train["sentiment"]
y_val=val["sentiment"]

model=LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

start=time.perf_counter()
model.fit(X_train,y_train)

pred=model.predict(X_val)

print("Iterations:",model.n_iter_)
print("Training time:",round((time.perf_counter()-start)/60,2),"min")
print("Accuracy:",round(accuracy_score(y_val,pred),4))
print("Macro-F1:",round(f1_score(y_val,pred,average="macro"),4))
print(classification_report(y_val,pred,zero_division=0))

joblib.dump(
    {"vectorizer":vectorizer,"model":model,"labels":bundle["labels"]},
    BASE/"logistic_regression_tuned.joblib"
)
print("Saved tuned model.")
