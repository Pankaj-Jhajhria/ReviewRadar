from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import classification_report,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"
MODEL=ROOT/"models"/"sentiment_baselines"/"logistic_regression_tuned.joblib"
OUT=ROOT/"reports"

bundle=joblib.load(MODEL)
vectorizer=bundle["vectorizer"]
model=bundle["model"]

val=pd.read_parquet(DATA/"validation_sample_100000.parquet")
val=val[val["text"].notna() & val["text"].str.strip().ne("")].copy()

X=vectorizer.transform(val["text"].astype(str))
y=val["sentiment"].to_numpy()
pred=model.predict(X)

labels=["negative","neutral","positive"]

print("Classification report:")
print(classification_report(y,pred,labels=labels,zero_division=0))

print("Confusion matrix (actual rows, predicted columns):")
print(pd.DataFrame(
    confusion_matrix(y,pred,labels=labels),
    index=labels,columns=labels
))

val["predicted_sentiment"]=pred
errors=val[val["sentiment"]!=val["predicted_sentiment"]].copy()
errors["error_type"]=errors["sentiment"]+" -> "+errors["predicted_sentiment"]

OUT.mkdir(parents=True,exist_ok=True)
errors[["text","sentiment","predicted_sentiment","error_type"]].head(1000).to_csv(
    OUT/"sentiment_misclassified_reviews.csv",index=False
)

summary=errors["error_type"].value_counts().to_dict()
(OUT/"sentiment_error_summary.json").write_text(
    json.dumps(summary,indent=2),encoding="utf-8"
)

print(f"\nIncorrect predictions: {len(errors):,}/{len(val):,}")
print("Saved misclassified reviews and error summary in reports/.")