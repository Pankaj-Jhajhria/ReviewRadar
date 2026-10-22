
from pathlib import Path
import json
import time
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"
BASE=ROOT/"models"/"sentiment_baselines"
REPORTS=ROOT/"reports"
LABELS=["negative","neutral","positive"]

def main():
    bundle=joblib.load(BASE/"logistic_regression_tuned.joblib")
    vectorizer=bundle["vectorizer"]

    train=pd.read_parquet(DATA/"train_sample_500000.parquet")
    val=pd.read_parquet(DATA/"validation_sample_100000.parquet")

    train=train[train.text.notna() & train.text.str.strip().ne("")]
    val=val[val.text.notna() & val.text.str.strip().ne("")]

    print("Transforming text...")
    X_train=vectorizer.transform(train.text.astype(str))
    X_val=vectorizer.transform(val.text.astype(str))
    y_train=train.sentiment.to_numpy()
    y_val=val.sentiment.to_numpy()

    del train,val

    configs={
        "unweighted":None,
        "balanced":"balanced",
        "neutral_2x":{"negative":1.0,"neutral":2.0,"positive":1.0},
        "neutral_3x":{"negative":1.0,"neutral":3.0,"positive":1.0},
        "moderate_balance":{"negative":1.5,"neutral":3.0,"positive":0.7}
    }

    results={}
    best_model=None
    best_score=-1

    for name,weights in configs.items():
        print(f"\nTraining: {name}")
        model=LogisticRegression(
            max_iter=500,
            class_weight=weights,
            solver="lbfgs",
            random_state=42
        )

        start=time.perf_counter()
        model.fit(X_train,y_train)
        pred=model.predict(X_val)

        report=classification_report(
            y_val,pred,labels=LABELS,
            output_dict=True,zero_division=0
        )
        score=f1_score(y_val,pred,average="macro")

        results[name]={
            "accuracy":float(accuracy_score(y_val,pred)),
            "macro_f1":float(score),
            "fit_minutes":round((time.perf_counter()-start)/60,2),
            "class_f1":{
                label:float(report[label]["f1-score"])
                for label in LABELS
            }
        }

        print(
            f"Accuracy={results[name]['accuracy']:.4f} | "
            f"Macro-F1={score:.4f} | "
            f"Neutral F1={report['neutral']['f1-score']:.4f}"
        )

        if score>best_score:
            best_score=score
            best_model=model

    REPORTS.mkdir(parents=True,exist_ok=True)
    (REPORTS/"class_weight_results.json").write_text(
        json.dumps(results,indent=2),encoding="utf-8"
    )

    joblib.dump(
        {"vectorizer":vectorizer,"model":best_model,"labels":LABELS},
        BASE/"logistic_regression_weight_tuned.joblib"
    )

    best=max(results,key=lambda name:results[name]["macro_f1"])
    print(f"\nBest configuration: {best}")
    print(f"Macro-F1: {best_score:.4f}")
    print("Results saved to reports/class_weight_results.json")

if __name__=="__main__":
    main()
