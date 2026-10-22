
from pathlib import Path
import json
import time
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"
MODELS=ROOT/"models"/"sentiment_baselines"
REPORTS=ROOT/"reports"

TRAIN_N=5_000_000
VAL_N=100_000
LABELS=["negative","neutral","positive"]

def sql_path(path):
    return "'"+path.resolve().as_posix().replace("'","''")+"'"

def get_sample(con,split,limit):
    cache=DATA/f"{split}_sample_{limit}.parquet"

    if cache.exists():
        print(f"Using cached {split} sample: {cache.name}")
        return pd.read_parquet(cache)

    print(f"Preparing {limit:,} {split} reviews...")
    query=f"""
        SELECT text,sentiment,parent_asin,asin,timestamp_ms,rating
        FROM read_parquet({sql_path(DATA/"amazon_reviews_labeled_split.parquet")})
        WHERE split='{split}'
          AND text IS NOT NULL
          AND TRIM(text)<>''
          AND sentiment IN ('negative','neutral','positive')
        ORDER BY hash(
            COALESCE(parent_asin,'') || '|' ||
            COALESCE(asin,'') || '|' ||
            COALESCE(CAST(timestamp_ms AS VARCHAR),'') || '|' ||
            COALESCE(CAST(rating AS VARCHAR),'') || '|' ||
            LEFT(text,100)
        )
        LIMIT {limit}
    """
    df=con.execute(query).fetchdf()
    if len(df)<limit:
        raise RuntimeError(f"Expected {limit:,} rows, got {len(df):,}")
    df.to_parquet(cache,index=False)
    return df

def main():
    DATASET=DATA/"amazon_reviews_labeled_split.parquet"
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    MODELS.mkdir(parents=True,exist_ok=True)
    REPORTS.mkdir(parents=True,exist_ok=True)

    con=duckdb.connect()
    try:
        train=get_sample(con,"train",TRAIN_N)
        val=get_sample(con,"validation",VAL_N)
    finally:
        con.close()

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")

    y_train=train["sentiment"].to_numpy()
    y_val=val["sentiment"].to_numpy()

    vectorizer=TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1,2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32
    )

    print("\nFitting TF-IDF on 5 million reviews...")
    start=time.perf_counter()
    X_train=vectorizer.fit_transform(train["text"].astype(str))
    X_val=vectorizer.transform(val["text"].astype(str))
    tfidf_minutes=(time.perf_counter()-start)/60

    print("Feature count:",X_train.shape[1])
    print(f"TF-IDF time: {tfidf_minutes:.2f} min")

    del train,val

    model=LogisticRegression(
        max_iter=700,
        class_weight={
            "negative":1.5,
            "neutral":3.0,
            "positive":0.7
        },
        solver="lbfgs",
        random_state=42
    )

    print("\nTraining Logistic Regression...")
    start=time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train,y_train)

    fit_minutes=(time.perf_counter()-start)/60
    convergence_warning=any(
        issubclass(w.category,ConvergenceWarning) for w in caught
    )

    pred=model.predict(X_val)
    report=classification_report(
        y_val,pred,labels=LABELS,
        output_dict=True,zero_division=0
    )

    result={
        "train_samples":TRAIN_N,
        "validation_samples":VAL_N,
        "features":int(X_train.shape[1]),
        "accuracy":float(accuracy_score(y_val,pred)),
        "macro_f1":float(f1_score(y_val,pred,average="macro")),
        "tfidf_minutes":round(tfidf_minutes,2),
        "fit_minutes":round(fit_minutes,2),
        "iterations":model.n_iter_.tolist(),
        "convergence_warning":convergence_warning,
        "class_f1":{
            label:float(report[label]["f1-score"])
            for label in LABELS
        },
        "classification_report":report,
        "note":"Validation results only; test set remains untouched."
    }

    joblib.dump(
        {"vectorizer":vectorizer,"model":model,"labels":LABELS},
        MODELS/"logistic_regression_5m.joblib"
    )

    (REPORTS/"sentiment_5m_results.json").write_text(
        json.dumps(result,indent=2),encoding="utf-8"
    )

    print("\nAccuracy:",round(result["accuracy"],4))
    print("Macro-F1:",round(result["macro_f1"],4))
    print("Class-wise F1:",result["class_f1"])
    print("Training minutes:",result["fit_minutes"])
    print("Iterations:",result["iterations"])
    print("Convergence warning:",convergence_warning)
    print("Results saved to reports/sentiment_5m_results.json")

if __name__=="__main__":
    main()
