
from pathlib import Path
import json
import time

import duckdb
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"amazon_reviews_labeled_split.parquet"
SAMPLES=ROOT/"data"/"processed"
MODELS=ROOT/"models"/"sentiment_baselines"
REPORTS=ROOT/"reports"

TRAIN_N=500_000
VAL_N=100_000
LABELS=["negative","neutral","positive"]

def sql_path(path):
    return "'"+path.resolve().as_posix().replace("'","''")+"'"

def load_sample(con,split,limit):
    cache=SAMPLES/f"{split}_sample_{limit}.parquet"

    if cache.exists():
        print(f"Loading cached sample: {cache.name}")
        return __import__("pandas").read_parquet(cache)

    print(f"Selecting {limit:,} {split} reviews...")
    query=f"""
        SELECT text,sentiment,parent_asin,asin,timestamp_ms,rating
        FROM read_parquet({sql_path(DATA)})
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
    df.to_parquet(cache,index=False)
    return df

def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    MODELS.mkdir(parents=True,exist_ok=True)
    REPORTS.mkdir(parents=True,exist_ok=True)

    con=duckdb.connect()
    try:
        train=load_sample(con,"train",TRAIN_N)
        val=load_sample(con,"validation",VAL_N)
    finally:
        con.close()

    if train.empty or val.empty:
        raise RuntimeError("Training or validation sample is empty.")

    print(f"Training reviews: {len(train):,}")
    print(f"Validation reviews: {len(val):,}")

    X_train_text=train["text"].tolist()
    y_train=train["sentiment"].to_numpy()
    X_val_text=val["text"].tolist()
    y_val=val["sentiment"].to_numpy()

    del train,val

    print("\nBuilding TF-IDF features...")
    vectorizer=TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1,2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32
    )

    start=time.perf_counter()
    X_train=vectorizer.fit_transform(X_train_text)
    X_val=vectorizer.transform(X_val_text)

    print(f"Features: {X_train.shape[1]:,}")
    print(f"Training matrix: {X_train.shape}")
    print(f"TF-IDF time: {(time.perf_counter()-start)/60:.2f} min")

    del X_train_text,X_val_text

    models={
        "naive_bayes":MultinomialNB(alpha=0.1),
        "logistic_regression":LogisticRegression(
            max_iter=100,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ),
        "linear_svm":LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            dual="auto",
            random_state=42
        )
    }

    weights=compute_sample_weight("balanced",y_train)
    results={}

    for name,model in models.items():
        print(f"\nTraining {name}...")
        start=time.perf_counter()

        if name=="naive_bayes":
            model.fit(X_train,y_train,sample_weight=weights)
        else:
            model.fit(X_train,y_train)

        fit_seconds=time.perf_counter()-start

        start=time.perf_counter()
        predictions=model.predict(X_val)
        predict_seconds=time.perf_counter()-start

        report=classification_report(
            y_val,predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0
        )

        results[name]={
            "accuracy":float(accuracy_score(y_val,predictions)),
            "macro_f1":float(f1_score(
                y_val,predictions,
                labels=LABELS,average="macro"
            )),
            "fit_minutes":round(fit_seconds/60,2),
            "predict_seconds":round(predict_seconds,2),
            "classification_report":report
        }

        joblib.dump(
            {"vectorizer":vectorizer,"model":model,"labels":LABELS},
            MODELS/f"{name}.joblib"
        )

        print(
            f"Accuracy={results[name]['accuracy']:.4f} | "
            f"Macro-F1={results[name]['macro_f1']:.4f} | "
            f"Fit={fit_seconds/60:.2f} min"
        )

    best=max(results,key=lambda k:results[k]["macro_f1"])

    output={
        "training_sample":TRAIN_N,
        "validation_sample":VAL_N,
        "features":int(X_train.shape[1]),
        "selection_metric":"validation_macro_f1",
        "best_pilot_model":best,
        "results":results,
        "note":"Pilot results only; test set not used."
    }

    (REPORTS/"sentiment_baseline_results.json").write_text(
        json.dumps(output,indent=2),
        encoding="utf-8"
    )

    print(f"\nBest pilot model: {best}")
    print(f"Results: {REPORTS/'sentiment_baseline_results.json'}")
    print(f"Models saved in: {MODELS}")

if __name__=="__main__":
    main()
