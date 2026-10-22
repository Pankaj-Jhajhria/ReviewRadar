
from pathlib import Path
import json
import time

import duckdb
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"amazon_reviews_labeled_split.parquet"
MODEL=ROOT/"models"/"sentiment_baselines"/"logistic_regression_5m.joblib"
OUT=ROOT/"reports"/"sentiment_final_test.json"

LABELS=["negative","neutral","positive"]
BATCH_SIZE=50_000

def sql_path(path):
    return "'"+path.resolve().as_posix().replace("'","''")+"'"

def main():
    if not DATA.exists() or not MODEL.exists():
        raise FileNotFoundError("Dataset or trained 5M model is missing.")

    bundle=joblib.load(MODEL)
    vectorizer=bundle["vectorizer"]
    model=bundle["model"]

    con=duckdb.connect()
    source=f"read_parquet({sql_path(DATA)})"
    condition="""
        split='test'
        AND text IS NOT NULL
        AND TRIM(text)<>''
        AND sentiment IN ('negative','neutral','positive')
    """

    total=con.execute(
        f"SELECT COUNT(*) FROM {source} WHERE {condition}"
    ).fetchone()[0]

    if total==0:
        con.close()
        raise RuntimeError("No eligible test reviews found.")

    cursor=con.execute(
        f"SELECT text,sentiment FROM {source} WHERE {condition}"
    )

    matrix=np.zeros((3,3),dtype=np.int64)
    start=time.perf_counter()

    try:
        with tqdm(
            total=total,
            desc="Evaluating untouched test set",
            unit="reviews"
        ) as progress:
            while True:
                rows=cursor.fetchmany(BATCH_SIZE)
                if not rows:
                    break

                texts=[r[0] for r in rows]
                actual=[r[1] for r in rows]

                X=vectorizer.transform(texts)
                predicted=model.predict(X)

                matrix+=confusion_matrix(
                    actual,predicted,labels=LABELS
                )
                progress.update(len(rows))

    finally:
        con.close()

    tp=np.diag(matrix).astype(float)
    support=matrix.sum(axis=1).astype(float)
    predicted_count=matrix.sum(axis=0).astype(float)

    precision=np.divide(
        tp,predicted_count,
        out=np.zeros(3),where=predicted_count!=0
    )
    recall=np.divide(
        tp,support,
        out=np.zeros(3),where=support!=0
    )
    f1=np.divide(
        2*precision*recall,
        precision+recall,
        out=np.zeros(3),
        where=(precision+recall)!=0
    )

    accuracy=float(tp.sum()/matrix.sum())
    macro_f1=float(f1.mean())
    weighted_f1=float(np.average(f1,weights=support))
    elapsed=time.perf_counter()-start

    result={
        "model":"logistic_regression_5m",
        "test_reviews":int(matrix.sum()),
        "accuracy":accuracy,
        "macro_f1":macro_f1,
        "weighted_f1":weighted_f1,
        "class_metrics":{
            label:{
                "precision":float(precision[i]),
                "recall":float(recall[i]),
                "f1":float(f1[i]),
                "support":int(support[i])
            }
            for i,label in enumerate(LABELS)
        },
        "confusion_matrix_actual_rows_predicted_columns":{
            label:{
                pred_label:int(matrix[i,j])
                for j,pred_label in enumerate(LABELS)
            }
            for i,label in enumerate(LABELS)
        },
        "evaluation_seconds":round(elapsed,2)
    }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")

    print("\nFINAL TEST RESULTS")
    print(f"Test reviews: {matrix.sum():,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    for label,metrics in result["class_metrics"].items():
        print(f"{label}: F1={metrics['f1']:.4f}")

    print(f"Evaluation time: {elapsed/60:.2f} minutes")
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved: {OUT}")

if __name__=="__main__":
    main()
