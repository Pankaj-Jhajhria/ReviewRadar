
from pathlib import Path
import os
import time

import duckdb
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"
SOURCE=DATA/"amazon_reviews_labeled_split.parquet"
MODEL=ROOT/"models"/"sentiment_baselines"/"logistic_regression_5m.joblib"
OUTPUT=DATA/"sentiment_predictions.parquet"
TEMP=DATA/"sentiment_predictions.tmp.parquet"

BATCH_SIZE=50_000

SCHEMA=pa.schema([
    ("review_id",pa.int64()),
    ("asin",pa.string()),
    ("parent_asin",pa.string()),
    ("timestamp_ms",pa.int64()),
    ("rating",pa.float32()),
    ("rating_sentiment",pa.string()),
    ("split",pa.string()),
    ("title",pa.string()),
    ("text",pa.string()),
    ("predicted_sentiment",pa.string()),
    ("prediction_confidence",pa.float32()),
    ("p_negative",pa.float32()),
    ("p_neutral",pa.float32()),
    ("p_positive",pa.float32()),
])

def sql_path(path):
    return "'"+path.resolve().as_posix().replace("'","''")+"'"

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not MODEL.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL}"
        )
    if OUTPUT.exists() or TEMP.exists():
        raise FileExistsError(
            "Output or temporary file already exists. "
            "Inspect it before rerunning; do not overwrite it."
        )

    DATA.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load(MODEL)
    vectorizer=bundle["vectorizer"]
    model=bundle["model"]

    con=duckdb.connect()
    writer=None
    start=time.perf_counter()
    review_id=0
    predicted_count=0
    missing_text=0

    try:
        source=f"read_parquet({sql_path(SOURCE)})"
        total=con.execute(
            f"SELECT COUNT(*) FROM {source}"
        ).fetchone()[0]

        cursor=con.execute(f"""
            SELECT rating,title,text,asin,parent_asin,
                   timestamp_ms,sentiment,split
            FROM {source}
        """)

        writer=pq.ParquetWriter(
            TEMP,SCHEMA,compression="zstd"
        )
        class_index={
            label:i for i,label in enumerate(model.classes_)
        }

        with tqdm(
            total=total,
            desc="Predicting all reviews",
            unit="reviews",
            dynamic_ncols=True,
            mininterval=1
        ) as progress:

            while True:
                batch=cursor.fetchmany(BATCH_SIZE)
                if not batch:
                    break

                texts=[]
                positions=[]

                for i,row in enumerate(batch):
                    text=row[2]
                    if isinstance(text,str) and text.strip():
                        positions.append(i)
                        texts.append(text)
                    else:
                        missing_text+=1

                predictions=[None]*len(batch)
                confidences=[None]*len(batch)
                probabilities=[None]*len(batch)

                if texts:
                    X=vectorizer.transform(texts)
                    preds=model.predict(X)
                    probs=model.predict_proba(X)

                    for j,pos in enumerate(positions):
                        predictions[pos]=str(preds[j])
                        confidences[pos]=float(probs[j].max())
                        probabilities[pos]={
                            label:float(probs[j,index])
                            for label,index in class_index.items()
                        }

                    predicted_count+=len(texts)

                output_rows=[]

                for i,row in enumerate(batch):
                    rating,title,text,asin,parent_asin,ts,label,split=row
                    prob=probabilities[i] or {}

                    output_rows.append({
                        "review_id":review_id+i+1,
                        "asin":asin,
                        "parent_asin":parent_asin,
                        "timestamp_ms":ts,
                        "rating":rating,
                        "rating_sentiment":label,
                        "split":split,
                        "title":title,
                        "text":text,
                        "predicted_sentiment":predictions[i],
                        "prediction_confidence":confidences[i],
                        "p_negative":prob.get("negative"),
                        "p_neutral":prob.get("neutral"),
                        "p_positive":prob.get("positive"),
                    })

                table=pa.Table.from_pylist(
                    output_rows,schema=SCHEMA
                )
                writer.write_table(table)
                review_id+=len(batch)
                progress.update(len(batch))

        writer.close()
        writer=None
        os.replace(TEMP,OUTPUT)

    finally:
        if writer is not None:
            writer.close()
        con.close()

    print("\nFULL-DATASET INFERENCE COMPLETE")
    print(f"Total reviews: {review_id:,}")
    print(f"Predictions generated: {predicted_count:,}")
    print(f"Missing/empty text: {missing_text:,}")
    print(f"Output: {OUTPUT}")
    print(f"Output size: {OUTPUT.stat().st_size/1024**3:.2f} GB")
    print(f"Elapsed: {(time.perf_counter()-start)/60:.2f} minutes")

if __name__=="__main__":
    main()

