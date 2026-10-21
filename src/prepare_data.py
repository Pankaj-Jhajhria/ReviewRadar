
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/"Dataset"/"Cell_Phones_and_Accessories.jsonl.gz"
OUT=ROOT/"data"/"processed"/"amazon_reviews_core.parquet"
TEMP=OUT.with_name("amazon_reviews_core.tmp.parquet")
BATCH_SIZE=50_000

SCHEMA=pa.schema([
    ("rating",pa.float32()),
    ("title",pa.string()),
    ("text",pa.string()),
    ("asin",pa.string()),
    ("parent_asin",pa.string()),
    ("timestamp_ms",pa.int64()),
    ("helpful_vote",pa.int32()),
    ("verified_purchase",pa.bool_()),
])

def as_number(value,kind):
    try:
        return kind(value) if value is not None else None
    except (ValueError,TypeError,OverflowError):
        return None

def flush_batch(writer,rows):
    if rows:
        writer.write_table(
            pa.Table.from_pylist(rows,schema=SCHEMA)
        )
        rows.clear()

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    OUT.parent.mkdir(parents=True,exist_ok=True)

    if OUT.exists():
        raise FileExistsError(
            f"{OUT} already exists. Verify it before rerunning."
        )
    if TEMP.exists():
        raise FileExistsError(
            f"{TEMP} already exists. Inspect it before rerunning."
        )

    rows=[]
    ratings=Counter()
    lines=valid=errors=missing_text=0
    writer=None
    start=time.perf_counter()

    try:
        writer=pq.ParquetWriter(
            TEMP,SCHEMA,compression="zstd"
        )

        with gzip.open(SOURCE,"rt",encoding="utf-8") as f:
            for line in f:
                lines+=1

                try:
                    obj=json.loads(line)
                    if not isinstance(obj,dict):
                        errors+=1
                        continue
                except (json.JSONDecodeError,ValueError):
                    errors+=1
                    continue

                rating=as_number(obj.get("rating"),float)
                text=obj.get("text")
                title=obj.get("title")

                if not isinstance(text,str) or not text.strip():
                    missing_text+=1

                if rating is not None:
                    ratings[str(rating)]+=1

                rows.append({
                    "rating":rating,
                    "title":title if isinstance(title,str) else None,
                    "text":text if isinstance(text,str) else None,
                    "asin":obj.get("asin"),
                    "parent_asin":obj.get("parent_asin"),
                    "timestamp_ms":as_number(obj.get("timestamp"),int),
                    "helpful_vote":as_number(obj.get("helpful_vote"),int),
                    "verified_purchase":(
                        obj.get("verified_purchase")
                        if isinstance(obj.get("verified_purchase"),bool)
                        else None
                    ),
                })
                valid+=1

                if len(rows)>=BATCH_SIZE:
                    flush_batch(writer,rows)

                if lines%1_000_000==0:
                    elapsed=(time.perf_counter()-start)/60
                    print(
                        f"Scanned {lines:,} lines | "
                        f"Valid {valid:,} | "
                        f"Errors {errors:,} | "
                        f"{elapsed:.1f} min"
                    )

        flush_batch(writer,rows)
        writer.close()
        writer=None
        os.replace(TEMP,OUT)

    finally:
        if writer is not None:
            writer.close()

    print("\nFULL DATASET CONVERSION COMPLETE")
    print(f"Lines scanned: {lines:,}")
    print(f"Valid records: {valid:,}")
    print(f"Invalid records: {errors:,}")
    print(f"Missing review text: {missing_text:,}")
    print(f"Rating distribution: {dict(ratings)}")
    print(f"Output: {OUT}")
    print(f"Output size: {OUT.stat().st_size/1024**3:.2f} GB")
    print(f"Time: {(time.perf_counter()-start)/60:.1f} minutes")

if __name__=="__main__":
    main()
