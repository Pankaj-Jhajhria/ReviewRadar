
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data"/"processed"/"amazon_reviews_core.parquet"
OUTPUT=ROOT/"data"/"processed"/"amazon_reviews_labeled_split.parquet"

def sql_path(path):
    return "'"+path.as_posix().replace("'","''")+"'"

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Inspect it before rerunning."
        )

    con=duckdb.connect()

    print("Creating sentiment labels and product-level splits...")

    con.execute(f"""
        COPY (
            SELECT *,
                CASE
                    WHEN rating<=2 THEN 'negative'
                    WHEN rating=3 THEN 'neutral'
                    WHEN rating>=4 THEN 'positive'
                    ELSE 'unknown'
                END AS sentiment,
                CASE
                    WHEN hash(
                        COALESCE(
                            NULLIF(parent_asin,''),
                            NULLIF(asin,''),
                            '__missing_id__'
                        )
                    ) % 100 < 80 THEN 'train'
                    WHEN hash(
                        COALESCE(
                            NULLIF(parent_asin,''),
                            NULLIF(asin,''),
                            '__missing_id__'
                        )
                    ) % 100 < 90 THEN 'validation'
                    ELSE 'test'
                END AS split
            FROM read_parquet({sql_path(SOURCE)})
        )
        TO {sql_path(OUTPUT)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)

    result=con.execute(f"""
        SELECT split,sentiment,COUNT(*) AS reviews,
               SUM(CASE
                   WHEN text IS NULL OR TRIM(text)=''
                   THEN 1 ELSE 0
               END) AS missing_text
        FROM read_parquet({sql_path(OUTPUT)})
        GROUP BY split,sentiment
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                ELSE 3
            END,
            sentiment
    """).fetchdf()

    leakage=con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT COALESCE(
                NULLIF(parent_asin,''),
                NULLIF(asin,''),
                '__missing_id__'
            ) AS product_key
            FROM read_parquet({sql_path(OUTPUT)})
            GROUP BY product_key
            HAVING COUNT(DISTINCT split)>1
        )
    """).fetchone()[0]

    print("\\nSplit summary:")
    print(result.to_string(index=False))
    print(f"\\nProducts leaking across splits: {leakage}")
    print(f"Output: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size/1024**3:.2f} GB")

    con.close()

if __name__=="__main__":
    main()
