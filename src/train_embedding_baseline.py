from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"
CACHE=DATA/"embeddings"
MODELS=ROOT/"models"/"embedding_baselines"
REPORTS=ROOT/"reports"

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)

    train=pd.read_parquet(DATA/"train_sample_500000.parquet")
    val=pd.read_parquet(DATA/"validation_sample_100000.parquet")

    train=train[train["text"].notna() & train["text"].str.strip().ne("")]
    val=val[val["text"].notna() & val["text"].str.strip().ne("")]

    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device:",device)

    encoder=SentenceTransformer(MODEL_NAME,device=device)
    encoder.max_seq_length=256

    def embed(df,name):
        path=CACHE/f"{name}_minilm.npy"

        if path.exists():
            arr=np.load(path,mmap_mode="r")
            if arr.shape==(len(df),encoder.get_sentence_embedding_dimension()):
                print(f"Using cached {name} embeddings:",arr.shape)
                return arr

        print(f"Generating {name} embeddings...")
        start=time.perf_counter()
        vectors=encoder.encode(
            df["text"].astype(str).tolist(),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        np.save(path,vectors)
        print(f"{name} time: {(time.perf_counter()-start)/60:.2f} min")
        del vectors
        return np.load(path,mmap_mode="r")

    X_train=embed(train,"train")
    X_val=embed(val,"validation")
    y_train=train["sentiment"].to_numpy()
    y_val=val["sentiment"].to_numpy()

    del train,val

    model=LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )

    print("\nTraining embedding-based Logistic Regression...")
    start=time.perf_counter()
    model.fit(X_train,y_train)
    fit_minutes=(time.perf_counter()-start)/60

    pred=model.predict(X_val)
    report=classification_report(
        y_val,pred,
        labels=["negative","neutral","positive"],
        output_dict=True,
        zero_division=0
    )

    results={
        "encoder":MODEL_NAME,
        "train_samples":len(y_train),
        "validation_samples":len(y_val),
        "accuracy":float(accuracy_score(y_val,pred)),
        "macro_f1":float(f1_score(y_val,pred,average="macro")),
        "fit_minutes":round(fit_minutes,2),
        "classification_report":report
    }

    joblib.dump(
        {"model":model,"encoder_name":MODEL_NAME,
         "labels":["negative","neutral","positive"]},
        MODELS/"embedding_logistic_regression.joblib"
    )

    (REPORTS/"embedding_baseline_results.json").write_text(
        json.dumps(results,indent=2),encoding="utf-8"
    )

    print("\nAccuracy:",round(results["accuracy"],4))
    print("Macro-F1:",round(results["macro_f1"],4))
    print("Training time:",results["fit_minutes"],"min")
    for label in ["negative","neutral","positive"]:
        print(label,"F1:",round(report[label]["f1-score"],4))

    print("Results saved to reports/embedding_baseline_results.json")

if __name__=="__main__":
    main()
    