
from pathlib import Path
from collections import Counter
import gzip
import json
import xml.etree.ElementTree as ET

PROJECT_DIR=Path(__file__).resolve().parents[1]
DATASET_DIR=PROJECT_DIR.parent/"Dataset"
REPORT_DIR=PROJECT_DIR/"reports"

def audit_jsonl(path,limit=5000):
    result={"file":path.name,"exists":path.exists()}

    if not path.exists():
        return result

    result["size_gb"]=round(path.stat().st_size/1024**3,3)
    fields=Counter()
    types={}
    ratings=Counter()
    records=0
    errors=0

    try:
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if line_no>limit:
                    break
                if not line.strip():
                    continue
                try:
                    obj=json.loads(line)
                    if not isinstance(obj,dict):
                        errors+=1
                        continue

                    records+=1
                    for key,value in obj.items():
                        fields[key]+=1
                        types.setdefault(key,Counter())[type(value).__name__]+=1

                    if "rating" in obj:
                        ratings[str(obj["rating"])]+=1

                except (json.JSONDecodeError,ValueError):
                    errors+=1

        result.update({
            "sampled_records":records,
            "sample_limit":limit,
            "parse_errors":errors,
            "fields":dict(fields),
            "field_types":{k:dict(v) for k,v in types.items()},
            "rating_distribution":dict(ratings)
        })

    except Exception as exc:
        result["error"]=f"{type(exc).__name__}: {exc}"

    return result

def audit_xml(path):
    result={"file":path.name,"exists":path.exists()}

    if not path.exists():
        return result

    result["size_kb"]=round(path.stat().st_size/1024,2)
    sentences=0
    aspect_terms=0
    polarities=Counter()
    errors=[]

    try:
        for _,elem in ET.iterparse(path,events=("end",)):
            if elem.tag=="sentence":
                sentences+=1
                terms=elem.find("aspectTerms")

                if terms is not None:
                    for term in terms.findall("aspectTerm"):
                        aspect_terms+=1
                        polarities[term.get("polarity","unknown")]+=1

                elem.clear()

        result.update({
            "xml_valid":True,
            "sentences":sentences,
            "aspect_terms":aspect_terms,
            "aspect_polarities":dict(polarities)
        })

    except ET.ParseError as exc:
        errors.append(str(exc))
        result.update({"xml_valid":False,"errors":errors})

    return result

def main():
    results={
        "amazon_reviews":audit_jsonl(
            DATASET_DIR/"Cell_Phones_and_Accessories.jsonl.gz"
        ),
        "amazon_metadata":audit_jsonl(
            DATASET_DIR/"meta_Cell_Phones_and_Accessories.jsonl.gz"
        ),
        "semeval_train":audit_xml(
            DATASET_DIR/"Laptop_Train_v2.xml"
        ),
        "semeval_test":audit_xml(
            DATASET_DIR/"Laptops_Test_Gold.xml"
        )
    }

    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    output=REPORT_DIR/"dataset_audit.json"
    output.write_text(
        json.dumps(results,indent=2),
        encoding="utf-8"
    )

    print(json.dumps(results,indent=2))
    print(f"\nAudit report saved to: {output}")

if __name__=="__main__":
    main()

