import unicodedata
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import os

DATA_ROOT = r"C:\Users\User\.vscode\mirea_asset\corpus"  # universe.csv, manifest.jsonl, raw/ 가 있는 경로로 수정
random.seed(42)


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------
# 0. 기본 로드
# ---------------------------------------------------------
universe = pd.read_csv(
    os.path.join(DATA_ROOT, "universe.csv"),
    dtype={"corp_code": str, "stock_code": str},
    encoding="utf-8-sig",
)

records = []
with open(os.path.join(DATA_ROOT, "manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

manifest = pd.DataFrame(records)

# manifest에 있는 corp_name
name_from_manifest = manifest.loc[manifest["corp_name"].str.contains("삼성전자"), "corp_name"].iloc[0]

# 실제 raw/periodic 안의 폴더명
actual_folders = os.listdir(Path(DATA_ROOT) / "raw" / "periodic")

for f in actual_folders:
    if unicodedata.normalize("NFC", f) == unicodedata.normalize("NFC", name_from_manifest):
        print("정규화하면 일치:", repr(f), repr(name_from_manifest))
        print("정규화 전 동일?:", f == name_from_manifest)  # 여기서 False면 원인 확정