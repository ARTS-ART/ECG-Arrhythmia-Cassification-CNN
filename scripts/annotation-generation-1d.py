import json
import os
import os.path as osp
from glob import glob
import pandas as pd

# Classes of interest
classes = ["N", "V", "\\", "R", "L", "A", "!", "E"]
lead = "MLII"
extension = "npy"  # or 'png' for 2D
val_size = 0.1  # 10% for validation
random_state = 7

# Build absolute path
data_glob = osp.abspath(osp.join("..", "data", "*", "*", "*", "*", f"*.{extension}"))
files = glob(data_glob, recursive=True)
print(f"🔍 Found {len(files)} data files.")

dataset = []

for file in files:
    # Cross-platform split
    parts = file.replace("\\", "/").split("/")
    try:
        name, lead_name, label, filename = parts[-5], parts[-4], parts[-3], parts[-1]
        if label not in classes or lead_name != lead:
            continue
        dataset.append({
            "name": name,
            "lead": lead_name,
            "label": label,
            "filename": osp.splitext(filename)[0],
            "path": file,
        })
    except Exception as e:
        print(f"⚠️ Skipping: {file} | Error: {e}")

data = pd.DataFrame(dataset)
data = data.sample(frac=1, random_state=random_state)

# Validation split
val_ids = []
for cl in classes:
    val_ids.extend(
        data[data["label"] == cl]
        .sample(frac=val_size, random_state=random_state)
        .index
    )

val = data.loc[val_ids]
train = data[~data.index.isin(val.index)]

# Output folder
output_path = osp.abspath("..")
train.to_json(osp.join(output_path, "train.json"), orient="records")
val.to_json(osp.join(output_path, "val.json"), orient="records")

# Create class mapper
mapper = {label: idx for idx, label in enumerate(sorted(train["label"].unique()))}
with open(osp.join(output_path, "class-mapper.json"), "w") as f:
    json.dump(mapper, f, indent=2)

print(f"✅ Saved: {len(train)} train, {len(val)} val, {len(mapper)} classes.")
