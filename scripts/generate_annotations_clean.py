import json
import os
import os.path as osp
from glob import glob
import pandas as pd

# ✅ Label classes used in your project
classes = ["N", "V", "\\", "R", "L", "A", "!", "E"]
lead = "MLII"
extension = "npy"
val_size = 0.1
random_state = 7

# ✅ Adjust this if your path differs
data_path = "./clean_1D_dataset/*/*/*/*.npy"
output_path = osp.abspath("./clean_1D_dataset")

if __name__ == "__main__":
    dataset = []
    files = glob(data_path, recursive=True)
    print(f"🔍 Found {len(files)} .npy files")

    if not files:
        print("⚠️ No .npy files found. Check `data_path` or run clean_generate_1d.py again.")
        exit()

    for file in files:
        parts = file.replace("\\", "/").split("/")[-5:]
        if len(parts) != 5:
            print(f"⚠️ Skipping malformed path: {file}")
            continue

        name, lead_used, label, filename = parts[1], parts[2], parts[3], parts[4]
        dataset.append({
            "name": name,
            "lead": lead_used,
            "label": label,
            "filename": osp.splitext(filename)[0],
            "path": file,
        })

    data = pd.DataFrame(dataset)
    data = data[data["lead"] == lead]
    data = data[data["label"].isin(classes)]
    data = data.sample(frac=1, random_state=random_state)

    if data.empty:
        print("⚠️ No usable samples found with specified lead and labels.")
        exit()

    val_ids = []
    for cl in classes:
        val_ids.extend(
            data[data["label"] == cl]
            .sample(frac=val_size, random_state=random_state)
            .index,
        )

    val = data.loc[val_ids]
    train = data.drop(val.index)
    os.makedirs(output_path, exist_ok=True)

    train.to_json(osp.join(output_path, "train.json"), orient="records")
    val.to_json(osp.join(output_path, "val.json"), orient="records")

    label_map = {label: idx for idx, label in enumerate(sorted(train.label.unique()))}
    with open(osp.join(output_path, "class-mapper.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("✅ train.json, val.json, and class-mapper.json generated successfully.")
