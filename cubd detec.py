import wfdb
import os

CUDB_PATH = "C:/Users/eshan/Downloads/ecg-classification-master/ecg-classification-master/CUBD"

def inspect_cudb_annotations():
    for i in range(1, 36):
        record_name = f"cu{i:02d}"
        record_path = os.path.join(CUDB_PATH, record_name)
        try:
            ann = wfdb.rdann(record_path, 'atr')
            notes = set(ann.aux_note)
            print(f"{record_name} – {len(ann.sample)} annotations")
            if notes:
                print(f"  ➤ aux_note: {notes}")
            else:
                print("  ➤ No aux_note found")
        except Exception as e:
            print(f"{record_name} – ⚠️ ERROR: {str(e)}")

if __name__ == "__main__":
    inspect_cudb_annotations()
