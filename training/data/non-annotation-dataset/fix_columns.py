"""Quick fix: rename avg_score -> similarity_score, drop num_votes/num_yes/num_no."""
import csv
from pathlib import Path

DROP = {"num_votes", "num_yes", "num_no"}
RENAME = {"avg_score": "similarity_score"}

for name in ["train.csv", "val.csv", "test.csv"]:
    path = Path(name)
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames
        for row in reader:
            rows.append(row)

    new_fields = [RENAME.get(f, f) for f in old_fields if f not in DROP]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            new_row = {}
            for f in old_fields:
                if f in DROP:
                    continue
                new_row[RENAME.get(f, f)] = row[f]
            writer.writerow(new_row)

    print(f"  {name}: {len(rows)} rows, columns: {new_fields}")

print("Done.")
