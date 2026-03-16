import base64
import json
import os
from pathlib import Path

NOTEBOOK_PATH = Path("Tetris.ipynb")
OUT_DIR = Path("assets/plots/notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

saved = []

for idx, cell in enumerate(nb.get("cells", []), start=1):
    if cell.get("cell_type") != "code":
        continue

    outputs = cell.get("outputs", [])
    image_index = 0

    for out in outputs:
        data = out.get("data", {}) if isinstance(out, dict) else {}
        img = data.get("image/png")
        if not img:
            continue

        image_index += 1
        if isinstance(img, list):
            b64 = "".join(img)
        else:
            b64 = img

        filename = OUT_DIR / f"cell_{idx:02d}_plot_{image_index:02d}.png"
        with filename.open("wb") as pf:
            pf.write(base64.b64decode(b64))
        saved.append(str(filename))

print(f"Extracted {len(saved)} PNG plot(s) from notebook outputs.")
for path in saved:
    print(f"- {path}")
