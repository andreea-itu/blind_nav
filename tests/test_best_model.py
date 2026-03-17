# Run best.pt on all images in tests/test_images/
# Usage: cd blind_nav && poetry run python tests/test_best_model.py


import glob
from ultralytics import YOLO

model = YOLO("models/best.pt")
images = sorted(glob.glob("tests/test_images/*.*"))

for path in images:
    results = model(path, conf=0.3)
    for r in results:
        for box in r.boxes:
            name = model.names[int(box.cls[0])]
            print(f"  {name}: {float(box.conf[0]):.0%}")