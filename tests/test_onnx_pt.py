# Verify exported best.onnx output tensor shape
import onnxruntime as ort
sess = ort.InferenceSession('models/best.onnx', providers=['CPUExecutionProvider'])
for o in sess.get_outputs():
    print(o.name, o.shape)


# Compare best.pt vs re-exported best.onnx

from ultralytics import YOLO
conf = 0.3
img = 'tests/test_images/puddle_1.jpg'
mpt = YOLO('models/best.pt')
r1 = mpt(img, conf=conf, verbose=False)[0]
monnx = YOLO('models/best.onnx')
r2 = monnx(img, conf=conf, verbose=False)[0]
print('PT:', [(mpt.names[int(b.cls[0])], float(b.conf[0])) for b in r1.boxes])
print('ONNX:', [(monnx.names[int(b.cls[0])], float(b.conf[0])) for b in r2.boxes])


import numpy as np
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

img_path = 'tests/test_images/puddle_1.jpg'
conf = 0.3

# PyTorch
m = YOLO('models/best.pt')
r = m(img_path, conf=conf, verbose=False)[0]
print('=== best.pt ===')
print('names:', m.names)
for box in r.boxes:
    print(int(box.cls[0]), m.names[int(box.cls[0])], float(box.conf[0]))

# ONNX
sess = ort.InferenceSession('models/best.onnx', providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print('=== best.onnx ===')
print('input:', inp.name, inp.shape, inp.type)
print('output:', out.name, out.shape, out.type)

# Preprocess like ultralytics: need same as predict
# Use ultralytics to export prediction path - or manual letterbox
from ultralytics.engine.results import Results
# Simpler: use model.predict with same but onnx - ultralytics can run onnx
monnx = YOLO('models/best.onnx')
r2 = monnx(img_path, conf=conf, verbose=False)[0]
print('=== YOLO(best.onnx) via ultralytics ===')
for box in r2.boxes:
    print(int(box.cls[0]), monnx.names[int(box.cls[0])], float(box.conf[0]))
if len(r2.boxes) == 0:
    print('(no detections)')
