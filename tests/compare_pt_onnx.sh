#!/usr/bin/env bash
# Compare models/best.pt with models/best.onnx from the blind_nav project root.
#
# Usage:
#   chmod +x tests/compare_pt_onnx.sh
#   ./tests/compare_pt_onnx.sh              # run all checks
#   ./tests/compare_pt_onnx.sh shapes       # ONNX I/O shapes only
#   ./tests/compare_pt_onnx.sh one          # single-image PT vs ONNX
#   ./tests/compare_pt_onnx.sh script       # tests/test_onnx_pt.py
#   ./tests/compare_pt_onnx.sh folder       # box-count diff across test_images
#   ./tests/compare_pt_onnx.sh yolo         # yolo predict for both (writes runs/detect)
#   ./tests/compare_pt_onnx.sh pytorch      # test_best_model.py (PT only)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POETRY="${POETRY:-poetry}"
CONF="${CONF:-0.3}"
SAMPLE_IMG="${SAMPLE_IMG:-tests/test_images/puddle_1.jpg}"

section() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

cmd_shapes() {
  section "ONNX session: input / output tensor shapes"
  $POETRY run python -c "
import onnxruntime as ort
s = ort.InferenceSession('models/best.onnx', providers=['CPUExecutionProvider'])
print('inputs:', [(i.name, list(i.shape)) for i in s.get_inputs()])
print('outputs:', [(o.name, list(o.shape)) for o in s.get_outputs()])
"
}

cmd_one() {
  section "Single image: Ultralytics best.pt vs best.onnx (conf=${CONF})"
  $POETRY run python -c "
from ultralytics import YOLO
conf = float('${CONF}')
img = '${SAMPLE_IMG}'
pt = YOLO('models/best.pt')
nx = YOLO('models/best.onnx')
r1 = pt(img, conf=conf, verbose=False)[0]
r2 = nx(img, conf=conf, verbose=False)[0]
print('PT: ', [(pt.names[int(b.cls[0])], float(b.conf[0])) for b in r1.boxes])
print('ONNX:', [(nx.names[int(b.cls[0])], float(b.conf[0])) for b in r2.boxes])
"
}

cmd_script() {
  section "tests/test_onnx_pt.py (shapes + same-image compare)"
  $POETRY run python tests/test_onnx_pt.py
}

cmd_folder() {
  section "All tests/test_images/*.jpg: print only when PT vs ONNX box counts differ"
  $POETRY run python -c "
from pathlib import Path
from ultralytics import YOLO
conf = float('${CONF}')
pt = YOLO('models/best.pt')
nx = YOLO('models/best.onnx')
any_mismatch = False
for p in sorted(Path('tests/test_images').glob('*.jpg')):
    a = len(pt(str(p), conf=conf, verbose=False)[0].boxes)
    b = len(nx(str(p), conf=conf, verbose=False)[0].boxes)
    if a != b:
        any_mismatch = True
        print(f'{p.name}: PT boxes={a}  ONNX boxes={b}')
if not any_mismatch:
    print('No box-count mismatches across *.jpg (for conf=' + str(conf) + ').')
"
}

cmd_yolo() {
  section "yolo predict (writes under runs/detect/ — conf=${CONF})"
  echo "Model: best.pt"
  $POETRY run yolo predict model=models/best.pt source="${SAMPLE_IMG}" conf="${CONF}" save=False
  echo ""
  echo "Model: best.onnx"
  $POETRY run yolo predict model=models/best.onnx source="${SAMPLE_IMG}" conf="${CONF}" save=False
}

cmd_pytorch() {
  section "tests/test_best_model.py (PyTorch best.pt only, conf 0.25 and 0.3)"
  $POETRY run python tests/test_best_model.py
}

cmd_all() {
  echo "Project root: $ROOT"
  echo "Sample image: $SAMPLE_IMG  CONF=$CONF"
  cmd_shapes
  cmd_one
  cmd_script
  cmd_folder
  section "Optional: run './tests/compare_pt_onnx.sh yolo' or './tests/compare_pt_onnx.sh pytorch' for more"
}

case "${1:-all}" in
  shapes)  cmd_shapes ;;
  one)     cmd_one ;;
  script)  cmd_script ;;
  folder)  cmd_folder ;;
  yolo)    cmd_yolo ;;
  pytorch) cmd_pytorch ;;
  all)     cmd_all ;;
  help|-h|--help)
    sed -n '1,20p' "$0"
    ;;
  *)
    echo "Unknown command: $1" >&2
    echo "Use: $0 [all|shapes|one|script|folder|yolo|pytorch|help]" >&2
    exit 1
    ;;
esac
