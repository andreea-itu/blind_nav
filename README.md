# 🦯 Blind Navigation Assistant

A real-time navigation assistant for blind and visually impaired people.  
Uses **YOLO object detection**, a **remote LLM** (Ollama/Mistral) for scene description, and **Piper** neural TTS (ONNX) with **pw-play** (PipeWire) for speech.

---

## How It Works

```
Webcam or browser camera → YOLOv8 → scene text → WebSocket → Ollama → Piper → speakers
```

1. **Video input** — Local **webcam** (`run.py`) or **browser** JPEG stream (`run_web.py` + `tests/teststayontrails.html`).
2. **Two YOLO models** (every 3rd frame): **COCO** (80 classes) and optional **custom** `best.pt` (puddle, fence, stairs, …).
3. Detections become **human-readable text** with position (*left / ahead / right*) and proximity (*very close / nearby / in the distance*).
4. Scene text is sent over **WebSocket** to a remote machine running **Ollama + Mistral** (throttled to about every **8 seconds** when the scene is non-empty).
5. The LLM returns a **very short** safety-oriented line (see system prompt in `anydesk_server/signaling_client.py`).
6. **Piper** synthesises speech locally; **pw-play** plays it through the laptop speakers.

---

## Architecture

The system runs across **two machines** connected via a WebSocket signaling server:

| Machine | Role | Key components |
|---------|------|----------------|
| **Laptop** | Detection + TTS | `blind/detector.py` (webcam), `blind/web_receiver.py` (browser), `blind/scene_builder.py`, `blind/speaker.py` |
| **Remote server** | LLM | `anydesk_server/signaling_client.py`, Ollama (`mistral:latest`) |
| **Signaling server** | Relay | `wss://signaling.ehb.be` (EHB infrastructure) |

> Full diagrams, threading, browser mode, and ML metrics: [`documentation/architecture.md`](documentation/architecture.md).

---

## Project structure

```
blind_nav/
├── run.py                 ← Local webcam entry: poetry run python run.py
├── run_web.py             ← Browser camera entry: poetry run python run_web.py
├── README.md
├── .gitignore
│
├── blind/
│   ├── detector.py        ← Webcam → YOLO → WebSocket → Piper
│   ├── web_receiver.py    ← Browser JPEG → YOLO → WebSocket → Piper
│   ├── scene_builder.py   ← Detections → scene text
│   └── speaker.py         ← Piper ONNX + pw-play
│
├── anydesk_server/
│   └── signaling_client.py
│
├── models/
│   ├── yolov8n.pt         ← COCO weights
│   ├── best.pt            ← Custom weights (optional)
│   └── piper/             ← Piper voice (e.g. en_US-amy-medium.onnx)
│
├── tests/
│   ├── test_signaling.py
│   ├── test_speaker.py
│   ├── test_piper.py
│   ├── test_best_model.py
│   ├── test_onnx_pt.py
│   ├── teststayontrails.html
│   └── …
│
└── documentation/
    └── architecture.md
```

---

## Getting started

### Prerequisites

- **Python 3.12** (recommended; project is usually managed with **Poetry**)
- **Linux** with **PipeWire** and **`pw-play`** on `PATH` (audio playback)
- **Piper** voice files under `models/piper/` (see `blind/speaker.py` for the expected ONNX path)
- **Ollama** on the remote server with **`mistral:latest`**
- **Webcam** (for `run.py`) or a browser + HTTP server for `teststayontrails.html` (for `run_web.py`)

### Dependencies (overview)

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 |
| `opencv-python` | Camera / decode / preview |
| `websockets` | Signaling client |
| `numpy` | JPEG decode in browser mode |
| `piper-tts` (imports as `piper`) | Neural TTS |
| `ollama` | On the **remote** machine with `signaling_client.py` |

Dataset tooling (e.g. `roboflow`) is only needed for training workflows, not for the default runtime path.

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd blind_nav

# If you use Poetry and have a lockfile / pyproject at the repo root:
poetry install

# Otherwise install the runtime stack explicitly, for example:
# pip install ultralytics opencv-python websockets numpy piper-tts
```

### Running

#### Laptop — local webcam

```bash
poetry run python run.py
```

Press **`q`** in the OpenCV window to quit.

#### Laptop — browser camera

Serve the `tests/` folder over HTTP(S), open `teststayontrails.html`, start the camera, then:

```bash
poetry run python run_web.py
```

Details and signaling caveats: [`documentation/architecture.md`](documentation/architecture.md) (browser mode section).

#### Remote server — Ollama + WebSocket listener

```bash
cd anydesk_server
python signaling_client.py --token YOUR_BEARER_TOKEN
```

Use `--skip-test` to skip the local Ollama smoke test, and `--room` if you override the default room path (see `signaling_client.py`).

---

## Custom YOLO model

The custom model was trained on a combined **Roboflow** dataset using **Google Colab** (T4 GPU).

### Detected classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | puddle | Water puddles on pavement |
| 1 | Fence | Fences and barriers |
| 2 | Fence Anomaly | Damaged or broken fences |
| 3 | stairs | Staircases |
| 4 | up_steps | Steps going up |
| 5 | down_steps | Steps going down |

### Training metrics (snapshot)

| Metric | Value |
|--------|-------|
| Precision | **0.824** |
| Recall | **0.707** |
| mAP50 | **0.744** |
| mAP50-95 | **0.430** |
| Inference speed | **4.3 ms/image** (T4 GPU) |

> Per-class breakdown: [`documentation/architecture.md`](documentation/architecture.md#custom-model--training-metrics).

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Object detection | YOLOv8 Nano (Ultralytics) |
| LLM | Ollama + `mistral:latest` |
| WebSocket | `websockets` + EHB signaling server |
| Text-to-speech | Piper (ONNX) + `pw-play` (PipeWire) |
| Camera | OpenCV (`cv2.VideoCapture`) |
| Training | Google Colab (typical) |
| Datasets | Roboflow |
| Package manager | Poetry (typical) |

---

## Example output

```
Detected: person (ahead of you, very close), car (to your left, in the distance),
puddle (to your right, nearby)

LLM says: Person very close ahead; puddle right; car far left.
```

🔊 *Piper speaks the LLM line through the system audio device.*
