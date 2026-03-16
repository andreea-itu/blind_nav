# 🦯 Blind Navigation Assistant

A real-time navigation assistant for blind and visually impaired people.  
Uses **YOLO object detection**, a **remote LLM** (Ollama/Mistral) for scene description, and **text-to-speech** to guide the user safely.

---

## How It Works

```
Webcam → YOLOv8 Detection → Scene Text → WebSocket → Ollama LLM → Speech
```

1. **Webcam** captures live video on the laptop
2. **Two YOLO models** detect objects in each frame:
   - **COCO model** — 80 common classes (person, car, bicycle, traffic light …)
   - **Custom model** — 6 domain-specific classes (puddle, fence, stairs …)
3. Detections are converted to **human-readable text** with position (*left / ahead / right*) and proximity (*very close / nearby / in the distance*)
4. The scene text is sent via **WebSocket** to a remote server running **Ollama + Mistral**
5. The LLM generates a short, safety-focused description
6. The description is **spoken aloud** through the laptop speakers

---

## Architecture

The system runs across **two machines** connected via a WebSocket signaling server:

| Machine | Role | Key Components |
|---------|------|----------------|
| **Laptop** | Detection + Camera + TTS | `detector.py`, `scene_builder.py`, `speaker.py` |
| **Remote Server** | LLM inference | `signaling_client.py`, Ollama (Mistral) |
| **Signaling Server** | Message relay | `wss://signaling.ehb.be` (EHB infrastructure) |

> 📖 See [`documentation/architecture.md`](documentation/architecture.md) for detailed architecture diagrams, data flow, threading model, and ML metrics.

---

## Project Structure

```
blind/
├── run.py                        ← Entry point
├── .gitignore
├── README.md
│
├── blind/                    ← Core application package
│   ├── __init__.py
│   ├── detector.py               ← Main loop: camera → YOLO → WebSocket → TTS
│   ├── scene_builder.py          ← Converts detections → human-readable text
│   └── speaker.py                ← Text-to-speech via spd-say (Linux)
│
├── anydesk_server/               ← Runs on the remote server
│   └── signaling_client.py       ← WebSocket client + Ollama integration
│
├── models/                       ← YOLO model weights
│   ├── yolov8n.pt                ← Pre-trained COCO (80 classes, 6.3 MB)
│   └── best.pt                   ← Custom-trained (6 classes, 6.0 MB)
│
├── tests/                        ← Test scripts
│   ├── test_signaling.py         ← WebSocket + Ollama connectivity tests
│   └── test_speaker.py           ← TTS tests
│
└── documentation/                ← Architecture docs & project notes
    ├── architecture.md           ← Full technical documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.12** (managed via Poetry)
- **Poetry** — [install instructions](https://python-poetry.org/docs/#installation)
- **Linux** with `spd-say` (speech-dispatcher) for TTS
- **Ollama** running on the remote server with the `mistral:latest` model
- A webcam

### Key Dependencies

Managed via `pyproject.toml` at the repository root:

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.3 | YOLOv8 object detection |
| `opencv-python` | ≥ 4.4 | Webcam capture & display |
| `websockets` | ≥ 16.0 | WebSocket communication with signaling server |
| `ollama` | ≥ 0.6 | Ollama Python client (remote server only) |
| `roboflow` | ≥ 1.2 | Dataset download & management (training only) |

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/blind-navigation-assistant.git
cd blind-navigation-assistant

# Install all dependencies via Poetry
poetry add ultralytics opencv-python websockets ollama
```

### Running

#### On the Laptop (detection + camera + TTS)

```bash
poetry run python run.py
```

Press **`q`** in the OpenCV window to quit.

#### On the Remote Server (LLM)

```bash
poetry run python anydesk_server/signaling_client.py
```

This connects to the signaling server and waits for scene requests.  
When one arrives, it calls Ollama and sends the description back.

---

## Custom YOLO Model

The custom model was trained on a combined dataset from [Roboflow](https://roboflow.com/) using **Google Colab** (T4 GPU).

### Detected Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | puddle | Water puddles on pavement |
| 1 | Fence | Fences and barriers |
| 2 | Fence Anomaly | Damaged or broken fences |
| 3 | stairs | Staircases |
| 4 | up_steps | Steps going up |
| 5 | down_steps | Steps going down |

### Training Metrics

| Metric | Value |
|--------|-------|
| Precision | **0.824** |
| Recall | **0.707** |
| mAP50 | **0.744** |
| mAP50-95 | **0.430** |
| Inference speed | **4.3 ms/image** (T4 GPU) |

> 📖 Full per-class breakdown and analysis in [`documentation/architecture.md`](documentation/architecture.md#custom-model--training-metrics).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 Nano (Ultralytics) |
| LLM | Ollama + Mistral:latest |
| WebSocket | `websockets` + EHB Signaling Server |
| Text-to-Speech | `spd-say` (speech-dispatcher) |
| Camera | OpenCV |
| Training | Google Colab (free T4 GPU) |
| Dataset Management | Roboflow |
| Package Manager | Poetry |

---

## Example Output

```
Detected: person (ahead of you, very close), car (to your left, in the distance),
puddle (to your right, nearby)

LLM says: There is a person directly ahead of you, very close. Watch out for a
puddle on your right. A car is parked to your left but far away.
```

🔊 *This is then spoken aloud through the laptop speakers.*
