# Blind Navigation Assistant — Architecture & Flow

A real-time navigation assistant for blind and visually impaired people.  
Detects objects via YOLO, describes the scene via a remote LLM (Ollama/Mistral), and speaks the result aloud.

---

## System Architecture

The system runs across **two machines** connected via a **WebSocket signaling server**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAPTOP (Local PC)                        │
│                                                                 │
│   ┌──────────┐    ┌───────────────┐    ┌────────────────────┐   │
│   │  Webcam   │───▶│  detector.py  │───▶│  scene_builder.py  │   │
│   │ (OpenCV)  │    │  (YOLOv8)     │    │  (YOLO → text)     │   │
│   └──────────┘    └───────┬───────┘    └────────┬───────────┘   │
│                           │                      │               │
│                           │         scene_text   │               │
│                           │◀─────────────────────┘               │
│                           │                                      │
│                           ▼                                      │
│                 ┌──────────────────┐                              │
│                 │ signaling_client │──── WebSocket (send) ──────────┐
│                 │  (detector_pc)   │◀── WebSocket (recv) ──────────┐│
│                 └────────┬─────────┘                              ││
│                          │                                       ││
│                          ▼                                       ││
│                   ┌─────────────┐                                ││
│                   │  speaker.py │                                ││
│                   │  (spd-say)  │                                ││
│                   └──────┬──────┘                                ││
│                          │                                       ││
│                          ▼                                       ││
│                    🔊 Audio out                                  ││
│                   (laptop speakers)                              ││
└─────────────────────────────────────────────────────────────────┘││
                                                                   ││
                    ┌──────────────────────────┐                   ││
                    │  wss://signaling.ehb.be  │◀──────────────────┘│
                    │   (EHB Signaling Server)  │───────────────────┘
                    └──────────────────────────┘
                                                                    
┌─────────────────────────────────────────────────────────────────┐
│                     ANYDESK SERVER (Remote)                      │
│                                                                 │
│   ┌────────────────────┐    ┌─────────────────────────────┐     │
│   │ signaling_client.py│───▶│  Ollama (Mistral:latest)    │     │
│   │ (anydesk_worker)   │◀───│  Local LLM on server GPU    │     │
│   └────────────────────┘    └─────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Step by Step

| Step | Where | What happens | File |
|------|-------|-------------|------|
| **1** | Laptop | Webcam captures a frame via OpenCV | `detector.py` |
| **2** | Laptop | YOLOv8 runs object detection (every 3rd frame) — two models: COCO (80 classes) + custom (puddle, fence, stairs) | `detector.py` |
| **3** | Laptop | Detections are converted to human-readable text with position (left/center/right) and proximity (close/nearby/distance) | `scene_builder.py` |
| **4** | Laptop | Scene text is sent as a `scene_request` JSON message via WebSocket to the signaling server (every 5 seconds) | `detector.py` → `signaling_client.py` |
| **5** | Signaling Server | `wss://signaling.ehb.be` relays the message to all connected clients | EHB infrastructure |
| **6** | Anydesk Server | `signaling_client.py` (running as `anydesk_worker`) receives the `scene_request` | `signaling_client.py` |
| **7** | Anydesk Server | Ollama (Mistral) generates a short, safety-focused scene description | `signaling_client.py` → Ollama API |
| **8** | Anydesk Server | The LLM response is sent back as a `scene_response` JSON message via WebSocket | `signaling_client.py` |
| **9** | Signaling Server | Relays the response back to the laptop | EHB infrastructure |
| **10** | Laptop | `detector.py` receives the description and stores it in shared state | `detector.py` |
| **11** | Laptop | `speaker.py` speaks the description aloud via `spd-say` (Linux speech-dispatcher) | `speaker.py` |

---

## WebSocket Message Format

### `scene_request` (Laptop → Anydesk Server)

```json
{
  "type": "scene_request",
  "from": "detector_pc",
  "data": {
    "scene_text": "Detected: person (ahead of you, very close), car (to your left, in the distance)"
  }
}
```

### `scene_response` (Anydesk Server → Laptop)

```json
{
  "type": "scene_response",
  "from": "anydesk_worker",
  "data": {
    "description": "There is a person very close ahead of you. A car is parked to your left in the distance. Please proceed with caution.",
    "original_scene": "Detected: person (ahead of you, very close), car (to your left, in the distance)"
  }
}
```

---

## Threading Model

The laptop runs **three concurrent execution contexts** because OpenCV (synchronous) and WebSockets (async) cannot share the same thread:

```
┌──────────────────────────────────────────────────────────────┐
│ Main Thread (synchronous)                                    │
│   • OpenCV camera capture (cap.read)                         │
│   • YOLO inference (coco_model + custom_model)               │
│   • Scene text generation (build_scene_description)          │
│   • Read LLM responses from shared state                     │
│   • Display annotated frame (cv2.imshow)                     │
│   • Trigger TTS (speak)                                      │
├──────────────────────────────────────────────────────────────┤
│ Background Thread — asyncio event loop (ws_loop)             │
│   • WebSocket connect / send / recv                          │
│   • Runs forever as a daemon thread                          │
│   • Accessed from main thread via run_coroutine_threadsafe() │
├──────────────────────────────────────────────────────────────┤
│ LLM Thread (spawned per request, daemon)                     │
│   • Calls call_llm(scene_text)                               │
│   • Submits send + recv to ws_loop                           │
│   • Writes result to last_description (protected by lock)    │
│   • Only one active at a time                                │
└──────────────────────────────────────────────────────────────┘
```

**Why three threads?**
- `cv2.waitKey()` blocks the main thread — it must stay synchronous
- `websockets` library is async (`await`) — needs its own event loop
- LLM calls take 5–30 seconds — a separate thread prevents the camera from freezing

---

## YOLO Detection — Two-Model Approach

The system runs **two YOLO models** simultaneously for comprehensive detection:

| Model | File | Classes | Source |
|-------|------|---------|--------|
| **COCO model** | `models/yolov8n.pt` | 80 classes (person, car, truck, bicycle, traffic light, etc.) | Pre-trained by Ultralytics |
| **Custom model** | `models/best.pt` | 6 classes (puddle, Fence, Fence Anomaly, stairs, up_steps, down_steps) | Fine-tuned on combined Roboflow dataset via Google Colab |

**Why two models instead of one?**
- The COCO model is already highly accurate for common objects — no need to retrain
- The custom model adds domain-specific classes (puddles, fences, stairs) that COCO doesn't cover
- Results from both models are **merged** in `scene_builder.py` into a single description

### Custom Model Training

The custom model was trained by:
1. **Collecting datasets** from Roboflow Universe (puddle detection, fences, stairs)
2. **Merging** them with `tools/merge_datasets.py` — remapping class IDs to avoid conflicts
3. **Training** YOLOv8n on Google Colab (free GPU) for 50 epochs with `imgsz=640`, `batch=16`
4. **Downloading** the `best.pt` weights to `models/`

#### Custom Class Mapping

| ID | Class | Original Dataset | Original ID |
|----|-------|-----------------|-------------|
| 0 | puddle | puddle-detection.v3 | 0 |
| 1 | Fence | Fences.v2 | 0 |
| 2 | Fence Anomaly | Fences.v2 | 1 |
| 3 | stairs | Dataset-Stairs-1 | 3 |
| 4 | up_steps | Dataset-Stairs-1 | 4 |
| 5 | down_steps | Dataset-Stairs-1 | 0 |

---

## Scene Builder — How Detections Become Text

`scene_builder.py` converts raw YOLO bounding boxes into spoken descriptions using two heuristics:

### Position (horizontal thirds)

```
┌───────────┬───────────┬───────────┐
│  LEFT     │  CENTER   │  RIGHT    │
│  (0–33%)  │ (33–66%)  │ (66–100%) │
│           │           │           │
│ "to your  │ "ahead    │ "to your  │
│   left"   │  of you"  │   right"  │
└───────────┴───────────┴───────────┘
```

The center x-coordinate of each bounding box determines which third it falls in.

### Proximity (bounding box area ratio)

| Size ratio (box area / frame area) | Label |
|-------------------------------------|-------|
| > 25% | **very close** |
| > 5% | **nearby** |
| ≤ 5% | **in the distance** |

Larger bounding boxes mean the object is closer to the camera.

### Example Output

```
Detected: person (ahead of you, very close), refrigerator (to your right, nearby), 
chair (to your left, in the distance), chair (to your right, in the distance)
```

---

## Project Structure

```
assignment/
├── run.py                        ← Entry point: poetry run python run.py
├── .gitignore                    ← Ignores weights, datasets, caches
│
├── blind/                    ← Runtime application package
│   ├── __init__.py               ← Makes blind a Python package
│   ├── detector.py               ← Main loop: camera → YOLO → WebSocket → TTS
│   ├── scene_builder.py          ← Converts YOLO detections → human-readable text
│   └── speaker.py                ← Text-to-speech via spd-say (Linux)
│
├── anydesk_server/               ← Runs on the remote Anydesk server
│   ├── signaling_client.py       ← WebSocket client + Ollama integration
│   ├── _send_json.py             ← Deprecated: early WebSocket send prototype
│   └── _receive_json.py          ← Deprecated: early WebSocket receive prototype
│
├── models/                       ← Model weights
│   ├── yolov8n.pt                ← Pre-trained COCO (80 classes)
│   └── best.pt                   ← Custom-trained (puddle, fence, stairs)
│
├── tools/                        ← Offline scripts (not used at runtime)
│   ├── train_model.py            ← YOLOv8 training script
│   ├── merge_datasets.py         ← Combines Roboflow datasets with class remapping
│   └── roboflow.py               ← Roboflow dataset download script
│
├── tests/                        ← Test scripts
│   ├── test_signaling.py         ← Tests WebSocket + Ollama connectivity
│   └── test_speaker.py           ← Tests TTS output
│
├── roboflow/                     ← Training datasets (git-ignored, ~25k images)
│   ├── combined/                 ← Merged dataset used for training
│   ├── puddle-detection.v3-v_2.yolov8/
│   ├── Fences.v2-v_2.yolov8/
│   └── Dataset-Stairs-1/
│
├── runs/                         ← YOLO training output (git-ignored)
│
└── _documentation/               ← This file and other docs
    ├── architecture.md           ← ← You are here
    ├── ai_project_proposal.md
    └── project_notes.md
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Object Detection | **YOLOv8 Nano** (Ultralytics) | Fast, lightweight, real-time capable |
| LLM | **Ollama + Mistral:latest** | Local LLM, no cloud dependency, safety-focused prompts |
| WebSocket | **websockets** library + **EHB Signaling Server** | Bridges laptop ↔ Anydesk server |
| Text-to-Speech | **spd-say** (speech-dispatcher) | Built into Linux, non-blocking |
| Camera | **OpenCV** (`cv2.VideoCapture`) | Standard Python camera interface |
| Training | **Google Colab** (free GPU) | YOLOv8 training too slow on CPU |
| Dataset Management | **Roboflow** | Annotation, export, dataset hosting |
| Package Manager | **Poetry** | Dependency management for the project |

---

## How to Run

### On the Laptop (detection + camera + TTS)

```bash
cd assignment/
poetry run python run.py
```

Press `q` in the OpenCV window to quit.

### On the Anydesk Server (LLM)

```bash
python signaling_client.py
```

This connects to the signaling server and waits for `scene_request` messages.  
When one arrives, it calls Ollama and sends the response back.

### Running Tests

```bash
# Test WebSocket connectivity (from laptop)
poetry run python tests/test_signaling.py

# Test WebSocket + Ollama (from Anydesk server)
python tests/test_signaling.py --ollama
```

---

## Timing & Performance

| Operation | Frequency | Duration |
|-----------|-----------|----------|
| Camera frame capture | Every frame (~30 FPS) | < 1 ms |
| YOLO inference | Every 3rd frame (~10 FPS) | ~20–50 ms (CPU) |
| Scene text generation | Every 3rd frame | < 1 ms |
| WebSocket send | Every 5 seconds | < 100 ms |
| Ollama LLM response | Every 5 seconds | 3–15 seconds |
| TTS playback | After each LLM response | 2–5 seconds |

The 5-second interval between LLM calls balances responsiveness with server load.  
YOLO runs every 3rd frame to keep the camera display smooth while saving CPU.

---

## Custom Model — Training Metrics

The custom YOLOv8n model was trained on **Google Colab** (T4 GPU) using the combined Roboflow dataset.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n.pt` (YOLOv8 Nano, pre-trained on COCO) |
| Training approach | Transfer learning (fine-tune detection head, keep COCO backbone) |
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Training images | 5,618 |
| Validation images | 535 |
| Test images | 267 |
| Inference speed | **4.3 ms/image** (T4 GPU) |

### Overall Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **Precision** | **0.824** | 82.4% of detections are correct (low false positives) |
| **Recall** | **0.707** | 70.7% of real objects are found (misses ~30%) |
| **mAP50** | **0.744** | Overall accuracy at 50% IoU threshold — good for a first model |
| **mAP50-95** | **0.430** | Stricter accuracy averaged across IoU 0.50–0.95 |

### Per-Class Breakdown

| Class | Precision | Recall | mAP50 | mAP50-95 | Val Samples | Assessment |
|-------|-----------|--------|-------|----------|-------------|------------|
| **puddle** | 0.819 | 0.777 | **0.842** | 0.478 | 399 | Good — high precision despite low-contrast targets |
| **Fence** | 0.563 | 0.702 | 0.670 | 0.352 | 84 | Moderate — limited training data |
| **Fence Anomaly** | 1.000 | 0.545 | 0.548 | 0.289 | 11 | Weak — only 11 validation samples, too few to generalize |
| **stairs** | 0.970 | 0.960 | **0.971** | 0.775 | 174 | **Excellent** — visually distinct, plenty of data |
| **up_steps** | 0.857 | 0.754 | **0.842** | 0.456 | 499 | Good — reliable detection |
| **down_steps** | 0.737 | 0.503 | 0.588 | 0.226 | 206 | Weak — hardest to distinguish visually |

### Key Takeaways

1. **Stairs is the strongest class** (mAP50 = 0.971) — visually distinct features and good data volume
2. **Puddle detection works well** (mAP50 = 0.842) — despite puddles having low contrast against pavement
3. **Fence Anomaly is the weakest** (mAP50 = 0.548) — only 11 validation samples; needs more annotated data
4. **down_steps is inherently hard** (mAP50 = 0.588) — difficult to distinguish from regular stairs in images
5. **Precision (0.824) is high** — important for a navigation assistant, as false alarms erode user trust

### What the Metrics Mean for Blind Navigation

| Metric | Why it matters |
|--------|---------------|
| **High precision (82.4%)** | The system rarely reports objects that aren't there — critical for trust. A blind user won't be told to "watch out for stairs" when there are none. |
| **Moderate recall (70.7%)** | The system misses ~30% of objects. This is the main area for improvement — a missed puddle or staircase is a safety risk. |
| **mAP50 = 0.744** | At 50% overlap threshold, the model correctly locates and classifies ~74% of objects. Acceptable for an MVP, but should improve with more training data. |
| **4.3 ms inference** | Fast enough for real-time use on GPU. On a laptop CPU, inference is ~20–50 ms, still well within the 3-frame skip budget. |
