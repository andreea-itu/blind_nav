# Blind Navigation Assistant — Architecture & Flow

A real-time navigation assistant for blind and visually impaired people.  
Detects objects via YOLO, describes the scene via a remote LLM (Ollama/Mistral), and speaks the result aloud using **Piper** neural TTS (local ONNX) and **pw-play** (PipeWire).

---

## System Architecture

The system runs across **two machines** connected via a **WebSocket signaling server**. On the laptop there are **two entry points**: local webcam (`run.py` → `detector.py`) or browser-fed frames (`run_web.py` → `web_receiver.py`).

### Local webcam mode

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
│                 │ SignalingClient  │──── WebSocket (send) ─────────┐
│                 │  (detector_pc)   │◀── WebSocket (recv) ──────────┐│
│                 └────────┬─────────┘                              ││
│                          │                                       ││
│                          ▼                                       ││
│                   ┌─────────────┐                                ││
│                   │  speaker.py │                                ││
│                   │ Piper+pw-play│                                ││
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
│                     REMOTE SERVER (Anydesk / GPU host)           │
│                                                                 │
│   ┌────────────────────┐    ┌─────────────────────────────┐     │
│   │ signaling_client.py│───▶│  Ollama (Mistral:latest)    │     │
│   │ (anydesk_worker)   │◀───│  Local LLM on server GPU    │     │
│   └────────────────────┘    └─────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Browser camera mode

The browser (e.g. `tests/teststayontrails.html`) publishes **JPEG frames** and metadata over the same signaling WebSocket. The laptop runs `run_web.py`, which loads `blind/web_receiver.py`:

- **Connection A (async):** receives `frame_meta` JSON plus **binary JPEG** messages; decodes frames and runs the same YOLO → `scene_builder` pipeline as `detector.py` (including every 3rd frame and the **8 second** LLM throttle).
- **Connection B (background thread + `ws_loop`):** a `SignalingClient` named `web_receiver_pc` sends `scene_request` and reads `scene_response`. Because the server also broadcasts JPEG traffic to this client, `call_llm` **loops on `recv`**, skipping binary payloads and non-`scene_response` JSON until the Ollama reply arrives.

TTS is still **Piper** + **pw-play** via `speaker.py`.

**Configuration note:** `blind/detector.py` joins the signaling URL with `DEFAULT_ROOM` and sends `DEFAULT_TOKEN`. `blind/web_receiver.py` currently creates its LLM `SignalingClient` with **default** `server_uri` and **no** token. If your server expects the same room path and Bearer auth as the webcam path, mirror the `server_uri` / `token` arguments from `detector.py` in `web_receiver.py`.

---

## Data Flow — Step by Step (local webcam)

| Step | Where | What happens | File |
|------|-------|-------------|------|
| **1** | Laptop | Webcam captures a frame via OpenCV | `blind/detector.py` |
| **2** | Laptop | YOLOv8 runs object detection (every 3rd frame) — two models: COCO (80 classes) + custom (puddle, fence, stairs, …) | `blind/detector.py` |
| **3** | Laptop | Detections are converted to human-readable text with position (left / ahead / right) and proximity (very close / nearby / in the distance) | `blind/scene_builder.py` |
| **4** | Laptop | Scene text is sent as a `scene_request` JSON message via WebSocket (at most every **8 seconds**, and only if the scene is not empty) | `blind/detector.py` → `anydesk_server/signaling_client.py` (`SignalingClient`) |
| **5** | Signaling Server | `wss://signaling.ehb.be` relays the message to connected clients | EHB infrastructure |
| **6** | Remote server | `signaling_client.py` (role `anydesk_worker`) receives the `scene_request` | `anydesk_server/signaling_client.py` |
| **7** | Remote server | Ollama (Mistral) returns a **single short spoken line** (system prompt: safety-first, **max ~10 words**, one sentence) | `ask_ollama()` → Ollama API |
| **8** | Remote server | The reply is sent as a `scene_response` JSON message | `signaling_client.py` |
| **9** | Signaling Server | Relays the response to the laptop | EHB infrastructure |
| **10** | Laptop | The detector stores the description in shared state (`last_description` under a lock) | `blind/detector.py` |
| **11** | Laptop | `speaker.py` synthesises audio with **Piper** (ONNX) and plays it with **pw-play** | `blind/speaker.py` |

---

## WebSocket Message Format

`SignalingClient.send()` **adds `from` automatically** if omitted, using the client’s `client_name`.

### `scene_request` (Laptop → remote LLM worker)

```json
{
  "type": "scene_request",
  "from": "detector_pc",
  "data": {
    "scene_text": "Detected: person (ahead of you, very close), car (to your left, in the distance)"
  }
}
```

(In browser mode, `from` is `web_receiver_pc` when sent from `web_receiver.py`.)

### `scene_response` (Remote LLM worker → Laptop)

```json
{
  "type": "scene_response",
  "from": "anydesk_worker",
  "data": {
    "description": "Person very close ahead, car left in distance.",
    "original_scene": "Detected: person (ahead of you, very close), car (to your left, in the distance)"
  }
}
```

The live Ollama system prompt asks for **one short sentence** (roughly **10 words**), so real `description` strings are usually shorter than early prose-style examples.

### Browser frame traffic (browser mode only)

Per frame, the browser typically sends:

1. A JSON message with `type: "frame_meta"` (and ids / session fields).
2. A **binary** message containing raw **JPEG** bytes.

`web_receiver.py` ignores the JSON for inference and runs YOLO on the JPEG.

---

## Threading Model

The laptop uses **three concurrent execution contexts** for the webcam path because OpenCV (synchronous) and WebSockets (async) cannot share one thread:

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

- `cv2.waitKey()` keeps the main loop synchronous.
- The `websockets` library is asyncio-based.
- LLM round-trips take several seconds; offloading them avoids blocking capture and UI.

**Browser mode:** JPEG reception runs inside `asyncio.run(listen_for_frames())` on the main thread; `process_frame` is synchronous. The **same** background `ws_loop` + LLM thread pattern is used for Ollama traffic as in `detector.py`.

---

## YOLO Detection — Two-Model Approach

The system runs **two YOLO models** in parallel where weights are present:

| Model | File | Classes | Source |
|-------|------|---------|--------|
| **COCO model** | `models/yolov8n.pt` | 80 classes (person, car, truck, bicycle, traffic light, etc.) | Pre-trained by Ultralytics |
| **Custom model** | `models/best.pt` | 6 classes (puddle, Fence, Fence Anomaly, stairs, up_steps, down_steps) | Fine-tuned on a combined Roboflow dataset (e.g. Colab) |

If `best.pt` is missing, the code runs **COCO-only** and logs a warning.

**Why two models?**

- COCO covers common obstacles and traffic-related objects.
- The custom head adds puddles, fences, and stair/step classes COCO does not label.
- `scene_builder.py` **merges** detections from both result objects into one string.

### Optional ONNX artifacts

The repository may also contain exported weights (e.g. `models/yolov8n.onnx`) for experiments or browser-side inference tests — **runtime detection** in `detector.py` / `web_receiver.py` uses **Ultralytics `.pt`** loading.

### Custom Model Training (historical workflow)

The custom model was produced by:

1. Collecting Roboflow Universe exports (puddles, fences, stairs).
2. Merging datasets with **consistent class ID remapping** (done in the training project — not shipped as `tools/` in this repo).
3. Training YOLOv8n (e.g. on Google Colab) with `imgsz=640`, `batch=16`, ~50 epochs.
4. Placing `best.pt` under `models/`.

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

`scene_builder.py` converts YOLO boxes into phrases using the same rules as before:

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

### Proximity (bounding box area ratio)

| Size ratio (box area / frame area) | Label |
|-------------------------------------|-------|
| > 25% | **very close** |
| > 5% | **nearby** |
| ≤ 5% | **in the distance** |

Detections with confidence **≤ 0.3** are filtered out.

---

## Project Structure

```
blind_nav/                        ← repository root
├── run.py                        ← Entry point (local webcam): poetry run python run.py
├── run_web.py                    ← Entry point (browser frames): poetry run python run_web.py
├── .gitignore
├── README.md
│
├── blind/
│   ├── detector.py               ← Webcam loop: YOLO → WebSocket LLM → Piper TTS
│   ├── web_receiver.py         ← Browser JPEG loop + LLM client + Piper TTS
│   ├── scene_builder.py          ← YOLO → scene text
│   └── speaker.py                ← Piper ONNX + pw-play
│
├── anydesk_server/
│   └── signaling_client.py       ← SignalingClient + Ollama listener (run on GPU host)
│
├── models/
│   ├── yolov8n.pt                ← COCO weights (required for default path)
│   ├── best.pt                   ← Custom weights (optional)
│   ├── yolov8n.onnx              ← Optional export / experiments
│   └── piper/
│       └── en_US-amy-medium.onnx (+ sidecar JSON)  ← Piper voice
│
├── tests/
│   ├── test_signaling.py         ← WebSocket / Ollama checks
│   ├── test_speaker.py           ← TTS
│   ├── test_piper.py             ← Piper-specific checks
│   ├── test_best_model.py        ← Custom YOLO weights
│   ├── test_onnx_pt.py           ← ONNX vs PyTorch comparison
│   ├── test_onnx_browser.html    ← Browser ONNX experiments
│   └── teststayontrails.html     ← Browser camera demo (used with run_web.py)
│
├── documentation/
│   └── architecture.md           ← This file
│
├── roboflow/                     ← Training data (often git-ignored)
├── runs/                         ← YOLO training outputs (often git-ignored)
```

---

## Signaling URLs and Auth (implementation)

In `anydesk_server/signaling_client.py`:

- Base server: `SIGNALING_SERVER = "wss://signaling.ehb.be"`.
- Default room path: `DEFAULT_ROOM = "/ws/pathnavigation"` (appended to the base URL for the worker and for `detector.py`).
- Optional **Bearer** token: `DEFAULT_TOKEN` — passed as the `Authorization` header when non-empty.

`blind/detector.py` constructs `server_uri` as `SIGNALING_SERVER + DEFAULT_ROOM` and passes `DEFAULT_TOKEN` so it joins the same room as the remote worker.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Object Detection | **YOLOv8 Nano** (Ultralytics) | Fast, real-time on CPU with frame skipping |
| LLM | **Ollama + Mistral:latest** | Local inference on the remote machine |
| WebSocket | **websockets** + **EHB Signaling Server** | Relays JSON (and binary frames in browser mode) |
| Text-to-Speech | **Piper** (ONNX) + **pw-play** (PipeWire) | Local synthesis, non-blocking playback |
| Camera | **OpenCV** (`cv2.VideoCapture`) | Local webcam |
| Training | **Google Colab** (typical) | GPU for YOLO training |
| Dataset Management | **Roboflow** | Annotation and export |
| Package Manager | **Poetry** | Dependencies |

---

## How to Run

### Laptop — local webcam

```bash
cd blind_nav
poetry run python run.py
```

Press `q` in the OpenCV window to quit.

### Laptop — browser camera

Serve the `tests/` directory over HTTP(S), open `teststayontrails.html`, start the camera, then:

```bash
cd blind_nav
poetry run python run_web.py
```

### Remote server — Ollama + signaling listener

From the repo (or a copy of `anydesk_server/`):

```bash
cd anydesk_server
python signaling_client.py --token YOUR_BEARER_TOKEN
```

Flags:

- `--room` — override room path (default matches `DEFAULT_ROOM` in code).
- `--skip-test` — skip the local Ollama smoke test before connecting.

When a `scene_request` arrives, the script calls Ollama and sends `scene_response` back.

### Tests

```bash
cd blind_nav
poetry run python tests/test_signaling.py
poetry run python tests/test_speaker.py
poetry run python tests/test_piper.py
```

---

## Timing & Performance

| Operation | Frequency | Notes |
|-----------|-----------|--------|
| Camera / JPEG frame | Every frame | Webcam: `cap.read`; browser: each received JPEG |
| YOLO inference | Every 3rd frame | Reduces CPU load |
| Scene text | Every 3rd frame | Trivial cost vs YOLO |
| LLM WebSocket request | At most every **8 seconds** | `description_interval` / `DESCRIPTION_INTERVAL` — leaves time for Ollama and Piper |
| Ollama response | On each request | Often ~3–15 s (hardware dependent) |
| TTS | After each new description | Piper + pw-play; overlapping playback is cut off when a new line starts |

The **8 second** throttle balances responsiveness with server load and avoids queuing speech back-to-back.

---

## Custom Model — Training Metrics

The tables below are a **snapshot** from the training run that produced the documented `best.pt` (Colab / T4). Re-train or swap weights and metrics will change.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n.pt` (YOLOv8 Nano, COCO pre-trained) |
| Training approach | Transfer learning |
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
| **Precision** | **0.824** | Low false positives |
| **Recall** | **0.707** | Misses ~30% of objects |
| **mAP50** | **0.744** | Reasonable at IoU 0.5 |
| **mAP50-95** | **0.430** | Stricter localization |

### Per-Class Breakdown

| Class | Precision | Recall | mAP50 | mAP50-95 | Val Samples | Assessment |
|-------|-----------|--------|-------|----------|-------------|------------|
| **puddle** | 0.819 | 0.777 | **0.842** | 0.478 | 399 | Good |
| **Fence** | 0.563 | 0.702 | 0.670 | 0.352 | 84 | Moderate |
| **Fence Anomaly** | 1.000 | 0.545 | 0.548 | 0.289 | 11 | Weak — few val samples |
| **stairs** | 0.970 | 0.960 | **0.971** | 0.775 | 174 | Strong |
| **up_steps** | 0.857 | 0.754 | **0.842** | 0.456 | 499 | Good |
| **down_steps** | 0.737 | 0.503 | 0.588 | 0.226 | 206 | Harder class |

### What the Metrics Mean for Blind Navigation

| Metric | Why it matters |
|--------|---------------|
| **High precision** | Fewer false spoken alerts — better trust. |
| **Moderate recall** | Missed hazards remain the main safety gap; more data / training helps. |
| **mAP50** | Locates and classifies many objects at IoU 0.5; not a guarantee for every deployment. |
| **4.3 ms (GPU)** | Laptop CPU is typically ~20–50 ms per inference, still compatible with “every 3rd frame”. |
