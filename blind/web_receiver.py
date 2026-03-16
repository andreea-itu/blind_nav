#--- Mode 2: Browser Camera via WebSocket ---#

'''
Receives JPEG frames from the browser (teststayontrails.html) through
wss://signaling.ehb.be, runs YOLO detection on each frame, sends the
scene description to the remote LLM (Ollama on Anydesk), and speaks
the response with Piper TTS.

Flow:
  Browser camera → WebSocket (signaling.ehb.be) → this script receives JPEG
  → OpenCV decode → YOLO (COCO + custom) → scene_builder → LLM → Piper

Usage:
  poetry run python run_web.py
'''

import os
import sys
import json
import ssl
import time
import asyncio
import threading
import numpy as np
import cv2

import websockets
from ultralytics import YOLO

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blind.scene_builder import build_scene_description
from blind.speaker import speak, wait_for_speech
from anydesk_server import signaling_client


# ── Constants ──#

SIGNALING_SERVER = "wss://signaling.ehb.be"

# How often (seconds) to send a scene to the LLM.
# Must be long enough for Ollama to respond and Piper to finish speaking.
DESCRIPTION_INTERVAL = 8

# ── Model Setup ──#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Model 1 — Pre-trained YOLOv8 Nano (80 COCO classes)
coco_model = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))

# Model 2 — Custom-trained model (puddle, Fence, stairs …)
custom_weights = os.path.join(MODELS_DIR, "best.pt")
if os.path.exists(custom_weights):
    custom_model = YOLO(custom_weights)
    print(f"Custom model loaded: {custom_weights}")
else:
    custom_model = None
    print(f"Custom model not found at {custom_weights} — running COCO-only mode.")


# ── Shared State for LLM Responses ──#

last_description = None
description_lock = threading.Lock()
llm_thread = None
last_description_time = time.time()


# ── LLM WebSocket (separate connection for sending scene_requests) ──#

ws_loop = asyncio.new_event_loop()

def _start_ws_loop(loop):
    """Run the asyncio event loop forever in a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

ws_thread = threading.Thread(target=_start_ws_loop, args=(ws_loop,), daemon=True)
ws_thread.start()

# Connect to signaling server for LLM requests (same as detector.py)
llm_client = signaling_client.SignalingClient(client_name="web_receiver_pc")

try:
    future = asyncio.run_coroutine_threadsafe(llm_client.connect(), ws_loop)
    future.result(timeout=10)
    print("LLM WebSocket connected — ready to send scene requests.")
except Exception as e:
    print(f"Could not connect LLM WebSocket: {e}")
    print("Continuing without remote LLM — descriptions will not be available.")
    llm_client = None


def call_llm(scene_text):
    """Send a scene_request to the Anydesk LLM server and store the response."""
    global last_description

    if llm_client is None:
        return
    try:
        # Send the scene_request
        send_future = asyncio.run_coroutine_threadsafe(
            llm_client.send({
                "type": "scene_request",
                "data": {"scene_text": scene_text}
            }),
            ws_loop,
        )
        send_future.result(timeout=5)

        # Wait for the scene_response from Ollama
        # The signaling server broadcasts ALL messages (including binary JPEG
        # frames and frame_meta JSON) to every connected client.  We need to
        # keep draining the socket until we get the actual scene_response.
        while True:
            recv_future = asyncio.run_coroutine_threadsafe(
                llm_client.websocket.recv(),
                ws_loop,
            )
            raw_response = recv_future.result(timeout=15)

            # Skip binary messages (JPEG frames broadcast by signaling server)
            if isinstance(raw_response, bytes):
                continue

            data = json.loads(raw_response)
            if data.get("type") == "scene_response":
                description = data["data"]["description"]
                break
            # Other JSON types (frame_meta, etc.) — skip and keep waiting

        with description_lock:
            last_description = description

    except Exception as e:
        print(f"Remote LLM error: {e}")


# ── Frame Processing ──#

frame_count = 0

def process_frame(jpeg_bytes):
    """Decode a JPEG frame from the browser, run YOLO, and trigger the LLM."""
    global frame_count, llm_thread, last_description_time, last_description

    # Decode the JPEG bytes into an OpenCV BGR image
    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("Warning: could not decode JPEG frame.")
        return

    frame_count += 1

    # Run YOLO every 3rd frame to save CPU
    if frame_count % 3 == 0:
        # Model 1: COCO
        coco_results = coco_model(frame)

        # Model 2: Custom (if available)
        custom_results = custom_model(frame) if custom_model else None

        frame_width = frame.shape[1]
        scene_text = build_scene_description(coco_results, frame_width, custom_results)
        print(scene_text)

        current_time = time.time()
        if current_time - last_description_time > DESCRIPTION_INTERVAL:
            if scene_text != "No objects detected nearby.":
                if llm_thread is None or not llm_thread.is_alive():
                    llm_thread = threading.Thread(
                        target=call_llm, args=(scene_text,), daemon=True
                    )
                    llm_thread.start()
                    last_description_time = current_time

        # Draw bounding boxes on the frame (for the optional preview window)
        annotated_frame = coco_results[0].plot()
        if custom_results:
            annotated_frame = custom_results[0].plot(img=annotated_frame)
    else:
        annotated_frame = frame

    # Check if the LLM has responded
    with description_lock:
        description_to_speak = last_description
        last_description = None

    if description_to_speak:
        print(f"LLM says: {description_to_speak}\n")
        try:
            speak(description_to_speak)
        except Exception as e:
            print(f"Warning: TTS error: {e}")

    # Optional: show the annotated frame in an OpenCV window
    # Comment this out on headless / remote machines
    cv2.imshow("Browser Camera — YOLO Detection", annotated_frame)
    cv2.waitKey(1)


# ── WebSocket Listener (receives frames from the browser) ──#

async def listen_for_frames():
    """
    Connect to the signaling server and listen for JPEG frames
    sent by teststayontrails.html.

    The browser sends two messages per frame:
      1. JSON  — { type: "frame_meta", frame_id, sessionId, ... }
      2. Binary — raw JPEG bytes

    We only care about the binary JPEG data for YOLO processing.
    """
    ssl_context = ssl.create_default_context()

    print(f"Connecting to signaling server ({SIGNALING_SERVER}) to receive browser frames...")

    async with websockets.connect(
        SIGNALING_SERVER,
        ssl=ssl_context,
        origin="https://signaling.ehb.be",
        compression=None,
        additional_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        },
    ) as ws:
        print("Connected — waiting for browser frames...\n")

        while True:
            try:
                message = await ws.recv()

                # JSON messages are metadata — we skip them
                if isinstance(message, str):
                    try:
                        meta = json.loads(message)
                        # You could log session/frame info here if needed
                        if meta.get("type") == "frame_meta":
                            pass  # metadata received, JPEG follows next
                    except json.JSONDecodeError:
                        pass
                    continue

                # Binary messages are JPEG frames from the browser camera
                if isinstance(message, bytes):
                    process_frame(message)

            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed.")
                break
            except Exception as e:
                print(f"Error receiving frame: {e}")


# ── Entry Point ──#

def main():
    print("=" * 60)
    print("  Blind Navigation — Browser Camera Mode")
    print("  Open teststayontrails.html in your browser and press Start")
    print("=" * 60)
    print()

    try:
        asyncio.run(listen_for_frames())
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── Graceful Shutdown ──#
    if llm_thread is not None and llm_thread.is_alive():
        print("Waiting for LLM thread to finish...")
        llm_thread.join(timeout=10)

    with description_lock:
        final = last_description

    if final:
        print(f"LLM says: {final}")
        speak(final)
        wait_for_speech()

    if llm_client is not None:
        try:
            future = asyncio.run_coroutine_threadsafe(llm_client.close(), ws_loop)
            future.result(timeout=5)
        except Exception:
            pass

    ws_loop.call_soon_threadsafe(ws_loop.stop)
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
