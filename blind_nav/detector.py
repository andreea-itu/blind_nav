#--- Phase 1: YOLO Webcam Detection ---#
'''
Captures live webcam frames and runs YOLOv8 Nano for real-time object detection.
'''

import os
import sys
# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import cv2                      # OpenCV for webcam and displaying frames
import time                     # time — used to enforce the 5-second interval between LLM calls
import threading                # threading — runs the WebSocket send in a background thread so the camera doesn't freeze
import asyncio                  # asyncio — needed to call async WebSocket methods from a synchronous thread

from ultralytics import YOLO                            # YOLO — loads and runs the object detection model
from blind_nav.scene_builder import build_scene_description     # Converts YOLO results into human-readable scene text
from blind_nav.speaker import speak, wait_for_speech            # TTS — speaks the LLM description aloud

# Import the SignalingClient class from the anydesk_server package.
# This is the WebSocket client that connects to wss://signaling.ehb.be
# and communicates with the Anydesk server where Ollama runs.
from anydesk_server import signaling_client


# ── Model & Camera Setup ──#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
# Model 1 — Pre-trained YOLOv8 Nano (80 COCO classes: person, car, truck, bicycle …)
# We keep this for general obstacle detection.
coco_model = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))

# Model 2 — Custom-trained model (puddle, Fence, Fence Anomaly, stairs, up_steps, down_steps)
# Trained on our combined Roboflow dataset via Google Colab.
# If best.pt doesn't exist yet, we skip it gracefully and run COCO-only.
custom_weights = os.path.join(MODELS_DIR, "best.pt")
if os.path.exists(custom_weights):
    custom_model = YOLO(custom_weights)
    print(f"Custom model loaded: {custom_weights}")
else:
    custom_model = None
    print(f"Custom model not found at {custom_weights} - running COCO-only mode.")

# Open the default camera /dev/video0
cap = cv2.VideoCapture(0)

# ── Shared State for LLM Responses ──#

last_description = None                 # Holds the most recent LLM response
description_lock = threading.Lock()     # Protects last_description from race conditions
llm_thread = None                       # Track the current LLM background thread while waiting for a Websocket response.

# Set the initial time stamp to now so the first  LLM call waits 5 seconds, 
# giving the camera window time to open before we start sending requests.
last_description_time = time.time()

# How many seconds to wait between LLM calls.
# Calling every frame would flood the server; 5 seconds is a good balance
# between responsiveness and not overloading the Anydesk server.
description_interval = 5

# ── WebSocket Connection (Background Event Loop) ──#

# Create a dedicated asyncio event loop that runs in its own thread.
# The main loop is synchronous (OpenCV waitKey block), but 
# SignalingClient uses async/await (websockets library) - 
# We cannot mix the in the same thread, so we run separate background thread.
ws_loop = asyncio.new_event_loop()

def start_ws_loop(loop):
    """Run the asyncio event loop forever in a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start the background event loop as a daemon
# daemon=True means that it will automatically stop when the main program exists.
ws_thread = threading.Thread(target=start_ws_loop, args=(ws_loop,), daemon=True)
ws_thread.start()

# create a Signalint client instance so Anydesk server knows who sent the request
client = signaling_client.SignalingClient(client_name="detector_pc")


# Connect to Signaling server (wss://signaling.ehb.be) from the main thread
# Main thread - synchronous	OpenCV camera loop, YOLO, speaker
# Background thread	 - asyncio	WebSocket (connect, send, recv)

try:
    # it submits the async coroutine to ws_loop (the asyncio event loop running in the background thread we started earlier) and returns a concurrent.futures.Future objec
    future = asyncio.run_coroutine_threadsafe(client.connect(), ws_loop)
    future.result(timeout=10) # blocks the main thread and waits up to 10 seconds for client.connect() to finish.
    print("Websocket connected - Ready to send scene request")
except Exception as e:
    print(f"Could not connect to Signaling server: {e}")
    print("Continue without remote LLM - descriptions are not available.")
    client = None   # set to None to skip sending it later


def call_llm(scene_text):
    """Helper function that calls the LLM in the background."""
    global last_description

    if client is None:
        # No websocket client connection - skip silently
        # so the camera and YOLO detection still work even without the remote LLM.
        return
    try:
        # Send the scene_request message
        # This matches the data Anydesk server expects:
            # type: scene_request
            # data.scene_text: the scene description from scene_builder.py
        send_future = asyncio.run_coroutine_threadsafe(
            client.send({
                "type": "scene_request",
                "data": {
                    "scene_text": scene_text
                }
            }),
            ws_loop
        )
        # Wait up to 5 seconds for the send to complete
        send_future.result(timeout=5)

        # Now wait for the response from the Anydesk server.
        # client.websocket.recv() returns the next message on the WebSocket.
        # The Anydesk server will reply with:
        #   type: "scene_response"
        #   data.description: the LLM-generated text
        recv_future = asyncio.run_coroutine_threadsafe(
            client.websocket.recv(),
            ws_loop
        )
        # Wait up to 30 seconds for Ollama to generate a response.
        # Mistral can take a few seconds depending on server load.
        raw_response = recv_future.result(timeout=30)

        import json
        data = json.loads(raw_response)

        # Extract the description from the scene_response message
        if data.get("type") == "scene_response":
            description = data["data"]["description"]
        else:
            print(f"Unexpected message type: {data.get('type')}")
            return

        # Store the result - the main loop will pick it up via description_lock
        with description_lock:
            last_description = description
    
    except Exception as e:
        print(f"Remote LLM error {e}")

#── Main Camera Loop ──#

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    frame_count += 1

    # Only run YOLO every 3rd frame to save CPU.
    # YOLO is fast but not free — skipping frames keeps the display smooth.
    if frame_count % 3 == 0:
        # Run Model 1: COCO
        coco_results = coco_model(frame)
        
        # Run Model 2: Custom if available
        custom_results = custom_model(frame) if custom_model else None

        frame_width = frame.shape[1]
        scene_text = build_scene_description(coco_results, frame_width, custom_results)
        print(scene_text)

        current_time = time.time()
        # Only send a new request if enough time has passed (5 seconds)
        if current_time - last_description_time > description_interval:
            
            # Don't send empty scenes - no point asking ythe LLM about nothing
            if scene_text != "No objects detected nearby.":
            # Only spawn a new thread if the previous one has finished
            # so we never have two requests in flight at the same time
                if llm_thread is None or not llm_thread.is_alive():
                    llm_thread = threading.Thread(target=call_llm, args=(scene_text,),daemon=True)
                    llm_thread.start()
                    last_description_time = current_time

        # Draw bounding boxes — overlay COCO first, then custom on top
        annotated_frame = coco_results[0].plot()
        if custom_results:
            annotated_frame = custom_results[0].plot(img=annotated_frame)
    else:
        # On non-Yolo frame, just show the raw camera frame
        annotated_frame = frame

    # Safely read and clear last_description
    # The lock ensures the background thread isn't writing at the same time
    with description_lock:
        description_to_speak = last_description
        last_description = None
    
    # If a new description arrived, print it and speak it aloud
    if description_to_speak:
        print(f"LLM says: {description_to_speak}\n")
        try:
            speak(description_to_speak)
        except Exception as e:
            print(f"Warning: TTS error: {e}")
        
    # Display the annotated frame in an OpenCV window: Press q to quit.
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


#── Graceful Shutdown ──#
# Wait for any running LLM thread to finish so we don't lose a response
if llm_thread is not None and llm_thread.is_alive():
    print("Waiting for LLM thread to finish...")
    llm_thread.join(timeout=60)

# If the LLM finished while we were shutting down, speak the final result
with description_lock:
    final_description = last_description
    last_description = None

if final_description:
    print(f"\n🔊 LLM says: {final_description}\n")
    speak(final_description)
    wait_for_speech()

# Close the WebSocket connection cleanly
if client is not None:
    try:
        future = asyncio.run_coroutine_threadsafe(client.close(), ws_loop)
        future.result(timeout=5)
    except Exception:
        pass


# Stop the background asyncio event loop
ws_loop.call_soon_threadsafe(ws_loop.stop)

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()