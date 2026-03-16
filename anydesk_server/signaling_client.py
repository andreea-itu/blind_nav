#--- Phase 0.0:  WebSocket Prototyping (to be run on Anydesk Server)  ---#

'''
A unified WebSocket client that combines the send/receive prototypes into a single reusable class, 
plus Ollama integration for the Anydesk server.
'''

'''
send_json
    Connects to wss://signaling.ehb.be over SSL and sends a hardcoded JSON message with a type, from, and data field. 
    This proved the signaling server is reachable and accepts messages.

receive_json
    Connects to the same server and listens in a loop, 
    parsing incoming JSON messages and printing their type, from, and data fields.
'''



import asyncio
import websockets
import json
import ssl
import ollama

SIGNALING_SERVER = "wss://signaling.ehb.be"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "mistral:latest"

# Callback for handling incoming messages — override this for your use case
async def on_message_received(data: dict):
    """Default handler for incoming messages. Override this for custom logic."""
    print("📦 Parsed JSON:")
    print("   Type :", data.get("type"))
    print("   From :", data.get("from"))
    print("   Data :", data.get("data"))
    print("-" * 40)


def ask_ollama(scene_text: str) -> str:
    """Call Ollama to describe the scene for a blind person."""
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a real-time navigation assistant for a blind person. "
                    "You receive a list of detected objects with their positions and distances. "
                    "Reply with ONE short spoken sentence (max 10 words). "
                    "Prioritise safety hazards first (obstacles, stairs, puddles, vehicles), "
                    "then nearby people "
                    "Use simple directional language: left, right, ahead. "
                    "Never use bullet points, lists, or multiple sentences."
                ),
            },
            {
                "role": "user",
                "content": scene_text,
            },
        ]
    )
    return response["message"]["content"]


class SignalingClient:
    """Combined send/receive client for the EHB signaling server."""

    def __init__(self, server_uri=SIGNALING_SERVER, client_name="client1"):
        self.server_uri = server_uri
        self.client_name = client_name
        self.websocket = None
        self.message_handler = on_message_received

    async def connect(self):
        """Establish a persistent websocket connection to the signaling server."""
        ssl_context = ssl.create_default_context()

        print(f"🔌 Connecting to signaling server ({self.server_uri})...")

        self.websocket = await websockets.connect(
            self.server_uri,
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
        )
        print(f"Connected to signaling server ({self.server_uri})")

    async def send(self, data: dict):
        """Send a JSON message through the websocket."""
        if self.websocket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Auto-fill 'from' field if not provided
        if "from" not in data:
            data["from"] = self.client_name

        json_message = json.dumps(data)
        print(f"--> Sending: {json_message}")
        await self.websocket.send(json_message)
        print(" --> Message sent!")

    async def listen(self):
        """Listen for incoming messages in a loop.
        If a scene_request is received, calls Ollama and sends the response back."""
        if self.websocket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        print(" --> Listening for messages...")
        while True:
            try:
                message = await self.websocket.recv()

                # Skip binary messages (e.g JPEG frames broadcast by signaling server)
                if isinstance(message, bytes):
                    continue

                print(f"📩 Raw message received: {message}")

                data = json.loads(message)

                # If this is a scene request, call Ollama and respond
                if data.get("type") == "scene_request":
                    scene_text = data.get("data", {}).get("scene_text", "")
                    sender = data.get("from", "unknown")
                    print(f"🎯 Scene request from {sender}: {scene_text}")

                    print(f"🤖 Asking Ollama ({OLLAMA_MODEL})...")
                    try:
                        description = ask_ollama(scene_text)
                        print(f"✅ Ollama response: {description}")
                    except Exception as e:
                        description = f"Error calling Ollama: {e}"
                        print(f"❌ {description}")

                    # Send response back through signaling server
                    await self.send({
                        "type": "scene_response",
                        "data": {
                            "description": description,
                            "original_scene": scene_text
                        }
                    })
                else:
                    await self.message_handler(data)

            except websockets.exceptions.ConnectionClosed:
                print("⚠ Connection to server lost.")
                break
            except json.JSONDecodeError:
                print("❌ Could not parse JSON.")

    async def close(self):
        """Close the websocket connection."""
        if self.websocket:
            await self.websocket.close()
            print("🔒 Connection closed.")


# ── Standalone usage ──
# Run on Anydesk:  python signaling_client.py
#   → Tests Ollama locally, then listens for scene_requests via websocket
# Run on Anydesk (listen only, skip Ollama test):  python signaling_client.py --skip-test

async def main():
    skip_test = "--skip-test" in sys.argv

    client = SignalingClient(client_name="anydesk_worker")

    # Step 1: Test Ollama locally before going online
    if not skip_test:
        print("=" * 50)
        print("Testing Ollama locally before starting...")
        print("=" * 50)
        test_scene = "Detected: person (ahead of you, nearby), car (to your left, in the distance)"
        try:
            response = ask_ollama(test_scene)
            print(f"✅ Ollama works! Response: {response}\n")
        except Exception as e:
            print(f"❌ Ollama is not reachable: {e}")
            print(f"   Make sure Ollama is running: curl {OLLAMA_HOST}/api/tags")
            print("   Exiting.\n")
            return

    # Step 2: Connect to signaling server
    try:
        await client.connect()
    except Exception as e:
        print(f"❌ Could not connect to signaling server: {e}")
        return

    print("\n🟢 Ready — waiting for scene_request messages...\n")

    # Step 3: Listen for scene_requests, call Ollama, send responses
    await client.listen()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
