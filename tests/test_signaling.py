#--- Phase 0.1:  WebSocket Prototyping (to be run locally on the PC)  ---#

"""
Test suite for the SignalingClient.

Run on PC (websocket only):
    python test_signaling.py

Run on Anydesk server (websocket + Ollama):
    python test_signaling.py --ollama
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anydesk_server.signaling_client import SignalingClient, ask_ollama, OLLAMA_HOST, OLLAMA_MODEL

async def test_websocket():
    """Test 1 & 2: Connect to signaling server and send a message."""
    print("=" * 50)
    print("TEST 1: Websocket connection")
    print("=" * 50)

    client = SignalingClient(client_name="test_client")

    # 1. Connect
    try:
        await client.connect()
        print("✅ PASS — Connected to signaling server\n")
    except Exception as e:
        print(f"❌ FAIL — Could not connect: {e}\n")
        return False

    # 2. Send a test message
    print("=" * 50)
    print("TEST 2: Send a message")
    print("=" * 50)
    try:
        await client.send({
            "type": "topic",
            "data": {
                "name": "test",
                "value": "Hello from test_signaling.py!"
            }
        })
        print("✅ PASS — Message sent successfully\n")
    except Exception as e:
        print(f"❌ FAIL — Could not send: {e}\n")
        await client.close()
        return False

    # Listen briefly for any echo
    print("⏳ Listening for 5 seconds...\n")
    try:
        await asyncio.wait_for(client.listen(), timeout=5)
    except asyncio.TimeoutError:
        pass

    await client.close()
    return True


def test_ollama():
    """Test 3: Check if Ollama is reachable and responds correctly.
    Only relevant on the Anydesk server where Ollama runs locally."""
    print("=" * 50)
    print(f"TEST 3: Ollama local connection ({OLLAMA_HOST}, model: {OLLAMA_MODEL})")
    print("=" * 50)

    test_scene = "Detected: person (ahead of you, nearby), car (to your left, in the distance)"

    print(f"📤 Sending test scene: {test_scene}")
    print(f"🤖 Calling Ollama...")

    try:
        response = ask_ollama(test_scene)
        print(f"📩 Ollama response: {response}")

        if response and len(response) > 0:
            print("✅ PASS — Ollama responded successfully\n")
            return True
        else:
            print("❌ FAIL — Ollama returned an empty response\n")
            return False
    except Exception as e:
        print(f"❌ FAIL — Ollama error: {e}")
        print(f"   Make sure Ollama is running on this machine at {OLLAMA_HOST}")
        print(f"   Check with: curl {OLLAMA_HOST}/api/tags\n")
        return False


async def main():
    run_ollama = "--ollama" in sys.argv

    print("\n🧪 SIGNALING CLIENT — TEST SUITE")
    if run_ollama:
        print("   Mode: Anydesk server (websocket + Ollama)\n")
    else:
        print("   Mode: PC (websocket only)")
        print("   Tip: run with --ollama on the Anydesk server to also test Ollama\n")

    results = {}

    # Test 1 & 2: Websocket (always)
    results["Websocket"] = await test_websocket()

    # Test 3: Ollama (only on Anydesk server)
    if run_ollama:
        results["Ollama"] = test_ollama()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {name}")
    print()

    if all(results.values()):
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed — check output above.")


if __name__ == "__main__":
    asyncio.run(main())
