#--- Phase 4:  Text-to-Speech ---#

'''
Speaks the LLM's description aloud using spd-say (Linux speech-dispatcher).
'''


import subprocess

# Track the current speech process so we can wait for it on shutdown
_current_process = None

def speak(text):
    """Speak text using spd-say (speech-dispatcher), non-blocking."""
    global _current_process
    try:
        _current_process = subprocess.Popen(["spd-say", text])
    except Exception as e:
        print(f"Warning: TTS failed: {e}")

def wait_for_speech():
    """Block until the current speech finishes (call on shutdown)."""
    global _current_process
    if _current_process is not None:
        try:
            _current_process.wait(timeout=30)
        except Exception:
            pass
        _current_process = None
