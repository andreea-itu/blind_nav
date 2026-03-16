#--- Phase 4:  Text-to-Speech ---#

'''
Speaks the LLM's description aloud using Piper TTS.
Voice: en_US-amy-medium (American English, female).
Piper synthesises audio locally via ONNX — no network needed.
Playback uses pw-play (PipeWire) so audio goes through the system mixer.

A WAV file is the simplest audio file format — it stores raw, uncompressed audio data with a small header.

text → Piper (ONNX neural net) → temp WAV file → pw-play (PipeWire) → laptop speakers

'''

import os
import wave                 # Python built-in WAV file writer - wraps raw audio in a .wav container
import tempfile             # Creates temporary files in /tmp/ for pw-play to read
import subprocess           # Launches pw-play as an external process for audio playback
import threading            # Lock for thread-safety + daemon threads for cleanup

from piper.voice import PiperVoice

# Get the directory where this file (speaker.py) lives: blind_nav/
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VOICE_PATH = os.path.join(_BASE_DIR, "..", "models", "piper", "en_US-amy-medium.onnx")

# Load the 61 MB ONNX neural network into memory call)
_voice = PiperVoice.load(_VOICE_PATH)

# Track the current speech process so we can wait for it on shutdown
_current_process = None
_speak_lock = threading.Lock()

def speak(text):
    # """Speak text using spd-say (speech-dispatcher), non-blocking."""
    """Synthesise *text* with Piper and play it via pw-play (non-blocking)."""
    global _current_process
    try:
        # Create a temporary file like /tmp/tmpXXXXXX.wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Save the file path so we can pass it to pw-play
            temp_path = f.name
            # Open the temp file as a WAV writer
            with wave.open(f, "wb") as wav_file:
                 # Piper fills it with audio data
                _voice.synthesize_wav(text, wav_file)

        # Play the WAV in the background subprocess (non-blocking)
        # Acquire the lock so the main thread and LLM thread don't clash
        with _speak_lock:

            # If a previous sentence is still playing, stop it so they don't overlap
            # .poll() returns None if the process is still running, or an exit code if finished
            if _current_process and _current_process.poll() is None:
                _current_process.terminate()
            
            # Launch pw-play in the background — it reads the WAV and sends audio to PipeWire (speakers)
            # DEVNULL suppresses any console output from pw-play
            _current_process = subprocess.Popen(
                ["pw-play", temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def _cleanup(proc, path):
            "Helper that waits for playback to finish, then deletes the temp WAV file"
            proc.wait()         # Block until pw-play finishes playing
            try:        
                os.unlink(path) # Delete the temp WAV file from /tmp/
            except OSError:
                pass            # File already gone — ignore

        # Run _cleanup in a daemon thread so it doesn't block the camera loop
        # daemon=True means it dies automatically when the main program exits
        threading.Thread(target=_cleanup, args=(_current_process, temp_path), daemon=True).start()

    except Exception as e:
        # If anything fails (model error, disk full, pw-play missing), log it but don't crash
        print(f"Warning: TTS failed: {e}")
    

def wait_for_speech():
    """Block until the current speech finishes (call on shutdown)."""
    global _current_process
    if _current_process is not None:
        try:
            # Wait up to 30 seconds for pw-play to finish the last sentence
            _current_process.wait(timeout=30)
        except Exception:
            pass                   # Process already finished or was killed — ignore
        _current_process = None    # Reset so we know nothing is playing
        