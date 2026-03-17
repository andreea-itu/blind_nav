from piper.voice import PiperVoice
import wave, tempfile, subprocess, os

# Build path relative to this file: tests/ → ../models/piper/
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VOICE_PATH = os.path.join(_BASE_DIR, "..", "models", "piper", "en_US-amy-medium.onnx")

voice = PiperVoice.load(_VOICE_PATH)

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    tmp = f.name
    with wave.open(f, 'wb') as wav:
        voice.synthesize_wav('Stairs ahead on your left. Be careful.', wav)

print(f'WAV: {tmp} ({os.path.getsize(tmp)} bytes)')
print('Playing with pw-play...')
subprocess.run(['pw-play', tmp])
os.unlink(tmp)
print('Done!')
