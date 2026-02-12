from edge_impulse_linux.runner import ImpulseRunner
import numpy as np
import sounddevice as sd
import queue
import threading
import time

MODEL_PATH = "/home/ubuntu/projects/Primer-Software/lib/model.eim"
AUDIO_DEVICE_INDEX = None
# ---- Model parameters ----
SAMPLE_RATE = 48000           # Hz
WINDOW_SAMPLES = 48000        # matches info["features_shm"]["elements"]
HOP_SAMPLES = 8000            # ~167ms hop at 48k; tune as desired

# Class decision parameters
WAKEWORD_LABEL = "Primer"     # your positive class
THRESHOLD = 0.7               # tune for your model
SMOOTHING = 3                 # require N consecutive detections

def _list_input_devices():
    lines = []
    for idx, d in enumerate(sd.query_devices()):
        if d.get('max_input_channels', 0) > 0:
            lines.append(f"[{idx}] {d['name']}  — inputs: {d['max_input_channels']}, "
                         f"default_sr: {int(d['default_samplerate'])}")
    return "\n".join(lines) if lines else "No input devices found."

def audio_stream(q: queue.Queue, stop_event: threading.Event, device=None):
    """Capture mono audio as float32 at SAMPLE_RATE and enqueue chunks."""
    with sd.InputStream(channels=1, 
                        samplerate=SAMPLE_RATE, 
                        dtype="float32",
                        blocksize=HOP_SAMPLES,
                        device=device, 
                        callback=lambda indata, frames, time_info, status: q.put(indata.copy())
                        ) as stream:
        
        # >>> Print the real device that was opened
        dev_index = stream.device 
        dev_info = sd.query_devices(dev_index)
        print(f"[WakeWord] Capturing from device [{dev_index}] '{dev_info['name']}' "
              f"(inputs={dev_info['max_input_channels']},"
              f"default_sr={int(dev_info['default_samplerate'])})") 
        
        while not stop_event.is_set():
            time.sleep(0.01)

def wake_word():
    # Start model wake word model
    runner = ImpulseRunner(MODEL_PATH)
    info = runner.init(debug=False)

    # --- Print device info once here ---
    '''if AUDIO_DEVICE_INDEX is None:
        print("\n[WakeWord] No specific audio device set. Available input devices:\n")
        #print(_list_input_devices())
        #print("\n[WakeWord] Tip: set AUDIO_DEVICE_INDEX to the mic you want to use.\n")
    else:
        dev = sd.query_devices(AUDIO_DEVICE_INDEX)
        print(f"[WakeWord] Using input device [{AUDIO_DEVICE_INDEX}]: {dev['name']} "
              f"(inputs={dev['max_input_channels']}, default_sr={int(dev['default_samplerate'])})")
    # -----------------------------------'''

    # Sanity-check the feature size
    shm_elems = info["features_shm"]["elements"]
    if shm_elems != WINDOW_SAMPLES:
        raise RuntimeError(f"Model expects {shm_elems} samples; adjust WINDOW_SAMPLES.")

    # Ring buffer for latest WINDOW_SAMPLES audio
    buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    # Audio capture thread
    q = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=audio_stream, args=(q, stop_event, AUDIO_DEVICE_INDEX), daemon=True)
    t.start()
    print("Listening... (Ctrl+C to stop)")

    consecutive_hits = 0
    detected = False

    try:
        while True:
            # Get next chunk; this will be shape (HOP_SAMPLES, 1)
            chunk = q.get().flatten()
            if len(chunk) != HOP_SAMPLES:
                continue

            # Slide window and append
            buf = np.roll(buf, -HOP_SAMPLES)
            buf[-HOP_SAMPLES:] = chunk

            # Classify current window
            result = runner.classify(buf.tolist())
            classes = result.get("result", {}).get("classification", {})
            score = float(classes.get(WAKEWORD_LABEL, 0.0))
            nowords = float(classes.get("noWords", 0.0))

            # Simple smoothing / hysteresis
            if score >= THRESHOLD:
                consecutive_hits += 1
            else:
                consecutive_hits = max(0, consecutive_hits - 1)

            # Print compact status line
            print(f"\r{WAKEWORD_LABEL}: {score:0.3f} | noWords: {nowords:0.3f} | streak: {consecutive_hits}", end="")

            if consecutive_hits >= SMOOTHING:
                print(f"\n✅ Wake word detected ({WAKEWORD_LABEL})!")
                detected = True
                consecutive_hits = 0  # reset or keep latched depending on your UX
                break
        
        return detected
    
    except KeyboardInterrupt:
            print("\nStopping...")
    
    finally:
        stop_event.set()
        t.join(timeout=1)
        runner.stop()