
#!/usr/bin/env python3
"""
Minimal Whisper-ONNX runner that uses:
  - OpenAI Whisper (tiny.en) pre-processing: load_audio -> pad_or_trim -> log_mel_spectrogram
  - OpenAI Whisper tokenizer post-processing: decode(tokens)
  - ONNX Runtime for encoder/decoder inference

Recommended decoder: decoder_model.onnx (plain)
Optional: decoder_model_merged.onnx (with past_key_values + use_cache_branch)
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import onnxruntime as ort
import torch
import whisper  # pre/post processing only

# ---------- Audio helpers ----------
def list_devices():
    devices = sd.query_devices()
    lines = []
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            lines.append(f"[{idx}] {d['name']} - inputs: {d['max_input_channels']}, samplerate: {int(d['default_samplerate'])}")
    return "\n".join(lines) if lines else "No input devices found."

def record_wav(outfile: str, duration_sec: float, samplerate: int = 16000, channels: int = 1, device=None):
    print(f"⏺️ Recording {duration_sec:.1f}s @ {samplerate} Hz ({channels} ch)")
    if device is not None:
        print(f"🎛️ Input device: {device}")
    try:
        sd.check_input_settings(device=device, samplerate=samplerate, channels=channels)
    except Exception as e:
        print("⚠️ Device check failed:", e)
        print("Available input devices:\n", list_devices())
        sys.exit(1)

    frames = int(duration_sec * samplerate)
    print("🎤 Speak now…")
    start = time.time()
    recording = sd.rec(frames=frames, samplerate=samplerate, channels=channels, dtype="float32", device=device)
    sd.wait()
    elapsed = time.time() - start
    print(f"✅ Recording complete ({elapsed:.2f}s). Saving to {outfile}")
    sf.write(outfile, recording, samplerate, subtype="PCM_16")
    print(f"💾 Saved: {os.path.abspath(outfile)}")

# ---------- Whisper-ONNX runner ----------
class WhisperONNXRunner:
    """
    Supports two decoder types:
      - Plain decoder: decoder_model.onnx (input_ids + encoder_hidden_states; no caches)
      - Merged decoder: decoder_model_merged.onnx (input_ids + encoder_hidden_states + past_key_values.* + use_cache_branch)
    Uses Whisper's official pre/post for consistency with the original pipeline.
    """

    def __init__(self, model_dir: str, decoder_file: str = "decoder_model.onnx", providers: str = "cpu"):
        # Locate ONNX files
        enc_path = os.path.join(model_dir, "encoder_model.onnx")
        dec_path = os.path.join(model_dir, decoder_file)
        if not (os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            raise FileNotFoundError(f"Missing ONNX files under {model_dir}. Expected encoder_model.onnx and {decoder_file}.")

        # Providers
        if providers.lower() == "cuda":
            ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif providers.lower() == "dml":
            ep = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            ep = ["CPUExecutionProvider"]

        print(f"🧩 Loading encoder + decoder ({providers})…")
        self.encoder_sess = ort.InferenceSession(enc_path, providers=ep)
        self.decoder_sess = ort.InferenceSession(dec_path, providers=ep)

        # Tokenizer (English-only tiny.en)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        self.eot_id    = self.tokenizer.eot
        self.sot_seq   = list(self.tokenizer.sot_sequence_including_notimestamps)

        # Encoder I/O
        self.enc_in  = self.encoder_sess.get_inputs()[0].name   # usually 'input_features'
        self.enc_out = self.encoder_sess.get_outputs()[0].name  # usually 'encoder_hidden_states'

        # Decoder I/O
        self.dec_inputs_list = list(self.decoder_sess.get_inputs())
        self.dec_inputs = {i.name: i for i in self.dec_inputs_list}
        self.dec_in_ids = next((i.name for i in self.dec_inputs_list if "input_ids" in i.name), None)
        self.dec_in_enc = next((i.name for i in self.dec_inputs_list if "encoder_hidden_states" in i.name), None)

        # Cache-related (merged only)
        self.dec_use_branch = next((i.name for i in self.dec_inputs_list if "use_cache_branch" in i.name), None)
        self.dec_cache_pos  = next((i.name for i in self.dec_inputs_list if "cache_position" in i.name), None)
        self.past_in_names  = [i.name for i in self.dec_inputs_list if i.name.startswith("past_key_values")]
        self.is_plain = (self.dec_use_branch is None and len(self.past_in_names) == 0)

        self.dec_outs_list = list(self.decoder_sess.get_outputs())
        self.dec_outs = [o.name for o in self.dec_outs_list]
        self.dec_logits = self.dec_outs[0]
        self.present_out_names = [o.name for o in self.dec_outs_list if o.name.startswith("present.")]

        if self.dec_in_ids is None or self.dec_in_enc is None:
            raise RuntimeError(f"Decoder missing required inputs: found {list(self.dec_inputs.keys())}")

        print("🔧 Decoder type:", "PLAIN" if self.is_plain else "MERGED")

    # --- Whisper pre-processing (OpenAI reference) ---
    def _mel(self, audio_path: str) -> np.ndarray:
        # FFmpeg loader → mono/16k → pad_or_trim → log-mel spectrogram
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel_t = whisper.log_mel_spectrogram(audio)  # torch.Tensor [80, 3000]
        mel_np = mel_t.detach().float().cpu().numpy()
        return np.expand_dims(mel_np, 0)  # [1, 80, 3000]

    # --- dtype helper ---
    def _dtype_from_onnx(self, onnx_type: str):
        t = (onnx_type or "").lower()
        if "float16" in t: return np.float16
        if "float"    in t: return np.float32
        if "int64"    in t: return np.int64
        if "bool"     in t: return np.bool_
        return np.float32

    # --- merged helpers ---
    def _zeros_for_past_input(self, inp):
        dtype = self._dtype_from_onnx(inp.type)
        shape = []
        name = inp.name
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            else:
                s = str(d).lower()
                if "batch" in s: shape.append(1)
                elif "past_decoder_sequence_length" in s or ("decoder" in name and "past_key_values" in name):
                    shape.append(0)   # decoder past length = 0
                elif "encoder_sequence_length" in s or ("encoder" in name and "past_key_values" in name):
                    shape.append(0)   # encoder past length = 0
                else: shape.append(1) # heads/head_dim → default 1
        return np.zeros(shape, dtype=dtype)

    def _init_past_feed(self):
        feed = {}
        for inp in self.dec_inputs_list:
            if inp.name.startswith("past_key_values"):
                feed[inp.name] = self._zeros_for_past_input(inp)
        branch_dtype = self._dtype_from_onnx(self.dec_inputs[self.dec_use_branch].type)
        feed[self.dec_use_branch] = np.asarray([0], dtype=branch_dtype)  # first step: no past
        if self.dec_cache_pos is not None:
            pos_dtype = self._dtype_from_onnx(self.dec_inputs[self.dec_cache_pos].type)
            feed[self.dec_cache_pos] = np.asarray([0], dtype=pos_dtype)
        return feed

    def _update_past_from_present(self, dec_out_list):
        next_past = {}
        for name, arr in zip(self.dec_outs, dec_out_list):
            if name.startswith("present."):
                past_name = name.replace("present.", "past_key_values.")
                next_past[past_name] = arr
        return next_past

    # --- transcription ---
    def transcribe(self, audio_path: str, max_tokens: int = 448, skip_special_tokens: bool = True) -> str:
        # 1) Pre: log-mel via OpenAI Whisper
        mel = self._mel(audio_path)

        # 2) Encoder → hidden states
        enc_hidden = self.encoder_sess.run(None, {self.enc_in: mel})[0]  # [1, encoder_seq_len, hidden]

        # 3) Decoder loop (greedy)
        tokens = self.sot_seq[:]

        if self.is_plain:
            # Plain decoder: simplest & robust
            while len(tokens) < max_tokens:
                input_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
                feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
                dec_out = self.decoder_sess.run(self.dec_outs, feed)
                logits = dec_out[0]
                next_id = int(np.argmax(logits[0, -1]))
                tokens.append(next_id)
                if next_id == self.eot_id:
                    break
        else:
            # Merged decoder: init with empty past + branch=0
            input_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
            feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
            feed.update(self._init_past_feed())
            dec_out = self.decoder_sess.run(self.dec_outs, feed)
            logits = dec_out[0]
            past = self._update_past_from_present(dec_out)

            # subsequent steps: branch=1 + reuse past
            branch_dtype = self._dtype_from_onnx(self.dec_inputs[self.dec_use_branch].type)
            use_past_arr = np.asarray([1], dtype=branch_dtype)

            while len(tokens) < max_tokens:
                next_id = int(np.argmax(logits[0, -1]))
                tokens.append(next_id)
                if next_id == self.eot_id:
                    break

                input_ids = np.asarray([next_id], dtype=np.int64).reshape(1, -1)
                feed = {
                    self.dec_in_ids: input_ids,
                    self.dec_in_enc: enc_hidden,
                    self.dec_use_branch: use_past_arr,
                    **past,
                }
                dec_out = self.decoder_sess.run(self.dec_outs, feed)
                logits = dec_out[0]
                past = self._update_past_from_present(dec_out)

        # 4) Post: Whisper tokenizer -> text
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens).strip()

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Whisper tiny.en ONNX runner (OpenAI pre/post)")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (default: 10)")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate Hz (default: 16000)")
    parser.add_argument("--channels", type=int, default=1, help="Channels (default: 1 mono)")
    parser.add_argument("--outfile", type=str, default="recording.wav", help="Output WAV filename (default: recording.wav)")
    parser.add_argument("--device", type=str, default=None, help="Input device index or name (optional)")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing ONNX export")
    parser.add_argument("--decoder", type=str, default="decoder_model.onnx",
                        help="Decoder filename: decoder_model.onnx (plain) or decoder_model_merged.onnx (merged)")
    parser.add_argument("--providers", type=str, default="cpu", choices=["cpu", "cuda", "dml"],
                        help="ONNX Runtime EP (cpu|cuda|dml)")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")

    args = parser.parse_args()
    if args.list_devices:
        print("Available input devices:")
        print(list_devices())
        return

    if args.outfile.endswith(os.sep):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outfile = os.path.join(args.outfile, f"recording_{ts}.wav")

    record_wav(args.outfile, args.duration, args.samplerate, args.channels, device=args.device)
    runner = WhisperONNXRunner(model_dir=args.model_dir, decoder_file=args.decoder, providers=args.providers)
    text = runner.transcribe(args.outfile)
    print("\n=== TRANSCRIPTION ===")
    print(text)

if __name__ == "__main__":
    main()

