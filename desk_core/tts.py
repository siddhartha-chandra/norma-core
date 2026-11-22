"""Non-blocking portable-speaker TTS cue queue (DESK-CORE issue #14).

Uses macOS `say` when available (reliable across threads/async loops) or `pyttsx3`
in a dedicated background worker thread. Disable with `DESK_TTS=0`.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading


class SpeakerTTS:
    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue(maxsize=16)
        self._last_utterance: str = ""
        self._enabled = os.environ.get("DESK_TTS", "1") != "0"
        self._backend = "disabled"
        self._worker: threading.Thread | None = None
        if self._enabled:
            if sys.platform == "darwin" and shutil.which("say"):
                self._backend = "macos-say"
            else:
                self._backend = "pyttsx3"
            self._worker = threading.Thread(target=self._loop, daemon=True, name="desk-tts")
            self._worker.start()

    def _loop(self) -> None:
        engine = None
        if self._backend == "pyttsx3":
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 185)
            except Exception:
                self._backend = "noop"

        while True:
            text = self._q.get()
            try:
                if self._backend == "macos-say":
                    subprocess.run(["say", "-r", "185", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif self._backend == "pyttsx3" and engine is not None:
                    engine.say(text)
                    engine.runAndWait()
            except Exception:
                pass
            finally:
                self._q.task_done()

    def speak(self, text: str) -> bool:
        clean = " ".join(str(text).split()).strip()
        if not clean:
            return False
        self._last_utterance = clean
        if not self._enabled or self._backend == "noop":
            return False
        try:
            self._q.put_nowait(clean)
            return True
        except queue.Full:
            return False

    def status(self) -> dict[str, object]:
        return {
            "enabled": self._enabled,
            "backend": self._backend,
            "last_utterance": self._last_utterance,
        }
