"""Keyboard EEF controller: non-blocking key press/release via pynput.

Uses pynput directly (not KeyboardEndEffectorTeleop) so we get both
key-press and key-release events needed for continuous hold-key motion,
and so we can distinguish ``resume_requested`` (Enter) from normal
movement keys without conflicting with any existing listeners.

Key mapping (matches KeyboardEndEffectorTeleop defaults):
    Left / Right arrow  → EEF X (±1)
    Up / Down arrow     → EEF Y (±1)
    Shift_R             → EEF Z +1 (up)
    Shift (Left Shift)  → EEF Z -1 (down)
    Ctrl_R              → gripper open  (discrete action 2)
    Ctrl_L              → gripper close (discrete action 0)
    Enter               → set ``resume_requested`` event
    ESC                 → set ``shutdown_requested`` event
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pynput import (graceful degradation for headless / Wayland envs)
# ---------------------------------------------------------------------------
_PYNPUT_AVAILABLE = False
keyboard = None  # module reference

try:
    from pynput import keyboard as _kb

    keyboard = _kb
    _PYNPUT_AVAILABLE = True
except Exception as _exc:
    logger.warning(
        "pynput not available; KeyboardEEFController will run in no-op mode: %s", _exc
    )


class KeyboardEEFController:
    """Thin wrapper around a pynput ``Listener`` for EEF keyboard control.

    Thread-safe: ``_pressed`` is accessed under ``_lock`` from the pynput
    background thread; ``read_action()`` acquires the same lock on the
    control thread.

    Usage::

        ctrl = KeyboardEEFController()
        ctrl.start()
        ...
        while not ctrl.resume_requested.is_set():
            action = ctrl.read_action()
            # use action["delta_x"] etc.
        ctrl.stop()

    If pynput is unavailable (headless / Wayland), the controller starts
    in *no-op* mode: ``read_action()`` always returns zero deltas and
    ``resume_requested`` / ``shutdown_requested`` remain unset.  Callers
    should check :attr:`is_available` and warn the user.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: dict = {}  # key_obj → bool (True = pressed)
        self._listener = None

        self.resume_requested: threading.Event = threading.Event()
        self.shutdown_requested: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """``True`` if pynput is importable and a listener can be started."""
        return _PYNPUT_AVAILABLE

    def start(self) -> None:
        """Start the background pynput listener.

        If pynput is unavailable, logs a warning and returns without error
        (no-op mode).
        """
        if not _PYNPUT_AVAILABLE:
            logger.warning(
                "KeyboardEEFController: pynput is unavailable. "
                "EEF keyboard control requires pynput on X11/macOS/Windows. "
                "Running in no-op mode (no EEF movement will be produced)."
            )
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info(
            "KeyboardEEFController started. "
            "Arrows=EEF XY, Shift/Shift_R=Z, Ctrl_L/Ctrl_R=gripper, "
            "Enter=resume, ESC=shutdown."
        )

    def stop(self) -> None:
        """Stop the pynput listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    # ------------------------------------------------------------------
    # Action API
    # ------------------------------------------------------------------

    def read_action(self) -> dict[str, float]:
        """Return the current EEF delta command from held keys.

        Returns
        -------
        dict with keys:
            ``delta_x``, ``delta_y``, ``delta_z`` : float in {-1, 0, +1}
            ``gripper`` : float in {0.0, 1.0, 2.0} (discrete: close/stay/open)
        """
        if not _PYNPUT_AVAILABLE:
            return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}

        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper = 1.0  # stay

        with self._lock:
            pressed = dict(self._pressed)  # shallow copy under lock

        for key, is_pressed in pressed.items():
            if not is_pressed:
                continue
            if key == keyboard.Key.left:
                delta_x = 1.0
            elif key == keyboard.Key.right:
                delta_x = -1.0
            elif key == keyboard.Key.up:
                delta_y = -1.0
            elif key == keyboard.Key.down:
                delta_y = 1.0
            elif key == keyboard.Key.shift_r:
                delta_z = 1.0
            elif key == keyboard.Key.shift:
                delta_z = -1.0
            elif key == keyboard.Key.ctrl_r:
                gripper = 2.0  # open
            elif key == keyboard.Key.ctrl_l:
                gripper = 0.0  # close

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "gripper": gripper,
        }

    # ------------------------------------------------------------------
    # pynput callbacks (called from pynput background thread)
    # ------------------------------------------------------------------

    def _on_press(self, key) -> None:
        # Normalise character keys so they compare equal to the Key enum
        if hasattr(key, "char") and key.char is not None:
            # ordinary printable character — store as-is (not used for motion)
            normalised = key.char
        else:
            normalised = key

        if normalised == keyboard.Key.enter:
            self.resume_requested.set()
            return
        if normalised == keyboard.Key.esc:
            self.shutdown_requested.set()
            return

        with self._lock:
            self._pressed[normalised] = True

    def _on_release(self, key) -> None:
        if hasattr(key, "char") and key.char is not None:
            normalised = key.char
        else:
            normalised = key

        with self._lock:
            self._pressed[normalised] = False
