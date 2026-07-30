"""Keyboard EEF controller: non-blocking key press/release via pynput.

Uses pynput directly (not KeyboardEndEffectorTeleop) so we get both
key-press and key-release events needed for continuous hold-key motion,
and so we can distinguish ``resume_requested`` (Enter) from normal
movement keys without conflicting with any existing listeners.

Key mapping (top-down robot base frame):
    Up / Down arrow     → EEF X (forward/backward)
    Left / Right arrow  → EEF Y (left/right)
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

from lerobot.utils.keyboard_input import pynput_can_capture, pynput_listener_is_trusted

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
    logger.warning("pynput not available; KeyboardEEFController will run in no-op mode: %s", _exc)


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

    If pynput cannot capture key-release events (headless, Wayland, or missing
    desktop permissions), :attr:`is_available` remains ``False``. Callers must
    fail closed instead of entering a manual-control loop that cannot be exited.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: dict = {}  # key_obj → bool (True = pressed)
        self._listener = None
        self._capture_available = False

        self.resume_requested: threading.Event = threading.Event()
        self.shutdown_requested: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """``True`` while a trusted pynput listener is alive."""
        return bool(self._capture_available and self._listener is not None and self._listener.is_alive())

    def start(self) -> None:
        """Start the background pynput listener.

        If pynput is unavailable, logs a warning and returns without error
        (no-op mode).
        """
        if self.is_available:
            return
        if not _PYNPUT_AVAILABLE or not pynput_can_capture():
            logger.warning(
                "KeyboardEEFController cannot capture key-release events. "
                "EEF keyboard control requires pynput on X11/macOS/Windows "
                "with the required desktop permissions."
            )
            return

        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()
            if not pynput_listener_is_trusted(self._listener):
                logger.error(
                    "pynput listener is not trusted. Grant Accessibility / Input Monitoring "
                    "permission, then restart the rollout."
                )
                self._listener.stop()
                self._listener = None
                return
        except Exception as exc:
            logger.error("Could not start pynput keyboard listener: %s", exc)
            if self._listener is not None:
                self._listener.stop()
                self._listener = None
            return

        self._capture_available = True
        logger.info(
            "KeyboardEEFController started. "
            "Arrows=EEF XY, Shift/Shift_R=Z, Ctrl_L/Ctrl_R=gripper, "
            "Enter=resume, ESC=shutdown."
        )

    def stop(self) -> None:
        """Stop the pynput listener."""
        self._capture_available = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._pressed.clear()

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
        if not self.is_available:
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
            if key == keyboard.Key.up:
                delta_x = 1.0
            elif key == keyboard.Key.down:
                delta_x = -1.0
            elif key == keyboard.Key.left:
                delta_y = 1.0
            elif key == keyboard.Key.right:
                delta_y = -1.0
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
        normalised = key.char if hasattr(key, "char") and key.char is not None else key

        if normalised == keyboard.Key.enter:
            self.resume_requested.set()
            return
        if normalised == keyboard.Key.esc:
            self.shutdown_requested.set()
            return

        with self._lock:
            self._pressed[normalised] = True

    def _on_release(self, key) -> None:
        normalised = key.char if hasattr(key, "char") and key.char is not None else key

        with self._lock:
            self._pressed[normalised] = False
