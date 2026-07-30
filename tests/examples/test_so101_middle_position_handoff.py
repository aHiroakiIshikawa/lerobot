"""Unit tests for examples/so101_middle_position_handoff/.

Tests cover:
- DwellDetector: timer logic, hysteresis, edge cases
- KeyboardEEFController: key state logic (no hardware required)
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from examples.so101_middle_position_handoff.dwell_detector import (
    DwellDetector,
    DwellDetectorConfig,
)
from examples.so101_middle_position_handoff.keyboard_eef import KeyboardEEFController

# ---------------------------------------------------------------------------
# DwellDetector tests
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic monotonic clock for unit tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class TestDwellDetectorConfig:
    def test_empty_middle_positions_raises(self):
        with pytest.raises(ValueError, match="middle_positions must not be empty"):
            DwellDetectorConfig(middle_positions={})

    def test_exit_less_than_entry_raises(self):
        with pytest.raises(ValueError, match="exit_tolerance_deg"):
            DwellDetectorConfig(
                middle_positions={"j.pos": 0.0},
                entry_tolerance_deg=15.0,
                exit_tolerance_deg=10.0,
            )

    def test_non_positive_dwell_raises(self):
        with pytest.raises(ValueError, match="dwell_time_s"):
            DwellDetectorConfig(middle_positions={"j.pos": 0.0}, dwell_time_s=0.0)

    def test_valid_config(self):
        cfg = DwellDetectorConfig(
            middle_positions={"shoulder_pan.pos": 0.0},
            entry_tolerance_deg=10.0,
            exit_tolerance_deg=15.0,
            dwell_time_s=1.0,
        )
        assert cfg.dwell_time_s == 1.0


class TestDwellDetector:
    def _make(self, middle=None, entry=10.0, exit_=15.0, dwell=1.0):
        if middle is None:
            middle = {"j.pos": 0.0}
        clock = FakeClock()
        cfg = DwellDetectorConfig(
            middle_positions=middle,
            entry_tolerance_deg=entry,
            exit_tolerance_deg=exit_,
            dwell_time_s=dwell,
        )
        det = DwellDetector(cfg, clock=clock)
        return det, clock

    def test_no_fire_before_dwell_time(self):
        det, clock = self._make(dwell=1.0)
        # Stay in zone for 0.9s — should not fire
        obs = {"j.pos": 5.0}  # within ±10 of 0
        for _ in range(9):
            clock.advance(0.1)
            assert det.update(obs) is False

    def test_fires_at_dwell_time(self):
        det, clock = self._make(dwell=1.0)
        obs = {"j.pos": 5.0}
        # Drive the detector to just before the threshold
        det.update(obs)  # records entered_at = 0.0
        clock.advance(0.999)
        assert det.update(obs) is False  # 0.999 < 1.0
        clock.advance(0.001)  # now 1.0
        assert det.update(obs) is True  # fires

    def test_fires_only_once(self):
        det, clock = self._make(dwell=1.0)
        obs = {"j.pos": 5.0}
        det.update(obs)
        clock.advance(2.0)
        assert det.update(obs) is True
        # Subsequent calls while still in zone — should NOT fire again
        assert det.update(obs) is False
        assert det.update(obs) is False

    def test_exit_zone_resets_timer(self):
        det, clock = self._make(dwell=1.0)
        obs_in = {"j.pos": 5.0}
        obs_out = {"j.pos": 50.0}  # clearly outside ±10 and ±15

        det.update(obs_in)
        clock.advance(0.5)
        det.update(obs_out)  # exit zone → reset timer
        clock.advance(0.8)
        assert det.update(obs_in) is False  # timer restarted, only 0.8s elapsed

    def test_rearming_requires_all_joints_to_exit(self):
        det, clock = self._make(dwell=1.0)
        obs_in = {"j.pos": 5.0}
        obs_out = {"j.pos": 50.0}  # outside exit_tol=15

        # Trigger once
        det.update(obs_in)
        clock.advance(2.0)
        assert det.update(obs_in) is True  # fired → disarmed

        # Still inside entry zone: should not re-trigger
        clock.advance(2.0)
        assert det.update(obs_in) is False

        # Leave exit zone to re-arm
        det.update(obs_out)  # now armed=True
        assert det.is_armed is True

        # Dwell again
        det.update(obs_in)
        clock.advance(2.0)
        assert det.update(obs_in) is True  # fires again

    def test_hysteresis_between_entry_and_exit(self):
        """Joints between entry_tol and exit_tol: disarmed state should NOT re-arm."""
        det, clock = self._make(entry=10.0, exit_=20.0, dwell=1.0)
        obs_in = {"j.pos": 5.0}  # inside entry (|5|=5 < 10)
        obs_mid = {"j.pos": 15.0}  # outside entry but inside exit
        obs_out = {"j.pos": 25.0}  # outside exit (|25|=25 > 20)

        # Fire once
        det.update(obs_in)
        clock.advance(2.0)
        assert det.update(obs_in) is True

        # In hysteresis zone: should NOT re-arm
        det.update(obs_mid)
        assert det.is_armed is False  # still disarmed

        # Fully out: re-arms
        det.update(obs_out)
        assert det.is_armed is True

    def test_multiple_joints_all_must_enter(self):
        det, clock = self._make(
            middle={"j1.pos": 0.0, "j2.pos": 90.0},
            entry=10.0,
        )
        obs_one_in = {"j1.pos": 5.0, "j2.pos": 200.0}  # j2 way out
        obs_both_in = {"j1.pos": 5.0, "j2.pos": 95.0}

        det.update(obs_one_in)  # only j1 in range
        clock.advance(2.0)
        assert det.update(obs_one_in) is False  # j2 still out

        det.update(obs_both_in)  # both in range, timer starts
        clock.advance(1.0)
        assert det.update(obs_both_in) is True

    def test_missing_key_treated_as_outside(self):
        det, clock = self._make(middle={"j.pos": 0.0}, dwell=1.0)
        obs_empty = {}  # j.pos absent → abs(inf - 0) > tol → outside
        det.update(obs_empty)
        clock.advance(2.0)
        assert det.update(obs_empty) is False  # never inside

    def test_reset_clears_timer(self):
        det, clock = self._make(dwell=1.0)
        obs = {"j.pos": 5.0}
        det.update(obs)
        clock.advance(0.5)
        det.reset()
        clock.advance(0.8)
        assert det.update(obs) is False  # only 0.8s since reset

    def test_force_arm(self):
        det, clock = self._make(dwell=1.0)
        obs_in = {"j.pos": 5.0}

        # Fire and disarm
        det.update(obs_in)
        clock.advance(2.0)
        det.update(obs_in)  # fires → disarmed

        assert det.is_armed is False
        det.force_arm()
        assert det.is_armed is True

        # Can trigger again without leaving the exit zone
        det.update(obs_in)
        clock.advance(2.0)
        assert det.update(obs_in) is True


# ---------------------------------------------------------------------------
# KeyboardEEFController tests (no hardware / no pynput required)
# ---------------------------------------------------------------------------


class TestKeyboardEEFControllerNoPynput:
    """Tests that exercise KeyboardEEFController when pynput is NOT available.

    We patch ``_PYNPUT_AVAILABLE`` to False so no hardware listener is opened.
    """

    def _make_noop(self):
        """Return a controller with pynput patched out."""
        with patch(
            "examples.so101_middle_position_handoff.keyboard_eef._PYNPUT_AVAILABLE",
            False,
        ):
            ctrl = KeyboardEEFController()
        return ctrl

    def test_is_available_false_when_pynput_missing(self):
        ctrl = self._make_noop()
        assert ctrl.is_available is False

    def test_read_action_returns_zero_deltas(self):
        ctrl = self._make_noop()
        action = ctrl.read_action()
        assert action["delta_x"] == 0.0
        assert action["delta_y"] == 0.0
        assert action["delta_z"] == 0.0
        assert action["gripper"] == 1.0  # stay

    def test_events_are_threading_events(self):
        ctrl = self._make_noop()
        assert isinstance(ctrl.resume_requested, threading.Event)
        assert isinstance(ctrl.shutdown_requested, threading.Event)

    def test_events_initially_unset(self):
        ctrl = self._make_noop()
        assert not ctrl.resume_requested.is_set()
        assert not ctrl.shutdown_requested.is_set()

    def test_start_stop_noop_no_error(self):
        ctrl = self._make_noop()
        ctrl.start()  # should not raise
        ctrl.stop()  # should not raise


class TestKeyboardEEFControllerLifecycle:
    def test_start_marks_trusted_live_listener_available(self, monkeypatch):
        import examples.so101_middle_position_handoff.keyboard_eef as mod

        listener = MagicMock()
        listener.is_alive.return_value = True
        fake_keyboard = MagicMock()
        fake_keyboard.Listener.return_value = listener
        monkeypatch.setattr(mod, "keyboard", fake_keyboard)
        monkeypatch.setattr(mod, "_PYNPUT_AVAILABLE", True)
        monkeypatch.setattr(mod, "pynput_can_capture", lambda: True)
        monkeypatch.setattr(mod, "pynput_listener_is_trusted", lambda _listener: True)

        ctrl = KeyboardEEFController()
        ctrl.start()

        assert ctrl.is_available
        listener.start.assert_called_once_with()
        ctrl.stop()
        listener.stop.assert_called_once_with()

    def test_start_rejects_untrusted_listener(self, monkeypatch):
        import examples.so101_middle_position_handoff.keyboard_eef as mod

        listener = MagicMock()
        listener.is_alive.return_value = True
        fake_keyboard = MagicMock()
        fake_keyboard.Listener.return_value = listener
        monkeypatch.setattr(mod, "keyboard", fake_keyboard)
        monkeypatch.setattr(mod, "_PYNPUT_AVAILABLE", True)
        monkeypatch.setattr(mod, "pynput_can_capture", lambda: True)
        monkeypatch.setattr(mod, "pynput_listener_is_trusted", lambda _listener: False)

        ctrl = KeyboardEEFController()
        ctrl.start()

        assert not ctrl.is_available
        listener.stop.assert_called_once_with()


class TestKeyboardEEFControllerKeyState:
    """Tests that exercise the _on_press / _on_release / read_action logic.

    We mock pynput.keyboard at the module level so we can simulate key events.
    """

    def _make_with_fake_keyboard(self):
        """Return (controller, fake_keyboard_module)."""
        # Build a minimal fake keyboard module
        fk = MagicMock()

        class FakeKey:
            left = object()
            right = object()
            up = object()
            down = object()
            shift = object()
            shift_r = object()
            ctrl_r = object()
            ctrl_l = object()
            enter = object()
            esc = object()

        fk.Key = FakeKey

        # Patch both _PYNPUT_AVAILABLE and the module-level `keyboard` reference
        import examples.so101_middle_position_handoff.keyboard_eef as mod

        orig_kb = mod.keyboard
        orig_avail = mod._PYNPUT_AVAILABLE
        mod.keyboard = fk
        mod._PYNPUT_AVAILABLE = True

        ctrl = KeyboardEEFController()
        ctrl._listener = MagicMock()
        ctrl._listener.is_alive.return_value = True
        ctrl._capture_available = True

        return ctrl, fk, mod, orig_kb, orig_avail

    def _restore(self, mod, orig_kb, orig_avail):
        mod.keyboard = orig_kb
        mod._PYNPUT_AVAILABLE = orig_avail

    def test_left_key_press_sets_delta_x_positive(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.left)
            action = ctrl.read_action()
            assert action["delta_x"] == 1.0
            assert action["delta_y"] == 0.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_right_key_press_sets_delta_x_negative(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.right)
            assert ctrl.read_action()["delta_x"] == -1.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_up_key_press_sets_delta_y_negative(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.up)
            assert ctrl.read_action()["delta_y"] == -1.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_down_key_press_sets_delta_y_positive(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.down)
            assert ctrl.read_action()["delta_y"] == 1.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_shift_r_sets_delta_z_positive(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.shift_r)
            assert ctrl.read_action()["delta_z"] == 1.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_shift_sets_delta_z_negative(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.shift)
            assert ctrl.read_action()["delta_z"] == -1.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_ctrl_r_sets_gripper_open(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.ctrl_r)
            assert ctrl.read_action()["gripper"] == 2.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_ctrl_l_sets_gripper_close(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.ctrl_l)
            assert ctrl.read_action()["gripper"] == 0.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_key_release_clears_delta(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.left)
            assert ctrl.read_action()["delta_x"] == 1.0
            ctrl._on_release(fk.Key.left)
            assert ctrl.read_action()["delta_x"] == 0.0
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_enter_sets_resume_requested(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            assert not ctrl.resume_requested.is_set()
            ctrl._on_press(fk.Key.enter)
            assert ctrl.resume_requested.is_set()
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_esc_sets_shutdown_requested(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            assert not ctrl.shutdown_requested.is_set()
            ctrl._on_press(fk.Key.esc)
            assert ctrl.shutdown_requested.is_set()
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_gripper_default_stay_when_no_gripper_key(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            ctrl._on_press(fk.Key.left)  # only movement key
            assert ctrl.read_action()["gripper"] == 1.0  # stay
        finally:
            self._restore(mod, orig_kb, orig_avail)

    def test_no_keys_returns_zero_action(self):
        ctrl, fk, mod, orig_kb, orig_avail = self._make_with_fake_keyboard()
        try:
            action = ctrl.read_action()
            assert action == {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
            }
        finally:
            self._restore(mod, orig_kb, orig_avail)
