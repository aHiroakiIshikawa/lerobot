"""Dwell detector: fires when all monitored joints stay within a tolerance band.

Hysteresis prevents re-triggering until the pose leaves the exit-tolerance region.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DwellDetectorConfig:
    """Configuration for :class:`DwellDetector`.

    Parameters
    ----------
    middle_positions:
        Mapping of joint key (e.g. ``"shoulder_pan.pos"``) to target angle
        in degrees.  All listed joints must be within ``entry_tolerance_deg``
        for the dwell timer to start.
    entry_tolerance_deg:
        Maximum deviation from ``middle_positions`` that still counts as
        "inside the zone".
    exit_tolerance_deg:
        Any joint must exceed this tolerance before the detector re-arms.
        Must be >= ``entry_tolerance_deg`` to avoid immediate re-trigger.
    dwell_time_s:
        How long (seconds) all joints must stay inside the entry zone before
        the detector fires.
    entry_dropout_grace_s:
        How long a temporary excursion outside the entry zone may last without
        resetting an in-progress dwell.  The detector can only fire while all
        joints are back inside the entry zone.
    """

    middle_positions: dict[str, float]
    entry_tolerance_deg: float = 10.0
    exit_tolerance_deg: float = 15.0
    dwell_time_s: float = 1.0
    entry_dropout_grace_s: float = 0.1

    def __post_init__(self) -> None:
        if not self.middle_positions:
            raise ValueError("middle_positions must not be empty")
        if self.exit_tolerance_deg < self.entry_tolerance_deg:
            raise ValueError(
                f"exit_tolerance_deg ({self.exit_tolerance_deg}) must be >= "
                f"entry_tolerance_deg ({self.entry_tolerance_deg})"
            )
        if self.dwell_time_s <= 0:
            raise ValueError(f"dwell_time_s must be positive, got {self.dwell_time_s}")
        if self.entry_dropout_grace_s < 0:
            raise ValueError(f"entry_dropout_grace_s must be non-negative, got {self.entry_dropout_grace_s}")


class DwellDetector:
    """Fires once when all joints dwell near the middle position for ``dwell_time_s``.

    Re-arming requires the pose to leave the exit-tolerance region (at least one
    joint exceeds its exit band), providing hysteresis so a single triggering
    doesn't immediately re-trigger on the next frame.

    Parameters
    ----------
    cfg:
        Detector configuration.
    clock:
        Callable returning the current time in seconds.  Defaults to
        ``time.perf_counter``.  Inject a fake clock in unit tests.
    """

    def __init__(
        self,
        cfg: DwellDetectorConfig,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._entered_at: float | None = None
        self._outside_since: float | None = None
        self._armed: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obs: dict[str, float]) -> bool:
        """Update detector state from the latest robot observation.

        Parameters
        ----------
        obs:
            Raw robot observation dict (e.g. ``{"shoulder_pan.pos": 45.0, ...}``).

        Returns
        -------
        bool
            ``True`` exactly once when the dwell condition is first satisfied.
            Returns ``False`` on all other frames, including during the dwell
            window and after the detector has fired (until it re-arms).
        """
        cfg = self._cfg

        all_in_entry = all(
            abs(obs.get(k, float("inf")) - v) <= cfg.entry_tolerance_deg
            for k, v in cfg.middle_positions.items()
        )
        any_out_exit = any(
            abs(obs.get(k, float("inf")) - v) > cfg.exit_tolerance_deg
            for k, v in cfg.middle_positions.items()
        )

        now = self._clock()

        # --- Disarmed: wait until all joints exit the exit band ---
        if not self._armed:
            if any_out_exit:
                self._armed = True
                self._entered_at = None
                self._outside_since = None
            return False

        # --- Armed ---
        if all_in_entry:
            if self._outside_since is not None:
                if now - self._outside_since > cfg.entry_dropout_grace_s:
                    self._entered_at = None
                self._outside_since = None
            if self._entered_at is None:
                self._entered_at = now
            elif now - self._entered_at >= cfg.dwell_time_s:
                # Fire once, then disarm
                self._armed = False
                self._entered_at = None
                self._outside_since = None
                return True
        else:
            if self._entered_at is not None:
                if cfg.entry_dropout_grace_s == 0:
                    self._entered_at = None
                elif self._outside_since is None:
                    self._outside_since = now
                elif now - self._outside_since > cfg.entry_dropout_grace_s:
                    self._entered_at = None
                    self._outside_since = None

        return False

    def reset(self) -> None:
        """Reset the dwell timer without changing the armed state."""
        self._entered_at = None
        self._outside_since = None

    def force_arm(self) -> None:
        """Re-arm the detector regardless of current joint positions."""
        self._armed = True
        self._entered_at = None
        self._outside_since = None

    @property
    def is_armed(self) -> bool:
        """``True`` if the detector will fire on the next qualifying dwell."""
        return self._armed

    @property
    def dwell_elapsed_s(self) -> float:
        """Elapsed wall time for the current dwell attempt, or zero when idle."""
        if self._entered_at is None:
            return 0.0
        return max(0.0, self._clock() - self._entered_at)
