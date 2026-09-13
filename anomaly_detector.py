#!/usr/bin/env python3
"""
anomaly_detector.py

Two-stage anomaly check per asset per tick, plus the backup-sensor-failover
state machine.

STAGE 1 (hard rule): any single sensor z-score with |z| > ANOMALY_HARD_Z_THRESHOLD
  is flagged immediately.
STAGE 2 (isolation forest): only checked if stage 1 didn't already flag it.
  Needs IFOREST_MIN_SAMPLES clean ticks of [z_mean, z_std, z_max, z_min]
  history before it turns on, refit every IFOREST_REFIT_EVERY ticks on a
  rolling IFOREST_WINDOW.

A tick is anomalous if either stage flags it. AssetHealthMonitor keeps a
rolling window (ANOMALY_WINDOW) of the last flags; once ANOMALY_FAILURE_COUNT
of them are anomalous, the asset flips to SENSOR_FAILURE and fails over to a
"backup" reading (median-ish average of the last few known-good z-vectors).

RECOVERY: a SENSOR_FAILURE clears itself automatically SENSOR_AUTO_FIX_SECONDS
(default 60) real seconds after it started -- "the technician swaps in the
spare within a minute", for a live demo. Can also be cleared instantly via
mark_maintenance_done() (POST /admin/maintenance/{asset_id} calls this).
"""
import os
import time
from collections import deque

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

ANOMALY_HARD_Z_THRESHOLD = float(os.environ.get("ANOMALY_HARD_Z_THRESHOLD", "4.5"))
ANOMALY_WINDOW = int(os.environ.get("ANOMALY_WINDOW", "10"))
ANOMALY_FAILURE_COUNT = int(os.environ.get("ANOMALY_FAILURE_COUNT", "3"))
ANOMALY_AUTO_RECOVER = os.environ.get("ANOMALY_AUTO_RECOVER", "false").lower() == "true"
ANOMALY_RECOVERY_STREAK = int(os.environ.get("ANOMALY_RECOVERY_STREAK", "5"))
IFOREST_MIN_SAMPLES = int(os.environ.get("IFOREST_MIN_SAMPLES", "20"))
IFOREST_REFIT_EVERY = int(os.environ.get("IFOREST_REFIT_EVERY", "5"))
IFOREST_WINDOW = int(os.environ.get("IFOREST_WINDOW", "100"))
IFOREST_CONTAMINATION = float(os.environ.get("IFOREST_CONTAMINATION", "0.1"))
# Exactly this many real seconds after a SENSOR_FAILURE starts, it auto-clears
# back to NORMAL/primary -- independent of ANOMALY_AUTO_RECOVER above.
SENSOR_AUTO_FIX_SECONDS = float(os.environ.get("SENSOR_AUTO_FIX_SECONDS", "60"))


class AssetHealthMonitor:
    def __init__(self):
        self.status = "NORMAL"
        self.active_sensor = "primary"
        self.flags = deque(maxlen=ANOMALY_WINDOW)
        self.clean_streak = 0
        self.good_history = deque(maxlen=20)
        self.feature_history = deque(maxlen=IFOREST_WINDOW)
        self._iforest = None
        self._ticks_since_fit = 0
        self.failure_started_at = None  # time.monotonic() when it tripped

    def mark_maintenance_done(self):
        self.status = "NORMAL"
        self.active_sensor = "primary"
        self.flags.clear()
        self.clean_streak = 0
        self.failure_started_at = None

    def force_failure(self):
        """Immediately trip this asset into SENSOR_FAILURE/backup, instead of
        waiting for ANOMALY_FAILURE_COUNT real anomalous ticks to accumulate.
        Used by fault_scenario.trigger_bit_bad()/trigger_worst() so a POST to
        /scenario/* has a visible effect on the very next tick rather than
        depending on random z-score drift crossing the threshold on its own."""
        self.flags.clear()
        for _ in range(ANOMALY_FAILURE_COUNT):
            self.flags.append(True)
        self.status = "SENSOR_FAILURE"
        self.active_sensor = "backup"
        self.clean_streak = 0
        self.failure_started_at = time.monotonic()

    def _hard_rule_hit(self, zs):
        return any(abs(z) > ANOMALY_HARD_Z_THRESHOLD for z in zs)

    def _iforest_hit(self, feat_vec):
        if IsolationForest is None:
            return None
        self.feature_history.append(feat_vec)
        if len(self.feature_history) < IFOREST_MIN_SAMPLES:
            return None
        if self._iforest is None or self._ticks_since_fit >= IFOREST_REFIT_EVERY:
            self._iforest = IsolationForest(contamination=IFOREST_CONTAMINATION, random_state=0)
            self._iforest.fit(list(self.feature_history))
            self._ticks_since_fit = 0
        self._ticks_since_fit += 1
        pred = self._iforest.predict([feat_vec])[0]  # -1 anomaly, 1 normal
        return pred == -1

    def _backup_vector(self):
        if not self.good_history:
            return (0.0, 0.0, 0.0, 0.0)
        n = len(self.good_history)
        return tuple(sum(v[i] for v in self.good_history) / n for i in range(4))

    def update(self, zs):
        """zs: per-sensor z-scores for this tick. Returns (health_dict, backup_vector_or_None)."""
        z_mean = sum(zs) / len(zs)
        z_max, z_min = max(zs), min(zs)
        z_std = (sum((z - z_mean) ** 2 for z in zs) / len(zs)) ** 0.5 if len(zs) > 1 else 0.0
        feat_vec = [z_mean, z_std, z_max, z_min]

        hard_hit = self._hard_rule_hit(zs)
        iforest_hit = None if hard_hit else self._iforest_hit(feat_vec)
        is_anomaly = bool(hard_hit or iforest_hit)
        self.flags.append(is_anomaly)

        if is_anomaly:
            self.clean_streak = 0
        else:
            self.good_history.append(feat_vec)
            self.clean_streak += 1

        # time-based auto-fix, fires regardless of ANOMALY_AUTO_RECOVER. This
        # runs BEFORE anomaly_count/status_message are computed below, so a
        # tick where the 60s timer fires reports the POST-reset state
        # (anomaly_count back to 0, status NORMAL) instead of one stale tick
        # of the old failed state.
        if self.status == "SENSOR_FAILURE" and self.failure_started_at is not None:
            if time.monotonic() - self.failure_started_at >= SENSOR_AUTO_FIX_SECONDS:
                self.mark_maintenance_done()

        # legacy clean-streak auto-recover, opt-in only
        if self.status == "SENSOR_FAILURE" and ANOMALY_AUTO_RECOVER:
            if self.clean_streak >= ANOMALY_RECOVERY_STREAK:
                self.mark_maintenance_done()

        anomaly_count = sum(self.flags)

        if self.status == "NORMAL" and anomaly_count >= ANOMALY_FAILURE_COUNT:
            self.status = "SENSOR_FAILURE"
            self.active_sensor = "backup"
            self.failure_started_at = time.monotonic()

        backup_vec = self._backup_vector() if self.status == "SENSOR_FAILURE" else None
        maintenance_required = self.status == "SENSOR_FAILURE"

        # No countdown/auto-fix promise in the user-facing message -- the
        # 60s auto-fix still happens internally (above), it's just not
        # narrated, so the dashboard doesn't show a timer that can be wrong
        # if the asset re-trips before it fires.
        if self.status == "SENSOR_FAILURE":
            status_message = (
                f"Sensor failure detected ({anomaly_count} anomalies in the "
                f"last {ANOMALY_WINDOW} readings). Running on backup sensor "
                f"-- maintenance required."
            )
        else:
            status_message = "Sensor operating normally (primary)."

        health = {
            "sensor_status": self.status,
            "active_sensor": self.active_sensor,
            "anomaly_count": anomaly_count,
            "is_anomaly": is_anomaly,
            "hard_rule_hit": hard_hit,
            "iforest_hit": iforest_hit,
            "maintenance_required": maintenance_required,
            "status_message": status_message,
        }
        return health, backup_vec
