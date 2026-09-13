#!/usr/bin/env python3
"""
fault_scenario.py

Controls WHICH assets/sensors actually degrade in the simulator, so the
whole 36-asset fleet doesn't gradually crash together. Everything else stays
close to NORMAL forever.

SCENARIOS (switch via POST /scenario/<name> in app.py):
  normal   (default, also the starting state) -- 2-3 assets have a real
           slow degrading trajectory, 1-2 sensors on each of THOSE assets
           carry the failing signal. Every other asset/sensor just jitters
           near a healthy baseline.
  bit_bad  -- 4-5 assets get pushed into a bad health state (low RUL) and
           STAY bad -- they do not self-heal -- until you fix them with
           POST /admin/maintenance/{asset_id}. 2-3 sensors per picked asset
           start drifting.
  worst    -- sudden shock: 7-8 sensors across the fleet jump close to
           failure, 2-3 of them hard enough to actually trip SENSOR_FAILURE.
           Assets that were only "touched" (not one of the hard-fail ones)
           get their predicted_rul_days smoothed (see dampen_for(), used by
           db_pipeline.score_pending) so the twin doesn't look like it's
           melting down just because a couple of raw values spiked.
"""
import random
import threading

_lock = threading.Lock()

_state = {
    "scenario": "normal",
    "fault_assets": set(),     # asset_ids with a real degrading trajectory
    "fault_sensors": {},       # asset_id -> set(sensor_id) carrying the drift
    "bad_until_fixed": set(),  # asset_ids stuck bad until /admin/maintenance
    "dampened_assets": set(),  # asset_ids whose predicted_rul gets smoothed
}


def init(fleet, seed=42):
    """Pick the default 2-3 gradually-degrading assets (1-2 sensors each).
    Call once at startup."""
    rng = random.Random(seed)
    fleet_by_id = {a["asset_id"]: a for a in fleet}
    chosen = rng.sample(list(fleet_by_id), rng.randint(2, 3))
    with _lock:
        _state["scenario"] = "normal"
        _state["fault_assets"] = set(chosen)
        _state["fault_sensors"] = {
            aid: set(rng.sample(
                [s["sensor_id"] for s in fleet_by_id[aid]["sensors"]],
                k=min(rng.randint(1, 2), len(fleet_by_id[aid]["sensors"])),
            ))
            for aid in chosen
        }
        _state["bad_until_fixed"] = set()
        _state["dampened_assets"] = set()


def is_fault_asset(asset_id):
    with _lock:
        return asset_id in _state["fault_assets"]


def fault_sensor_ids(asset_id):
    with _lock:
        return set(_state["fault_sensors"].get(asset_id, ()))


def dampen_for(asset_id):
    with _lock:
        return asset_id in _state["dampened_assets"]


def status():
    with _lock:
        return {
            "scenario": _state["scenario"],
            "fault_assets": sorted(_state["fault_assets"]),
            "fault_sensors": {k: sorted(v) for k, v in _state["fault_sensors"].items()},
            "bad_until_fixed": sorted(_state["bad_until_fixed"]),
            "dampened_assets": sorted(_state["dampened_assets"]),
        }


def trigger_bit_bad(fleet, trajectories, builders=None, seed=None):
    """4-5 assets pushed into a bad health state and forced straight into
    SENSOR_FAILURE on the next tick (not left to accumulate anomalies over
    several ticks by chance) -- they stay bad until
    /admin/maintenance/{asset_id} fixes them. `builders` is app.py's
    _pipeline["builders"] dict ({asset_id: AssetFeatureBuilder}); pass it in
    so the force-fail actually reaches each asset's AssetHealthMonitor."""
    rng = random.Random(seed)
    fleet_by_id = {a["asset_id"]: a for a in fleet}
    chosen = rng.sample(list(fleet_by_id), rng.randint(4, 5))
    with _lock:
        _state["scenario"] = "bit_bad"
        for aid in chosen:
            _state["fault_assets"].add(aid)
            _state["bad_until_fixed"].add(aid)
            sensors = [s["sensor_id"] for s in fleet_by_id[aid]["sensors"]]
            k = min(rng.randint(2, 3), len(sensors))
            _state["fault_sensors"].setdefault(aid, set()).update(rng.sample(sensors, k))
        for aid in chosen:
            if aid in trajectories:
                trajectories[aid].health = max(trajectories[aid].health, rng.uniform(0.55, 0.75))
            if builders and aid in builders:
                builders[aid].health_monitor.force_failure()
    return chosen


def trigger_worst(fleet, trajectories, builders=None, seed=None):
    """Sudden shock: 10-14 sensors across the fleet touched, and MOST of
    them (not just 2-3) are forced into an actual hard SENSOR_FAILURE right
    away -- "worst" means more parts actually fail than bit_bad, not a
    softened version of it. `builders` is app.py's _pipeline["builders"]
    dict; pass it in so the force-fail reaches each AssetHealthMonitor."""
    rng = random.Random(seed)
    fleet_by_id = {a["asset_id"]: a for a in fleet}
    n_sensors_total = rng.randint(10, 14)
    n_hard = max(1, round(n_sensors_total * rng.uniform(0.7, 0.9)))

    pool = list(fleet_by_id)
    rng.shuffle(pool)
    picked = []
    for aid in pool:
        if len(picked) >= n_sensors_total:
            break
        sensors = [s["sensor_id"] for s in fleet_by_id[aid]["sensors"]]
        take = min(rng.randint(1, 2), len(sensors), n_sensors_total - len(picked))
        if take <= 0:
            continue
        for sid in rng.sample(sensors, take):
            picked.append((aid, sid))

    hard_fail = set(rng.sample(picked, min(n_hard, len(picked)))) if picked else set()
    hard_fail_assets = {aid for aid, _sid in hard_fail}

    with _lock:
        _state["scenario"] = "worst"
        for aid, sid in picked:
            _state["fault_assets"].add(aid)
            _state["fault_sensors"].setdefault(aid, set()).add(sid)
        for aid in hard_fail_assets:
            _state["bad_until_fixed"].add(aid)
            if aid in trajectories:
                trajectories[aid].health = max(trajectories[aid].health, rng.uniform(0.85, 0.95))
            if builders and aid in builders:
                builders[aid].health_monitor.force_failure()
        # Only the handful of touched-but-not-hard-failed assets get
        # dampened, so they read as "a bit rattled" rather than critical --
        # the majority now genuinely fail instead of being smoothed over.
        for aid, _sid in picked:
            if aid not in hard_fail_assets:
                _state["dampened_assets"].add(aid)

    return {"touched": picked, "hard_fail": sorted(hard_fail)}


def reset(fleet, trajectories, seed=42):
    """Back to the default normal scenario; non-fault assets get their
    health pulled back down so RUL recovers."""
    init(fleet, seed=seed)
    with _lock:
        keep = set(_state["fault_assets"])
    for a in fleet:
        aid = a["asset_id"]
        if aid in trajectories and aid not in keep:
            trajectories[aid].health = min(trajectories[aid].health, 0.05)


def clear_asset(asset_id, trajectories):
    """Called by POST /admin/maintenance/{asset_id}: asset is fixed, drops
    out of bad_until_fixed/dampened, health resets low."""
    with _lock:
        _state["bad_until_fixed"].discard(asset_id)
        _state["dampened_assets"].discard(asset_id)
    if asset_id in trajectories:
        trajectories[asset_id].health = 0.02
