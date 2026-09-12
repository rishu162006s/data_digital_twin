#!/usr/bin/env python3
"""
test_anomaly_failover.py

Standalone demo/proof that the anomaly-detection + backup-sensor-failover
feature (anomaly_detector.py) works, for showing a mentor. Prints a clear
tick-by-tick log so you can point at exactly where it fails over and where
maintenance clears it. No xgboost/fastapi needed -- only sklearn/numpy/pandas
(already in requirements.txt), so this runs anywhere, including right here
without deploying anything.

WHAT IT DEMONSTRATES, IN ORDER
  1. NORMAL readings for one asset -> sensor_status stays NORMAL/primary.
  2. Inject one big out-of-range spike per tick, 3 ticks in a row -> the
     3rd one flips sensor_status to SENSOR_FAILURE, active_sensor to
     "backup", maintenance_required to True, and prints the exact dashboard
     message.
  3. Keep sending clean readings afterwards -> status STAYS SENSOR_FAILURE
     (by design -- see anomaly_detector.py's ANOMALY_AUTO_RECOVER; a real
     fault shouldn't clear itself), proving the backup keeps the pipeline
     running (features/RUL scoring don't crash) instead of just erroring out.
  4. Call the maintenance reset (mark_maintenance_done(), same function
     POST /admin/maintenance/{asset_id} calls on the live API) -> status
     goes back to NORMAL/primary.

USAGE
  python3 test_anomaly_failover.py                 # default asset A002
  python3 test_anomaly_failover.py --asset A014
  python3 test_anomaly_failover.py --asset A033 --spike-sigma 25
"""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone

from asset_fleet import build_fleet
from db_pipeline import init_db, seed_assets, ingest_readings_batch


def make_reading(asset, ts, spike=False, spike_sigma=20.0):
    """One hourly reading. spike=True pushes every sensor spike_sigma
    standard deviations past its baseline, in the failure direction --
    clearly outside normal operating range, the way a stuck/glitching
    sensor would look."""
    sensors = []
    for s in asset["sensors"]:
        value = s["baseline_mean"]
        if spike:
            value += spike_sigma * s["baseline_std"] * s["direction"]
        sensors.append({
            "sensor_id": s["sensor_id"], "sensor_type": s["sensor_type"],
            "measurement": s["measurement"], "unit": s["unit"], "value": value,
        })
    return {"asset_id": asset["asset_id"], "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "sensors": sensors}


def log_tick(i, label, health):
    print(f"  tick {i:>2} [{label:<6}] status={health['sensor_status']:<15} "
          f"active_sensor={health['active_sensor']:<8} anomaly_count={health['anomaly_count']} "
          f"maintenance_required={health['maintenance_required']}")
    if health["maintenance_required"]:
        print(f"           dashboard message -> \"{health['status_message']}\"")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", default="A002", help="asset_id to run the demo on (default A002)")
    parser.add_argument("--spike-sigma", type=float, default=20.0,
                         help="how many std-devs past baseline the injected spike goes (default 20)")
    args = parser.parse_args()

    fleet = build_fleet()
    fleet_by_id = {a["asset_id"]: a for a in fleet}
    asset = fleet_by_id.get(args.asset)
    if asset is None:
        parser.error(f"unknown asset_id {args.asset!r}. Valid ids: A001..A036")

    conn = sqlite3.connect(":memory:")  # in-memory DB, nothing written to disk
    init_db(conn)
    seed_assets(conn)

    builders = {}
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hour = 0

    print(f"Asset under test: {asset['asset_id']} ({asset['component_type']} on {asset['machine_name']})")
    print(f"Sensors: {[s['sensor_id'] + ' (' + s['sensor_type'] + ')' for s in asset['sensors']]}\n")

    print("STEP 1: a few normal readings -- should stay NORMAL/primary")
    for _ in range(3):
        builders = ingest_readings_batch(conn, fleet, [make_reading(asset, t0 + timedelta(hours=hour))], builders)
        log_tick(hour, "normal", builders[asset["asset_id"]].last_health)
        hour += 1

    print(f"\nSTEP 2: inject 3 spiked readings in a row ({args.spike_sigma} sigma out of range) "
          f"-- should fail over on the 3rd")
    for _ in range(3):
        builders = ingest_readings_batch(
            conn, fleet, [make_reading(asset, t0 + timedelta(hours=hour), spike=True, spike_sigma=args.spike_sigma)],
            builders)
        log_tick(hour, "SPIKE", builders[asset["asset_id"]].last_health)
        hour += 1

    health = builders[asset["asset_id"]].last_health
    assert health["sensor_status"] == "SENSOR_FAILURE", "FAILED: expected SENSOR_FAILURE after 3 anomalies"
    assert health["active_sensor"] == "backup", "FAILED: expected active_sensor == 'backup'"
    print("  -> PASS: 3 anomalies triggered SENSOR_FAILURE and switched to the backup sensor.")

    print("\nSTEP 3: send clean readings again -- status should STAY SENSOR_FAILURE "
          "(no silent auto-recovery), proving the backup is what's keeping predictions flowing")
    for _ in range(5):
        builders = ingest_readings_batch(conn, fleet, [make_reading(asset, t0 + timedelta(hours=hour))], builders)
        log_tick(hour, "clean", builders[asset["asset_id"]].last_health)
        hour += 1
    health = builders[asset["asset_id"]].last_health
    assert health["sensor_status"] == "SENSOR_FAILURE", "FAILED: status cleared on its own (unexpected)"
    print("  -> PASS: still SENSOR_FAILURE/backup after clean readings, as designed.")

    print("\nSTEP 4: run maintenance (same call POST /admin/maintenance/{asset_id} makes on the live API)")
    builders[asset["asset_id"]].health_monitor.mark_maintenance_done()
    builders = ingest_readings_batch(conn, fleet, [make_reading(asset, t0 + timedelta(hours=hour))], builders)
    health = builders[asset["asset_id"]].last_health
    log_tick(hour, "fixed", health)
    assert health["sensor_status"] == "NORMAL", "FAILED: maintenance did not clear the failure"
    assert health["active_sensor"] == "primary", "FAILED: active_sensor did not return to 'primary'"
    print("  -> PASS: maintenance cleared the failure, back to NORMAL/primary.")

    print("\nALL CHECKS PASSED.")
    conn.close()


if __name__ == "__main__":
    main()
