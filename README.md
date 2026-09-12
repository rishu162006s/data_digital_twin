# Digital Twin RUL API

Serves live remaining-useful-life predictions for all 36 assets, for a
digital twin dashboard to poll.

## What's in here

| File | Role |
|---|---|
| `app.py` | **New.** FastAPI service — the thing you deploy. |
| `asset_fleet.py` | Your 36-asset fleet definition + `MODEL_FEATURE_ORDER`. |
| `score_stream.py` | Rolling z-score feature builder + RUL→state binning. |
| `db_pipeline.py` | SQLite ingest/feature/score/join pipeline. |
| `bharati_sensor_simulator.py` | Generates realistic hourly sensor readings per asset. |
| `run_live_pipeline.py` | Original CLI loop (kept for offline/manual runs — the API doesn't call this file, it re-implements the same loop as a background thread so it can serve HTTP requests concurrently). |
| `xgboost_rul_model.joblib` | Your trained model. |
| `anomaly_detector.py` | **New.** Isolation Forest + hard-rule anomaly detection and sensor-failure/backup-failover logic. See "Anomaly detection & backup sensor failover" below. |
| `test_anomaly_failover.py` | **New.** Standalone offline demo/test script (PASS/FAIL asserts) proving the anomaly-detection feature works — see "How to test/demo this" below. |
| `requirements.txt`, `render.yaml`, `Procfile` | Deployment config. |

## Anomaly detection & backup sensor failover

Every tick, each asset's reading now goes through two checks before it's
trusted (implemented in `anomaly_detector.py`, wired into
`score_stream.py`'s `AssetFeatureBuilder.add()`, which already computed the
per-sensor z-scores this reuses):

1. **Hard rule (stage 1):** if any single sensor's z-score has `|z| > 4.5`,
   that's an obvious spike — flagged immediately, no history needed.
2. **Isolation Forest (stage 2):** only checked if stage 1 didn't already
   catch it. Once an asset has 20+ clean ticks of history, an
   `IsolationForest` is trained on that asset's own rolling window of
   `[z_mean, z_std, z_max, z_min]` and re-trained every 5 ticks (scikit-learn's
   `IsolationForest` has no incremental/`partial_fit` mode, so "retrain on a
   rolling window" is the standard approach for streaming data).

A tick is anomalous if **either** stage flags it. Each asset keeps a rolling
window of the last 10 flags; **once 3 of them are anomalous, the asset flips
to `SENSOR_FAILURE`** and fails over to a backup reading, then automatically
recovers back to `NORMAL` after 5 consecutive clean ticks.

**Backup sensor, honestly stated:** this fleet has no redundant physical
sensor per measurement (each component has 2-3 *different* sensor types, not
duplicate pairs — see `asset_fleet.py`). So "backup sensor" here means: stop
trusting the live reading and fall back to the **median of the last few
known-good (non-anomalous) readings**, so the model keeps scoring off a
trusted recent trend instead of a glitch. If you later add real redundant
hardware, swap `AssetHealthMonitor._backup_vector()` for that second sensor's
z-vector — everything downstream stays the same.

This does **not** change the model's input features or feature order — it
only decides *which* z-values (fresh vs. backup) feed into the existing
z_mean/z_std/z_max/z_min pipeline, so `xgboost_rul_model.joblib` does not
need retraining.

**New fields**, exposed on every asset via `GET /predictions`,
`GET /predictions/{asset_id}`, and the new `GET /sensor-status` (only assets
currently failed):
```json
{
  "sensor_status": "SENSOR_FAILURE",   // or "NORMAL"
  "active_sensor": "backup",           // or "primary"
  "anomaly_count": 3,                  // flagged ticks in the last 10
  "maintenance_required": true,        // dashboard: show a "needs maintenance" badge when true
  "status_message": "Sensor failure detected (3 anomalies in the last 10 readings). Running on BACKUP sensor -- maintenance required."
}
```
A failure **stays** `SENSOR_FAILURE`/`maintenance_required: true` even once
readings go back to normal — it does not silently clear itself (a real
sensor fault needs a technician, not a timer). Clear it with:
```bash
curl -X POST "https://<your-service>.onrender.com/admin/maintenance/A002?token=<ADMIN_TOKEN>"
```

**Tuning (env vars, all optional):**
`ANOMALY_HARD_Z_THRESHOLD` (4.5), `ANOMALY_WINDOW` (10),
`ANOMALY_FAILURE_COUNT` (3), `ANOMALY_AUTO_RECOVER` (false — set "true" to
let a clean streak auto-clear a failure instead of requiring
`/admin/maintenance`), `ANOMALY_RECOVERY_STREAK` (5, only used if
`ANOMALY_AUTO_RECOVER=true`), `IFOREST_MIN_SAMPLES` (20), `IFOREST_REFIT_EVERY`
(5), `IFOREST_WINDOW` (100), `IFOREST_CONTAMINATION` (0.1).

### How to test/demo this (for your mentor)

**Option A — offline, no deploy needed, ~2 seconds, prints PASS/FAIL:**
```bash
pip install -r requirements.txt   # only needs sklearn/numpy/pandas for this part
python3 test_anomaly_failover.py --asset A002
```
Runs 3 normal ticks, injects 3 spiked ticks (asserts failure + backup kick
in on the 3rd), sends 5 more clean ticks (asserts it stays failed, not
auto-cleared), then runs maintenance (asserts it clears). Prints a
tick-by-tick log plus the exact dashboard message at each failure step —
good to run live in front of your mentor or screenshot.

**Option B — against your actual deployed Render API:**
```bash
BASE=https://<your-service>.onrender.com
TOKEN=<your ADMIN_TOKEN>
ASSET=A002
SENSOR=S0021   # first sensor on A002 -- see GET /admin/query?table=assets or asset_fleet.py

# 1) baseline -- should be NORMAL/primary
curl -s "$BASE/predictions/$ASSET" | python3 -m json.tool

# 2) push 3 wildly out-of-range readings in a row (each /admin/inject call
#    is one tick) -- the 3rd response's new_prediction should flip to SENSOR_FAILURE
for i in 1 2 3; do
  curl -s -X POST "$BASE/admin/inject?token=$TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"asset_id\": \"$ASSET\", \"overrides\": {\"$SENSOR\": 9999}}" | python3 -m json.tool
done

# 3) confirm it's on the failures list, with the dashboard message
curl -s "$BASE/sensor-status" | python3 -m json.tool

# 4) prove the backup is actually serving requests (still a valid
#    predicted_rul_days, not an error, while active_sensor == "backup")
curl -s "$BASE/predictions/$ASSET" | python3 -m json.tool

# 5) run maintenance -- back to NORMAL/primary
curl -s -X POST "$BASE/admin/maintenance/$ASSET?token=$TOKEN" | python3 -m json.tool
curl -s "$BASE/predictions/$ASSET" | python3 -m json.tool
```

## How it works

On startup, a background thread repeats the same cycle as
`run_live_pipeline.py` — simulate one more hour → ingest → build features →
`model.predict()` → join asset metadata — but ticks every `TICK_SECONDS`
**real** seconds (default 10) instead of a real hour, so your digital twin
visibly updates. Each tick's results are cached in memory, so `GET`
requests are instant and never touch SQLite on the request path.

## Endpoints

- `GET /predictions` — latest prediction for **all** assets (array), e.g.:
  ```json
  [
    {
      "asset_id": "A002",
      "timestamp": "2025-05-01T12:00:00Z",
      "predicted_rul_days": 23.74,
      "current_state": "DEGRADING",
      "component_id": "C002",
      "component_type": "Bearing",
      "machine_id": "M002",
      "machine_name": "CHP Unit 1",
      "room_id": "R001",
      "sensor_status": "NORMAL",
      "active_sensor": "primary",
      "anomaly_count": 0,
      "maintenance_required": false,
      "status_message": "Sensor operating normally (primary)."
    },
    ...
  ]
  ```
- `GET /predictions/{asset_id}` — single asset, e.g. `/predictions/A002`. `404` until that asset's first tick completes.
- `GET /sensor-status` — only assets currently in `SENSOR_FAILURE` (see "Anomaly detection & backup sensor failover" above).
- `GET /health` — `{status, assets_tracked, hours_simulated, last_tick_utc, tick_seconds, error}`. Point your dashboard's connectivity check and Render's health check here.

## Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, point it at the repo — `render.yaml` configures everything automatically. (Or **New → Web Service** manually: build command `pip install -r requirements.txt`, start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.)
3. Once live, poll `https://<your-service>.onrender.com/predictions` from your digital twin frontend every `TICK_SECONDS` (or a bit slower) for the live feed.

**Schema note:** `init_db()` now also creates a `sensor_health` table and adds
`sensor_status` / `active_sensor` / `anomaly_count` columns to `predictions`.
On Render's free tier this is moot (see below — the DB is ephemeral and
recreated fresh on every deploy/restart anyway). If you're running this
against a **persisted** database from before this change, delete the old
`.db` file (or its `predictions`/`model_features` tables) once so it gets
rebuilt with the new columns — `CREATE TABLE IF NOT EXISTS` won't retrofit
columns onto an existing table.

**Free-tier note:** Render's free web services spin down after 15 min idle and cold-start on the next request (losing in-memory state and restarting the simulated clock from `START_TIME`/now). SQLite history in `bharati.db` also lives on ephemeral disk — it resets on redeploy/restart. For a persistent 24/7 twin, use a paid instance (no spin-down) and/or swap SQLite for a managed Postgres add-on — `db_pipeline.py`'s comments note this is just a connection-string change.

## Tuning

- `TICK_SECONDS` — real seconds per simulated hour. Lower = faster-moving twin, higher CPU/DB churn.
- `SEED` — simulator RNG seed, for reproducible demo runs.
- `START_TIME` — e.g. `2025-05-01T00:00:00Z` to pin the simulated clock instead of starting from "now".

## Local test

```bash
pip install -r requirements.txt
TICK_SECONDS=5 uvicorn app:app --reload
curl http://localhost:8000/predictions
```
