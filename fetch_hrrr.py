"""
FIRESTORM HRRR Wind Pipeline — CONUS + Alaska
=============================================

Pulls the latest available HRRR 10m wind (UGRD + VGRD) from NOAA NOMADS,
resamples the native Lambert Conformal grid onto a regular lat/lng grid,
and emits cambecc/earth-format JSON for FIRESTORM's particle renderer.

CONUS:  fetched via NOMADS filter CGI (server-side subset, ~4-5 MB)
Alaska: fetched via GRIB2 byte-range from the full surface file (~2-3 MB)
        NOMADS filter CGI rejects Alaska filename pattern, so we parse
        the .idx index and issue an HTTP Range request ourselves.

Refresh: hourly (CONUS) / every 3h (Alaska) — the GitHub Actions cron
runs hourly; Alaska outputs are simply stale 1-2 runs out of 3 without
breaking anything. The frontend treats missing/stale as fallback to GFS.

Backlog — Option 3 in the FIRESTORM hierarchy: extend to regional
high-resolution models for international coverage (ICON-EU / AROME /
ACCESS-C / HRDPS). Deferred — audience is US-primary.
"""

import os
import sys
import json
import time
import datetime as dt
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import numpy as np
import xarray as xr
from scipy.interpolate import griddata


NOMADS_BASE = "https://nomads.ncep.noaa.gov"
UA = "FIRESTORM-HRRR-Pipeline/1.0 (github.com/Deasus/firestorm-hrrr-data)"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Target regular grids (cambecc/earth format) ─────────────────────
# 0.1° resolution preserves essentially all HRRR detail (native ~0.027°).
# Going finer would inflate JSON size without meaningful gain.
TARGET = {
    "conus": {
        "lo1": -127.0, "la1": 50.0,      # top-left (la1 is north edge in cambecc format)
        "dx": 0.1, "dy": 0.1,
        "nx": int((-64.0 - (-127.0)) / 0.1),   # 630
        "ny": int((50.0 - 22.0) / 0.1),         # 280
    },
    "alaska": {
        # HRRR-AK domain covers 41–77°N, 156°E through ~244°E (wraps Aleutians).
        # We emit the grid in 0..360 longitude space to avoid antimeridian
        # wrapping issues in the renderer. Frontend subtracts 360 where lng>180.
        "lo1": 156.0, "la1": 78.0,
        "dx": 0.15, "dy": 0.15,    # slightly coarser target — Alaska is huge
        "nx": int((244.0 - 156.0) / 0.15),    # ~586
        "ny": int((78.0 - 41.0) / 0.15),       # ~246
    },
}


# ── URL builders ─────────────────────────────────────────────────────

def _latest_run(hours_per_cycle):
    """
    Return (date_YYYYMMDD, run_HH) for the most recent run likely to be
    fully written out. NOMADS takes ~80min to publish all hourly forecasts
    after run start, so we back off by 2h to avoid racing partial writes.
    """
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    run_hr = (now.hour // hours_per_cycle) * hours_per_cycle
    run_time = now.replace(hour=run_hr, minute=0, second=0, microsecond=0)
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")


def _conus_url(date, run):
    """NOMADS filter CGI — server-side subset, returns just UGRD+VGRD at 10m."""
    return (
        f"{NOMADS_BASE}/cgi-bin/filter_hrrr_2d.pl"
        f"?dir=/hrrr.{date}/conus"
        f"&file=hrrr.t{run}z.wrfsfcf00.grib2"
        f"&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
    )


def _ak_idx_url(date, run):
    return (f"{NOMADS_BASE}/pub/data/nccf/com/hrrr/prod/hrrr.{date}"
            f"/alaska/hrrr.t{run}z.wrfsfcf00.ak.grib2.idx")


def _ak_data_url(date, run):
    return (f"{NOMADS_BASE}/pub/data/nccf/com/hrrr/prod/hrrr.{date}"
            f"/alaska/hrrr.t{run}z.wrfsfcf00.ak.grib2")


# ── Download helpers ─────────────────────────────────────────────────

def _http_get(url, timeout=60, byte_range=None):
    """Minimal urllib wrapper — no requests dep, runs in clean Actions env."""
    req = Request(url, headers={"User-Agent": UA})
    if byte_range is not None:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_conus(date, run, dest_path):
    url = _conus_url(date, run)
    print(f"  CONUS → {url}")
    body = _http_get(url, timeout=120)
    if len(body) < 100000:    # sanity check — real CONUS subset is ~4.5 MB
        raise RuntimeError(f"CONUS fetch too small ({len(body)} bytes)")
    with open(dest_path, "wb") as f:
        f.write(body)
    print(f"  CONUS wrote {len(body)/1024/1024:.1f} MB → {dest_path}")


def _fetch_alaska(date, run, dest_path):
    """
    Two-step byte-range fetch:
      1. Download the ~9KB .idx file.
      2. Find UGRD/VGRD at 10m records (should be consecutive).
      3. Issue an HTTP Range request for just those bytes.
    """
    idx = _http_get(_ak_idx_url(date, run), timeout=30).decode("utf-8", errors="replace")
    # Parse idx lines: "record_num:byte_offset:date:var:level:..."
    lines = [l for l in idx.strip().split("\n") if l]
    records = [l.split(":") for l in lines]
    u_idx, v_idx, next_idx = None, None, None
    for i, r in enumerate(records):
        if len(r) < 5:
            continue
        if r[3] == "UGRD" and r[4] == "10 m above ground":
            u_idx = i
        elif r[3] == "VGRD" and r[4] == "10 m above ground":
            v_idx = i
            if i + 1 < len(records):
                next_idx = i + 1
    if u_idx is None or v_idx is None:
        raise RuntimeError("HRRR-AK idx missing UGRD/VGRD at 10 m above ground")
    if v_idx != u_idx + 1:
        # Not strictly required but our range fetch assumes adjacency
        print(f"  WARN: HRRR-AK UGRD and VGRD not adjacent in idx (u={u_idx}, v={v_idx}) — will fetch full span")
    start = int(records[u_idx][1])
    if next_idx is not None:
        end = int(records[next_idx][1]) - 1
    else:
        # Last record — need to fetch to EOF. Let urllib handle it with a huge end.
        end = start + 50 * 1024 * 1024
    size = end - start + 1
    print(f"  Alaska byte range: {start}-{end} ({size/1024/1024:.1f} MB)")
    body = _http_get(_ak_data_url(date, run), timeout=120, byte_range=(start, end))
    with open(dest_path, "wb") as f:
        f.write(body)
    print(f"  Alaska wrote {len(body)/1024/1024:.1f} MB → {dest_path}")


# ── GRIB2 parse + resample ───────────────────────────────────────────

def _resample_lambert_to_regular(grib_path, target, region_name):
    """
    HRRR is on a Lambert Conformal projection. cfgrib attaches 2D lat/lng
    coordinate arrays to each variable. We scatter those onto a regular
    lat/lng grid using scipy.interpolate.griddata.

    method="linear" gives smoother fields than "nearest" but is ~2-3x slower
    and fails gracefully at domain edges. Use linear for CONUS (full coverage
    expected), nearest for Alaska edges (ocean tiles would otherwise NaN).
    """
    print(f"  Parsing {grib_path}...")
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    u = ds["u10"].values
    v = ds["v10"].values
    lat = ds.latitude.values
    lng = ds.longitude.values

    # For CONUS target (West longitudes, negative), normalize lng from 0-360 to -180..180.
    # For Alaska target, keep as 0-360 so Aleutian wrap-around stays contiguous.
    if region_name == "conus":
        lng = np.where(lng > 180, lng - 360, lng)

    # Target mesh
    tlng_vec = np.arange(target["lo1"], target["lo1"] + target["nx"] * target["dx"], target["dx"])
    tlat_vec = np.arange(target["la1"], target["la1"] - target["ny"] * target["dy"], -target["dy"])
    tlng, tlat = np.meshgrid(tlng_vec, tlat_vec)

    src_pts = np.column_stack([lng.ravel(), lat.ravel()])
    method = "linear" if region_name == "conus" else "nearest"
    t0 = time.time()
    u_reg = griddata(src_pts, u.ravel(), (tlng, tlat), method=method, fill_value=0).astype(np.float32)
    v_reg = griddata(src_pts, v.ravel(), (tlng, tlat), method=method, fill_value=0).astype(np.float32)
    print(f"  Resampled ({method}) in {time.time()-t0:.1f}s → shape {u_reg.shape}")
    return u_reg, v_reg


def _write_cambecc_json(u, v, target, out_path):
    """
    Emit in cambecc/earth JSON format so FIRESTORM's existing particle
    renderer loads it with zero code changes.
    """
    header = {
        "parameterCategory": 2,
        "lo1": target["lo1"], "la1": target["la1"],
        "dx": target["dx"], "dy": target["dy"],
        "nx": target["nx"], "ny": target["ny"],
    }
    records = [
        {"header": {**header, "parameterNumber": 2}, "data": u.ravel().tolist()},
        {"header": {**header, "parameterNumber": 3}, "data": v.ravel().tolist()},
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"  Wrote {os.path.getsize(out_path)/1024/1024:.1f} MB → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────

def run_region(region_name, fetch_fn, hours_per_cycle):
    """
    Try the latest run; if NOMADS doesn't have it yet (common for Alaska
    which publishes every 3h), back off to the previous run. Max 2 retries.
    """
    for backoff_cycles in range(3):
        date, run = _latest_run(hours_per_cycle)
        if backoff_cycles > 0:
            # Step back N cycles
            target_time = dt.datetime.strptime(f"{date}{run}", "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
            target_time -= dt.timedelta(hours=hours_per_cycle * backoff_cycles)
            date = target_time.strftime("%Y%m%d")
            run = target_time.strftime("%H")

        print(f"\n[{region_name.upper()}] attempting run {date} {run}z (backoff={backoff_cycles})")
        grib_path = os.path.join(DATA_DIR, f"_tmp_{region_name}.grib2")
        try:
            fetch_fn(date, run, grib_path)
            u, v = _resample_lambert_to_regular(grib_path, TARGET[region_name], region_name)
            out_path = os.path.join(DATA_DIR, f"current-wind-10m-hrrr-{region_name}.json")
            _write_cambecc_json(u, v, TARGET[region_name], out_path)
            os.remove(grib_path)   # keep repo from committing GRIB2 binaries
            return {
                "region": region_name, "ok": True,
                "run": f"{date}{run}z",
                "shape": [TARGET[region_name]["ny"], TARGET[region_name]["nx"]],
                "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 2),
            }
        except (HTTPError, URLError) as e:
            print(f"  {region_name} {date}{run}z HTTP error: {e}")
            if os.path.exists(grib_path):
                os.remove(grib_path)
        except Exception as e:
            print(f"  {region_name} {date}{run}z failed: {type(e).__name__}: {e}")
            if os.path.exists(grib_path):
                os.remove(grib_path)
    return {"region": region_name, "ok": False}


def main():
    print("=" * 60)
    print("FIRESTORM HRRR Pipeline · CONUS + Alaska")
    print(f"Start: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    print("=" * 60)

    results = []
    # CONUS runs hourly — recent enough run is always recent
    results.append(run_region("conus",  _fetch_conus,  hours_per_cycle=1))
    # Alaska runs every 3 hours
    results.append(run_region("alaska", _fetch_alaska, hours_per_cycle=3))

    meta = {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "schema_version": 1,
        "regions": {r["region"]: r for r in results},
        "source": "NOAA NOMADS HRRR (via cfgrib + scipy griddata resample)",
        "pipeline": "https://github.com/Deasus/firestorm-hrrr-data",
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[META] wrote data/meta.json")

    any_failed = any(not r["ok"] for r in results)
    if any_failed:
        print(f"\n⚠ Some regions failed — see logs above")
        # Don't exit non-zero; partial success is valid and the frontend
        # falls back to GFS anyway. A hard exit would block the green region
        # from publishing.
    else:
        print(f"\n✓ All regions published")


if __name__ == "__main__":
    main()
