"""
FIRESTORM HRRR Pipeline — CONUS + Alaska (Wind + Thermo)
========================================================

Pulls the latest available HRRR fields from NOAA NOMADS:
  • UGRD/VGRD at 10 m  → cambecc/earth wind JSON (existing wind layer)
  • TMP at 2 m AGL     → temperature (Kelvin) JSON (NEW v2 — for HDW)
  • RH at 2 m AGL      → relative humidity (%) JSON (NEW v2 — for HDW)

Both grids share the same Lambert resample → regular lat/lng pipeline
so the wind file stays byte-compatible. Frontend reads thermo file
through a parallel sampler `_thermoAt(lat,lng)` which mirrors `_windAt`.

CONUS:  fetched via NOMADS filter CGI (server-side subset).
Alaska: fetched via GRIB2 byte-range from the full surface file.

Refresh: hourly (CONUS) / every 3h (Alaska) via GHA cron. Frontend
treats missing/stale thermo as "fall back to BI-only HDW proxy."

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
UA = "FIRESTORM-HRRR-Pipeline/2.0 (github.com/Deasus/firestorm-hrrr-data)"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Target regular grids (cambecc/earth format) ─────────────────────
TARGET = {
    "conus": {
        "lo1": -127.0, "la1": 50.0,
        "dx": 0.1, "dy": 0.1,
        "nx": int((-64.0 - (-127.0)) / 0.1),   # 630
        "ny": int((50.0 - 22.0) / 0.1),         # 280
    },
    "alaska": {
        "lo1": 156.0, "la1": 78.0,
        "dx": 0.15, "dy": 0.15,
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


def _conus_wind_url(date, run):
    """NOMADS filter CGI — UGRD+VGRD at 10m."""
    return (
        f"{NOMADS_BASE}/cgi-bin/filter_hrrr_2d.pl"
        f"?dir=/hrrr.{date}/conus"
        f"&file=hrrr.t{run}z.wrfsfcf00.grib2"
        f"&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
    )


def _conus_thermo_url(date, run):
    """NOMADS filter CGI — TMP+RH at 2m AGL.
    Note: HRRR stores RH at 2 m AGL as RH; some products use SPFH instead.
    The filter CGI accepts either; RH is what we want for VPD."""
    return (
        f"{NOMADS_BASE}/cgi-bin/filter_hrrr_2d.pl"
        f"?dir=/hrrr.{date}/conus"
        f"&file=hrrr.t{run}z.wrfsfcf00.grib2"
        f"&var_TMP=on&var_RH=on&lev_2_m_above_ground=on"
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


def _fetch_conus_wind(date, run, dest_path):
    url = _conus_wind_url(date, run)
    print(f"  CONUS wind → {url}")
    body = _http_get(url, timeout=120)
    if len(body) < 100000:
        raise RuntimeError(f"CONUS wind fetch too small ({len(body)} bytes)")
    with open(dest_path, "wb") as f:
        f.write(body)
    print(f"  CONUS wind wrote {len(body)/1024/1024:.1f} MB → {dest_path}")


def _fetch_conus_thermo(date, run, dest_path):
    url = _conus_thermo_url(date, run)
    print(f"  CONUS thermo → {url}")
    body = _http_get(url, timeout=120)
    if len(body) < 100000:
        raise RuntimeError(f"CONUS thermo fetch too small ({len(body)} bytes)")
    with open(dest_path, "wb") as f:
        f.write(body)
    print(f"  CONUS thermo wrote {len(body)/1024/1024:.1f} MB → {dest_path}")


def _fetch_alaska_records(date, run, dest_path, want_records):
    """
    Generic Alaska byte-range fetch.
      want_records: list of (var, level) tuples to extract
    Walks the .idx, finds matching records, fetches the smallest range
    that covers all of them in one HTTP call.
    """
    idx = _http_get(_ak_idx_url(date, run), timeout=30).decode("utf-8", errors="replace")
    lines = [l for l in idx.strip().split("\n") if l]
    records = [l.split(":") for l in lines]
    matched_indices = []
    for i, r in enumerate(records):
        if len(r) < 5:
            continue
        for var, level in want_records:
            if r[3] == var and r[4] == level:
                matched_indices.append(i)
                break
    if len(matched_indices) < len(want_records):
        raise RuntimeError(f"HRRR-AK idx missing records: wanted {want_records}, "
                           f"found {len(matched_indices)} of {len(want_records)}")
    matched_indices.sort()
    start = int(records[matched_indices[0]][1])
    last = matched_indices[-1]
    if last + 1 < len(records):
        end = int(records[last + 1][1]) - 1
    else:
        end = start + 80 * 1024 * 1024  # last record — fetch generously
    size = end - start + 1
    print(f"  Alaska byte range: {start}-{end} ({size/1024/1024:.1f} MB) "
          f"covering records {matched_indices}")
    body = _http_get(_ak_data_url(date, run), timeout=120, byte_range=(start, end))
    with open(dest_path, "wb") as f:
        f.write(body)
    print(f"  Alaska wrote {len(body)/1024/1024:.1f} MB → {dest_path}")


def _fetch_alaska_wind(date, run, dest_path):
    _fetch_alaska_records(
        date, run, dest_path,
        [("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")],
    )


def _fetch_alaska_thermo(date, run, dest_path):
    _fetch_alaska_records(
        date, run, dest_path,
        [("TMP", "2 m above ground"), ("RH", "2 m above ground")],
    )


# ── GRIB2 parse + resample ───────────────────────────────────────────

def _open_filtered_dataset(grib_path, filter_keys):
    """
    Open a GRIB2 file with cfgrib, optionally filtering by keys
    (typeOfLevel, level). Returns the xr.Dataset.
    """
    backend_kwargs = {"filter_by_keys": filter_keys} if filter_keys else None
    return xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs=backend_kwargs,
    )


def _resample_to_regular(values_2d, src_lat_2d, src_lng_2d, target, region_name, method=None):
    """
    Common resample step — scatter source values onto the regular target grid.
    """
    if region_name == "conus":
        src_lng_2d = np.where(src_lng_2d > 180, src_lng_2d - 360, src_lng_2d)
    tlng_vec = np.arange(target["lo1"], target["lo1"] + target["nx"] * target["dx"], target["dx"])
    tlat_vec = np.arange(target["la1"], target["la1"] - target["ny"] * target["dy"], -target["dy"])
    tlng, tlat = np.meshgrid(tlng_vec, tlat_vec)
    src_pts = np.column_stack([src_lng_2d.ravel(), src_lat_2d.ravel()])
    if method is None:
        method = "linear" if region_name == "conus" else "nearest"
    return griddata(
        src_pts, values_2d.ravel(), (tlng, tlat),
        method=method, fill_value=np.nan,
    ).astype(np.float32)


def _resample_wind(grib_path, target, region_name):
    print(f"  [wind] parsing {grib_path}...")
    ds = _open_filtered_dataset(grib_path, {"typeOfLevel": "heightAboveGround", "level": 10})
    u = ds["u10"].values
    v = ds["v10"].values
    lat = ds.latitude.values
    lng = ds.longitude.values
    t0 = time.time()
    u_reg = _resample_to_regular(u, lat, lng, target, region_name)
    v_reg = _resample_to_regular(v, lat, lng, target, region_name)
    # Replace NaN at edges with 0 (renderer convention — no flow there)
    u_reg = np.nan_to_num(u_reg, nan=0.0)
    v_reg = np.nan_to_num(v_reg, nan=0.0)
    print(f"  [wind] resampled in {time.time()-t0:.1f}s → shape {u_reg.shape}")
    return u_reg, v_reg


def _resample_thermo(grib_path, target, region_name):
    print(f"  [thermo] parsing {grib_path}...")
    ds = _open_filtered_dataset(grib_path, {"typeOfLevel": "heightAboveGround", "level": 2})
    # cfgrib variable names: 't2m' for 2m temperature, 'r2' for 2m RH (some
    # builds use 'rh2m'). Probe both.
    t_var = None
    rh_var = None
    for cand in ("t2m", "t", "TMP_2maboveground"):
        if cand in ds:
            t_var = cand; break
    for cand in ("r2", "rh2m", "r", "RH_2maboveground"):
        if cand in ds:
            rh_var = cand; break
    if t_var is None or rh_var is None:
        avail = list(ds.data_vars)
        raise RuntimeError(f"thermo: cannot find T/RH (got vars: {avail})")
    t_k = ds[t_var].values   # Kelvin
    rh = ds[rh_var].values   # %
    lat = ds.latitude.values
    lng = ds.longitude.values
    t0 = time.time()
    t_reg = _resample_to_regular(t_k, lat, lng, target, region_name)
    rh_reg = _resample_to_regular(rh, lat, lng, target, region_name)
    # NaN at edges — keep NaN as sentinel (frontend won't compute HDW there)
    print(f"  [thermo] resampled in {time.time()-t0:.1f}s → T shape {t_reg.shape}, RH shape {rh_reg.shape}")
    return t_reg, rh_reg


# ── Output writers ───────────────────────────────────────────────────

def _write_wind_json(u, v, target, out_path):
    """cambecc/earth wind JSON — UGRD param=2, VGRD param=3."""
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
    print(f"  wrote wind {os.path.getsize(out_path)/1024/1024:.1f} MB → {out_path}")


def _write_thermo_json(t_k, rh, target, out_path):
    """
    Thermo JSON — same envelope as wind for renderer compatibility.
      record[0]: TMP at 2m (K), parameterCategory=0, parameterNumber=0
      record[1]: RH  at 2m (%), parameterCategory=1, parameterNumber=1
    NaN values become null (sparse JSON to keep size manageable).
    """
    def _serialize(arr):
        # Convert NaN → None so JSON encoder writes null
        out = arr.ravel().tolist()
        for i, v in enumerate(out):
            if v != v:  # NaN check
                out[i] = None
        return out

    header_geom = {
        "lo1": target["lo1"], "la1": target["la1"],
        "dx": target["dx"], "dy": target["dy"],
        "nx": target["nx"], "ny": target["ny"],
    }
    records = [
        {"header": {**header_geom, "parameterCategory": 0, "parameterNumber": 0,
                    "parameterUnit": "K", "parameterName": "Temperature"},
         "data": _serialize(t_k)},
        {"header": {**header_geom, "parameterCategory": 1, "parameterNumber": 1,
                    "parameterUnit": "%", "parameterName": "Relative Humidity"},
         "data": _serialize(rh)},
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"  wrote thermo {os.path.getsize(out_path)/1024/1024:.1f} MB → {out_path}")


# ── Region orchestration ─────────────────────────────────────────────

def run_region(region_name, fetch_wind_fn, fetch_thermo_fn, hours_per_cycle):
    """
    For a region, attempt: wind (required), thermo (best-effort).
    Returns dict with both statuses. Wind failure with backoff up to 3 cycles.
    Thermo follows wind's selected run — if wind succeeded at backoff=N,
    thermo also tries backoff=N first (run alignment matters for HDW).
    """
    wind_result = {"ok": False, "kind": "wind"}
    thermo_result = {"ok": False, "kind": "thermo"}
    chosen_date = None
    chosen_run = None

    # ── Wind: required, with backoff retry ────────────────────────────
    for backoff_cycles in range(3):
        date, run = _latest_run(hours_per_cycle)
        if backoff_cycles > 0:
            target_time = dt.datetime.strptime(f"{date}{run}", "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
            target_time -= dt.timedelta(hours=hours_per_cycle * backoff_cycles)
            date = target_time.strftime("%Y%m%d")
            run = target_time.strftime("%H")
        print(f"\n[{region_name.upper()} wind] attempting run {date} {run}z (backoff={backoff_cycles})")
        grib_path = os.path.join(DATA_DIR, f"_tmp_{region_name}_wind.grib2")
        try:
            fetch_wind_fn(date, run, grib_path)
            u, v = _resample_wind(grib_path, TARGET[region_name], region_name)
            out_path = os.path.join(DATA_DIR, f"current-wind-10m-hrrr-{region_name}.json")
            _write_wind_json(u, v, TARGET[region_name], out_path)
            os.remove(grib_path)
            chosen_date, chosen_run = date, run
            wind_result = {
                "ok": True, "kind": "wind",
                "run": f"{date}{run}z",
                "shape": [TARGET[region_name]["ny"], TARGET[region_name]["nx"]],
                "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 2),
            }
            break
        except (HTTPError, URLError) as e:
            print(f"  {region_name} wind {date}{run}z HTTP error: {e}")
        except Exception as e:
            print(f"  {region_name} wind {date}{run}z failed: {type(e).__name__}: {e}")
        finally:
            if os.path.exists(grib_path):
                os.remove(grib_path)

    # ── Thermo: best-effort, aligned to wind run ──────────────────────
    if chosen_date and chosen_run:
        print(f"\n[{region_name.upper()} thermo] attempting run {chosen_date} {chosen_run}z")
        grib_path = os.path.join(DATA_DIR, f"_tmp_{region_name}_thermo.grib2")
        try:
            fetch_thermo_fn(chosen_date, chosen_run, grib_path)
            t_k, rh = _resample_thermo(grib_path, TARGET[region_name], region_name)
            out_path = os.path.join(DATA_DIR, f"current-thermo-2m-hrrr-{region_name}.json")
            _write_thermo_json(t_k, rh, TARGET[region_name], out_path)
            os.remove(grib_path)
            thermo_result = {
                "ok": True, "kind": "thermo",
                "run": f"{chosen_date}{chosen_run}z",
                "shape": [TARGET[region_name]["ny"], TARGET[region_name]["nx"]],
                "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 2),
            }
        except (HTTPError, URLError) as e:
            print(f"  {region_name} thermo HTTP error: {e}")
        except Exception as e:
            print(f"  {region_name} thermo failed: {type(e).__name__}: {e}")
        finally:
            if os.path.exists(grib_path):
                os.remove(grib_path)
    else:
        print(f"\n[{region_name.upper()} thermo] skipped — wind selection failed")

    return {"region": region_name, "wind": wind_result, "thermo": thermo_result}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FIRESTORM HRRR Pipeline · CONUS + Alaska · v2 (wind + thermo)")
    print(f"Start: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    print("=" * 60)

    results = []
    results.append(run_region("conus",  _fetch_conus_wind,  _fetch_conus_thermo,  hours_per_cycle=1))
    results.append(run_region("alaska", _fetch_alaska_wind, _fetch_alaska_thermo, hours_per_cycle=3))

    meta = {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "schema_version": 2,
        "fields": {
            "wind":   {"vars": ["u10", "v10"], "level": "10 m above ground", "units": "m/s"},
            "thermo": {"vars": ["t2m", "r2"],  "level": "2 m above ground",  "units": ["K", "%"]},
        },
        "regions": {r["region"]: r for r in results},
        "source": "NOAA NOMADS HRRR (cfgrib + scipy griddata resample)",
        "pipeline": "https://github.com/Deasus/firestorm-hrrr-data",
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[META] wrote data/meta.json")

    any_wind_failed = any(not r["wind"]["ok"] for r in results)
    any_thermo_failed = any(not r["thermo"]["ok"] for r in results)
    if any_wind_failed:
        print(f"\n⚠ Some wind regions failed — see logs")
    if any_thermo_failed:
        print(f"\n⚠ Some thermo regions failed — frontend will fall back to BI-only HDW for affected regions")
    if not (any_wind_failed or any_thermo_failed):
        print(f"\n✓ All regions published (wind + thermo)")


if __name__ == "__main__":
    main()
