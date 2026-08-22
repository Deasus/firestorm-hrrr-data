# FIRESTORM HRRR Wind Data

High-resolution surface wind pipeline for the [FIRESTORM](https://deasus.github.io/Firestorm/)
wildfire intelligence platform. Pulls NOAA HRRR 10m UGRD/VGRD (wind east/north
components) from NOMADS, resamples the native Lambert Conformal grid onto a
regular lat/lng grid, and publishes cambecc/earth-format JSON consumed by the
FIRESTORM particle-animation layer.

## Outputs

| File | Region | Grid | Size | Source |
|---|---|---|---|---|
| `data/current-wind-10m-hrrr-conus.json` | CONUS | 0.1° · 630×280 | ~5 MB | HRRR CONUS · hourly |
| `data/current-wind-10m-hrrr-alaska.json` | Alaska + Aleutians | 0.15° · ~586×246 | ~4 MB | HRRR-AK · every 3h |
| `data/meta.json` | — | — | — | Run metadata, timestamps, status |

## Refresh schedule

GitHub Actions runs `*/15` (every hour at :15 UTC). HRRR CONUS publishes
hourly; HRRR-AK every 3h. A run where Alaska repeats the previous output
is not a failure — the frontend just reuses the cached JSON.

## Why not use the NOMADS filter CGI for Alaska?

The `filter_hrrr_2d.pl` endpoint rejects filenames matching
`hrrr.t..z.wrfsfcf...ak.grib2` (its regex assumes CONUS naming). For Alaska
we byte-range fetch the full surface GRIB2 using the `.idx` file NOAA
publishes alongside it — parse the index, find the UGRD+VGRD records at
"10 m above ground", and HTTP-Range GET just those ~3 MB.

## Known limitations

- **CONUS and Alaska only.** HRRR-Hawaii / HRRR-Puerto Rico don't exist at NOAA.
- **Resampling loses a touch of fine detail.** Native HRRR is ~3 km; we emit
  0.1° (~11 km) so JSON size stays manageable. Almost all visual fidelity
  preserved at fire-ops zooms.

## International expansion (roadmap)

Extending to regional HRRR-equivalents globally would require separate
pipelines for each domain (different GRIB conventions, variable names,
projections). Deferred. Current global fallback remains GFS 25km.

System overview: [firestorm-platform](https://github.com/Deasus/firestorm-platform). Main-app deployment status is tracked in the private application repo.
