"""
heatwave_tools.py -- core functions to detect heatwaves or coldwaves in daily temperature.

run_detection.py calls these functions in this order:

  load_temperature, get_land_mask  read the file: domain, daily maximum, degC; land mask
  compute_climatology / load_climatology   threshold for every calendar day
  make_anomaly                     anomaly = temperature - threshold; points above (below) it
  coherence_filter                 keep only points whose neighbourhood is mostly hot (cold)
  track_obj                        follow the regions in time -> events
  filter_events                    keep long enough (and large enough) events
  summarise_events                 one table row per event
  merge_into_episodes              events close in time and space -> episodes
  build_event_mask, save_results   the tracked mask and the files

The ConTrack workflow (section 6) works with all time steps of the day (hourly, 3-hourly,
6-hourly, ...):

  load_timesteps                   read the file, keep every time step
  extreme_anomaly                  anomaly to a transient climatology, only where it is extreme
  clean_extreme_anomaly            land mask and removal of small objects
  mask_year_boundaries             make sure that no event continues from one year to the next
  run_contrack                     ConTrack: flag field and life cycle table
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.ndimage import binary_closing, convolve

EVENT_SIGN = {"heatwave": +1, "coldwave": -1}
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)     # structure of the closing
ALL8 = np.ones((3, 3), dtype=bool)                                   # points touching diagonally are connected


# =====================================================================
# 1. Reading the data
# =====================================================================
def to_daily_max(t):
    """Daily maximum of the time steps of each day. Daily data are returned unchanged."""
    day = t.time.dt.floor("D")
    if not day.to_series().duplicated().any():
        return t
    return t.groupby(day).max().rename(floor="time")


def load_temperature(file, variable, lat_range=None, lon_range=None, years=None):
    """
    Read the temperature and return it as daily maximum in degC, dims (time, lat, lon).
    Longitudes are converted to -180...180 and latitudes sorted from south to north.
    Years are read one by one, so that large files with hourly data stay manageable.
    """
    ds = xr.open_dataset(file)
    ds = ds.rename({old: new for old, new in
                    {"latitude": "lat", "longitude": "lon", "valid_time": "time"}.items()
                    if old in ds.dims})
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Variables in the file: {list(ds.data_vars)}")
    ds = ds.sortby("lat")
    if lat_range is not None:
        ds = ds.sel(lat=slice(*lat_range))
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    if lon_range is not None:
        ds = ds.sel(lon=slice(*lon_range))

    t = ds[variable]
    all_years = np.unique(t.time.dt.year.values)
    if years is not None:
        all_years = [y for y in all_years if years[0] <= y <= years[1]]
    t = xr.concat([to_daily_max(t.sel(time=str(y)).load()) for y in all_years], dim="time")

    if float(t.max()) > 200:                          # Kelvin -> degC
        t = t - 273.15
    t = t.sel(time=~((t.time.dt.month == 2) & (t.time.dt.day == 29)))   # 365-day calendar
    t.name = "temperature"
    t.attrs = {"units": "degC", "long_name": "daily maximum temperature"}
    return t


def get_land_mask(t, use_mask=True, mask_file=None, mask_variable=None):
    """
    Grid points where events may be detected.
    use_mask = False : all grid points.
    use_mask = True, no mask_file : points that have data in the temperature file (ocean = NaN).
    use_mask = True, with mask_file : points where mask_variable has data (first time step if it has
    a time axis). The mask must be on the same grid as the temperature.
    """
    if not use_mask:
        return xr.ones_like(t.isel(time=0), dtype=bool).drop_vars("time")
    if mask_file is None:
        return t.isel(time=0).notnull().drop_vars("time")
    ds = xr.open_dataset(mask_file)
    ds = ds.rename({old: new for old, new in {"latitude": "lat", "longitude": "lon"}.items() if old in ds.dims})
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon").sortby("lat")
    mask = ds[mask_variable]
    if "time" in mask.dims:
        mask = mask.isel(time=0)
    try:
        mask = mask.sel(lat=t.lat, lon=t.lon, method="nearest", tolerance=0.01)
    except KeyError:
        raise ValueError("The land mask file is not on the same grid as the temperature data.")
    return xr.DataArray(mask.notnull().values, coords={"lat": t.lat, "lon": t.lon}, dims=("lat", "lon"))


# =====================================================================
# 2. Climatology (threshold for every calendar day)
# =====================================================================
def circular_day_distance(doy, target, n=365):
    d = np.abs(doy - target)
    return np.minimum(d, n - d)


def season_days_of_year(months):
    """Days of the year that fall in the given months (normal year and leap year)."""
    days = set()
    for year in (2001, 2004):
        d = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        days |= set(d[d.month.isin(months)].dayofyear)
    return np.array(sorted(days))


def compute_climatology(t, baseline_start, baseline_end, percentile, window_days, months):
    """
    Threshold for every day of the year in `months`: the `percentile` of the temperature
    within +/- window_days of that day, between baseline_start and baseline_end
    (dates, e.g. "1992-06-01"), at every grid point.
    """
    t_base = t.sel(time=slice(baseline_start, baseline_end))
    if t_base.sizes["time"] == 0:
        raise ValueError(f"No data between {baseline_start} and {baseline_end}. "
                         f"The data cover {str(t.time.values[0])[:10]} to {str(t.time.values[-1])[:10]}.")
    days = season_days_of_year(months)
    doy = t_base.time.dt.dayofyear.values
    values = t_base.values
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)      # all-NaN (ocean) points
        for n, day in enumerate(days):
            keep = circular_day_distance(doy, day) <= window_days
            out.append(np.nanpercentile(values[keep], 100 * percentile, axis=0))
            if (n + 1) % 20 == 0 or n + 1 == len(days):
                print(f"  climatology: {n + 1} of {len(days)} days of the year", flush=True)
    clim = xr.DataArray(np.stack(out).astype("float32"), name="threshold",
                        coords={"dayofyear": days, "lat": t.lat, "lon": t.lon},
                        dims=("dayofyear", "lat", "lon"))
    clim.attrs = {"percentile": float(percentile), "window_days": int(window_days),
                  "baseline": f"{baseline_start} to {baseline_end}",
                  "months": ",".join(str(m) for m in months), "units": "degC"}
    return clim


def save_climatology(clim, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    clim.to_dataset().to_netcdf(file, encoding={"threshold": {"zlib": True, "complevel": 5}})
    print(f"  climatology written to {file} ({Path(file).stat().st_size / 1e6:.1f} MB)")


def load_climatology(file, percentile, window_days):
    """Read a climatology file and check that it was made with the same percentile and window."""
    clim = xr.open_dataset(file)["threshold"].load()
    if abs(clim.attrs["percentile"] - percentile) > 1e-9 or clim.attrs["window_days"] != window_days:
        raise ValueError(
            f"{file} was computed with percentile={clim.attrs['percentile']} and "
            f"window_days={clim.attrs['window_days']}, but the settings are percentile={percentile} and "
            f"window_days={window_days}. Change the settings, or set COMPUTE_CLIMATOLOGY = True.")
    print(f"  climatology read from {file} (baseline {clim.attrs['baseline']})")
    return clim


# =====================================================================
# 3. Anomaly, hot (cold) points and the coherence filter
# =====================================================================
def make_anomaly(t, clim, months, event_type, land):
    """
    anomaly = temperature - threshold (for the days in `months`).
    hot     = land points where the anomaly is above 0 (heatwave) or below 0 (coldwave).
    """
    if not (t.sizes["lat"] == clim.sizes["lat"] and t.sizes["lon"] == clim.sizes["lon"]
            and np.allclose(t.lat, clim.lat) and np.allclose(t.lon, clim.lon)):
        raise ValueError("The climatology and the temperature data are on different grids "
                         "(different domain or resolution). Compute the climatology again.")
    sign = EVENT_SIGN[event_type]
    t_season = t.sel(time=t.time.dt.month.isin(months))
    doys = t_season.time.dt.dayofyear.values
    missing = set(doys) - set(clim.dayofyear.values.tolist())
    if missing:
        raise ValueError(f"The climatology has no threshold for days of the year {sorted(missing)[:5]}... "
                         "Check SEASON_MONTHS.")
    thr = clim.sel(dayofyear=xr.DataArray(doys, dims="time"))
    thr = thr.drop_vars("dayofyear").assign_coords(time=t_season.time)
    anomaly = (t_season - thr).rename("anomaly")
    anomaly.attrs = {"units": "degC", "long_name": "temperature minus threshold"}
    hot = (sign * anomaly > 0) & land
    return t_season, anomaly, hot


def is_periodic(lon):
    """True if the longitudes go all the way round the globe."""
    dlon = np.abs(np.diff(lon)).mean()
    return bool(lon.max() - lon.min() + dlon >= 360 - 1e-6)


def coherence_filter(hot, land, window_deg, fraction):
    """
    Keep a hot point only if at least `fraction` of the land in the window around it (about
    window_deg wide, weighted with cos(latitude)) is hot as well. This removes isolated hot points.
    fraction = 0 switches the filter off.
    """
    if fraction <= 0:
        return hot
    dlat = float(abs(hot.lat[1] - hot.lat[0]))
    dlon = float(abs(hot.lon[1] - hot.lon[0]))
    nlat = max(1, int(round(window_deg / dlat)))
    nlon = max(1, int(round(window_deg / dlon)))
    nlat += (nlat % 2 == 0)                                  # odd window sizes
    nlon += (nlon % 2 == 0)
    kernel = np.ones((nlat, nlon))
    weights = np.cos(np.deg2rad(hot.lat.values))[:, None] * np.ones((hot.sizes["lat"], hot.sizes["lon"]))
    pad = nlon // 2 if is_periodic(hot.lon.values) else 0

    def window_sum(a):
        """Sum over the window around every point; wraps around the globe in longitude if periodic."""
        if pad > 0:
            a = np.pad(a, ((0, 0), (pad, pad)), mode="wrap")
            return convolve(a, kernel, mode="constant", cval=0.0)[:, pad:-pad]
        return convolve(a, kernel, mode="constant", cval=0.0)

    land_np = land.values
    denominator = window_sum(land_np.astype(float) * weights)
    hot_np = hot.values.astype(float)
    out = np.zeros(hot_np.shape, dtype=bool)
    for i in range(hot_np.shape[0]):
        numerator = window_sum(hot_np[i] * weights)
        share = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        out[i] = (share >= fraction) & land_np
    return xr.DataArray(out, coords=hot.coords, dims=hot.dims, name="coherent_mask")


# =====================================================================
# 4. Tracking
# =====================================================================
def split_into_seasons(times):
    """Blocks of consecutive days. Events never continue from one block to the next."""
    breaks = np.where(np.diff(times.values) != np.timedelta64(1, "D"))[0] + 1
    return np.split(np.arange(len(times)), breaks)


def group_cells(flat_labels):
    """{label: flat grid indices} for all labels above 0 in a flat label array."""
    cells = np.flatnonzero(flat_labels > 0)
    if cells.size == 0:
        return {}
    labels = flat_labels[cells]
    order = np.argsort(labels, kind="stable")
    cells, labels = cells[order], labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    ends = np.r_[starts[1:], labels.size]
    return {int(labels[s]): cells[s:e] for s, e in zip(starts, ends)}


def label_regions(mask, land, periodic, closing_iterations):
    """Close small gaps (cross-shaped structure), then label the regions of one day."""
    if closing_iterations > 0:
        if periodic:
            k = closing_iterations
            padded = np.pad(mask, ((0, 0), (k, k)), mode="wrap")
            closed = binary_closing(padded, structure=CROSS, iterations=k)[:, k:-k]
        else:
            closed = binary_closing(mask, structure=CROSS, iterations=closing_iterations)
        mask = closed & land
    labels, n = ndimage.label(mask, structure=ALL8)
    if periodic and n > 1:                                   # join regions that touch across the seam
        parent = np.arange(n + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        left, right = labels[:, 0], labels[:, -1]
        nrow = mask.shape[0]
        for r in range(nrow):
            if left[r] == 0:
                continue
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < nrow and right[rr] > 0:
                    a, b = find(left[r]), find(right[rr])
                    if a != b:
                        parent[max(a, b)] = min(a, b)
        lookup = np.array([find(i) for i in range(n + 1)])
        _, inverse = np.unique(lookup[labels], return_inverse=True)
        labels = inverse.reshape(labels.shape)
    return labels


def track_obj(mask, land, overlap_threshold, closing_iterations=1):
    """
    Follow the regions from day to day (the tracking of the paper). On every day the regions are
    labelled. An event continues into the region with the largest overlap fraction (overlap divided
    by the smaller of the two areas) if that fraction is larger than overlap_threshold. Events choose
    in the order they were created. Regions that continue no event start a new event.
    Returns a list of events {"event_id", "t_idx", "cells"}; event_id counts the events in the order
    they were created (it is not renumbered after filtering).
    """
    mask_np, land_np = mask.values, land.values
    times = pd.to_datetime(mask.time.values)
    periodic = is_periodic(mask.lon.values)
    events, next_id = [], 1
    for block in split_into_seasons(times):
        active = []
        for i in block:
            labels = label_regions(mask_np[i], land_np, periodic, closing_iterations)
            flat = labels.ravel()
            regions = group_cells(flat)
            n = int(labels.max())
            sizes = np.bincount(flat, minlength=n + 1)
            used, still_active = set(), []
            for ev in active:
                prev = ev["cells"][-1]
                overlap = np.bincount(flat[prev], minlength=n + 1).astype(float)
                share = overlap / np.minimum(prev.size, sizes)
                share[0] = 0.0
                for k in used:
                    share[k] = 0.0
                k = int(share.argmax())
                if k > 0 and share[k] > overlap_threshold:
                    used.add(k)
                    ev["t_idx"].append(i)
                    ev["cells"].append(regions[k])
                    still_active.append(ev)
                else:
                    events.append(ev)
            for k, cells in regions.items():
                if k not in used:
                    still_active.append({"event_id": next_id, "t_idx": [i], "cells": [cells]})
                    next_id += 1
            active = still_active
        events += active
    return events



# =====================================================================
# 5. Events, episodes and the tracked mask
# =====================================================================
def filter_events(events, min_duration_days, min_gridcells):
    """Keep the events that last long enough and are large enough (more than min_gridcells grid
    points on their largest day). The event ids are not changed."""
    kept = [ev for ev in events
            if len(ev["t_idx"]) >= min_duration_days
            and max(c.size for c in ev["cells"]) > min_gridcells]
    return sorted(kept, key=lambda ev: (ev["t_idx"][0], ev["event_id"]))


def _lon_box_and_centre(lon_values):
    """Bounding box and centre of longitudes. Correct also for regions across the dateline,
    where lon_min > lon_max (the box goes eastward from lon_min to lon_max)."""
    if lon_values.max() - lon_values.min() > 180:
        shifted = lon_values % 360
        wrap = lambda x: float((x + 180) % 360 - 180)
        return wrap(shifted.min()), wrap(shifted.max()), wrap(shifted.mean())
    return float(lon_values.min()), float(lon_values.max()), float(lon_values.mean())


def summarise_events(events, t_season, anomaly, event_type):
    """
    One row per event. Bounding box and centroid use all days of the event. The peak day is the
    day with the largest area; the peak temperature and anomaly are the extremes in the region
    of that day (maximum for a heatwave, minimum for a coldwave).
    """
    sign = EVENT_SIGN[event_type]
    pick = np.nanmax if sign > 0 else np.nanmin
    ext = "max" if sign > 0 else "min"
    times = pd.to_datetime(t_season.time.values)
    t_np, a_np = t_season.values, anomaly.values
    lat, lon = t_season.lat.values, t_season.lon.values
    shape = t_np.shape[1:]
    rows = []
    for ev in events:
        areas = [c.size for c in ev["cells"]]
        j = int(np.argmax(areas))
        i, peak_cells = ev["t_idx"][j], ev["cells"][j]
        ilat, ilon = np.unravel_index(np.unique(np.concatenate(ev["cells"])), shape)
        lon_min, lon_max, centroid_lon = _lon_box_and_centre(lon[ilon])
        rows.append({
            "event_id": ev["event_id"],
            "start_date": times[ev["t_idx"][0]], "end_date": times[ev["t_idx"][-1]],
            "duration_days": len(ev["t_idx"]), "peak_day": times[i], "max_gridcells": areas[j],
            "lat_min": float(lat[ilat].min()), "lat_max": float(lat[ilat].max()),
            "lon_min": lon_min, "lon_max": lon_max,
            "centroid_lat": float(lat[ilat].mean()), "centroid_lon": centroid_lon,
            f"peak_day_{ext}_temp": float(pick(t_np[i].ravel()[peak_cells])),
            f"peak_day_{ext}_temp_anom": float(pick(a_np[i].ravel()[peak_cells])),
        })
    columns = ["event_id", "start_date", "end_date", "duration_days", "peak_day", "max_gridcells",
               "lat_min", "lat_max", "lon_min", "lon_max", "centroid_lat", "centroid_lon",
               f"peak_day_{ext}_temp", f"peak_day_{ext}_temp_anom"]
    df = pd.DataFrame(rows, columns=columns)
    for column in ("start_date", "end_date", "peak_day"):
        df[column] = pd.to_datetime(df[column])               # also correct if there are no events
    return df.sort_values(["start_date", "event_id"]).reset_index(drop=True)


def _lon_ranges_overlap(min1, max1, min2, max2):
    if max1 < min1:                                           # region across the dateline
        max1 += 360
    if max2 < min2:
        max2 += 360
    return any(not (max1 < min2 + s or max2 + s < min1) for s in (-360, 0, 360))


def _should_merge(a, b, gap_days, centroid_deg):
    """Same year, small gap in time, and boxes that overlap or centroids that are close."""
    same_year = a["start_date"].year == b["start_date"].year
    time_ok = (b["start_date"] - a["end_date"]).days <= gap_days
    lat_overlap = not (a["lat_max"] < b["lat_min"] or b["lat_max"] < a["lat_min"])
    box_overlap = lat_overlap and _lon_ranges_overlap(a["lon_min"], a["lon_max"], b["lon_min"], b["lon_max"])
    dlat = a["centroid_lat"] - b["centroid_lat"]
    dlon = (a["centroid_lon"] - b["centroid_lon"] + 180) % 360 - 180
    centroids_close = np.sqrt(dlat ** 2 + dlon ** 2) <= centroid_deg
    return same_year and time_ok and (box_overlap or centroids_close)


def merge_into_episodes(events_df, gap_days, centroid_deg, event_type):
    """
    Events become one episode if they are in the same year, less than gap_days apart in time
    and their boxes overlap or their centroids are less than centroid_deg apart. Events are taken
    in order of start date and compared with the episode built so far. The peak of the episode
    is the most extreme of its events.
    """
    sign = EVENT_SIGN[event_type]
    ext = "max" if sign > 0 else "min"
    temp, anom = f"peak_day_{ext}_temp", f"peak_day_{ext}_temp_anom"
    df = events_df.sort_values(["start_date", "end_date"]).reset_index(drop=True)
    episodes, current = [], None
    for _, row in df.iterrows():
        row = row.copy()
        if current is not None and _should_merge(current, row, gap_days, centroid_deg):
            current["end_date"] = max(current["end_date"], row["end_date"])
            current["start_date"] = min(current["start_date"], row["start_date"])
            current["duration_days"] = (current["end_date"] - current["start_date"]).days + 1
            for column, pick in (("lat_min", min), ("lat_max", max), ("lon_min", min), ("lon_max", max),
                                 ("max_gridcells", max)):
                current[column] = pick(current[column], row[column])
            if sign * row[temp] > sign * current[temp]:
                current[temp] = row[temp]
            if sign * row[anom] > sign * current[anom]:
                current[anom] = row[anom]
                current["peak_day"] = row["peak_day"]
            current["centroid_lat"] = 0.5 * (current["lat_min"] + current["lat_max"])
            current["centroid_lon"] = 0.5 * (current["lon_min"] + current["lon_max"])
            current["source_event_ids"].append(int(row["event_id"]))
        else:
            if current is not None:
                episodes.append(current)
            current = row
            current["source_event_ids"] = [int(row["event_id"])]
    if current is not None:
        episodes.append(current)
    out = pd.DataFrame(episodes, columns=list(events_df.columns) + ["source_event_ids"])
    for column in ("start_date", "end_date", "peak_day"):
        out[column] = pd.to_datetime(out[column])
    return out.sort_values("start_date").reset_index(drop=True)


def episodes_without_merging(events_df):
    """Every event is its own episode."""
    out = events_df.copy()
    out["source_event_ids"] = [[int(i)] for i in out["event_id"]]
    return out


def build_event_mask(events, t_season, event_type):
    """
    The tracked mask: for every day and grid point the id of the event (as in events.csv),
    0 where there is no event. Also a 0/1 version, named heatwave_mask or coldwave_mask.
    """
    ids = np.zeros(t_season.shape, dtype=np.int32)
    flat = ids.reshape(ids.shape[0], -1)                     # a view: writing into it fills ids
    for ev in events:
        for i, cells in zip(ev["t_idx"], ev["cells"]):
            flat[i, cells] = ev["event_id"]
    coords = {"time": t_season.time, "lat": t_season.lat, "lon": t_season.lon}
    return xr.Dataset({
        "event_id": (("time", "lat", "lon"), ids,
                     {"long_name": "id of the tracked event (see events.csv), 0 = no event"}),
        f"{event_type}_mask": (("time", "lat", "lon"), (ids > 0).astype(np.int8),
                               {"long_name": f"1 where a tracked {event_type} is present"}),
    }, coords=coords)


def save_results(out_dir, events_df, episodes_df, mask_ds=None, anomaly=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out / "events.csv", index=False)
    episodes_csv = episodes_df.copy()
    episodes_csv["source_event_ids"] = episodes_csv["source_event_ids"].apply(lambda ids: ";".join(map(str, ids)))
    episodes_csv.to_csv(out / "episodes.csv", index=False)
    periods = {}
    for _, r in episodes_df.iterrows():
        periods.setdefault(str(r.start_date.year), []).append(
            [r.start_date.strftime("%Y-%m-%d"), r.end_date.strftime("%Y-%m-%d")])
    json.dump(periods, open(out / "episode_periods.json", "w"), indent=2)
    if mask_ds is not None:
        mask_ds.to_netcdf(out / "tracked_mask.nc",
                          encoding={v: {"zlib": True, "complevel": 5} for v in mask_ds.data_vars})
    if anomaly is not None:
        anomaly.to_dataset().to_netcdf(out / "anomaly.nc", encoding={"anomaly": {"zlib": True, "complevel": 5}})
    print(f"Results written to {out}/")


# =====================================================================
# 6. The ConTrack workflow
# =====================================================================
def load_timesteps(file, variable, lat_range=None, lon_range=None, months=None, years=None):
    """Read the temperature with all its time steps (hourly, 3-hourly, 6-hourly, ...)."""
    ds = xr.open_dataset(file)
    ds = ds.rename({old: new for old, new in
                    {"latitude": "lat", "longitude": "lon", "valid_time": "time"}.items()
                    if old in ds.dims})
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Variables in the file: {list(ds.data_vars)}")
    ds = ds.sortby("lat")
    if lat_range is not None:
        ds = ds.sel(lat=slice(*lat_range))
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    if lon_range is not None:
        ds = ds.sel(lon=slice(*lon_range))
    t = ds[variable]
    if months is not None:
        t = t.sel(time=t.time.dt.month.isin(months))
    if years is not None:
        year = t.time.dt.year
        t = t.sel(time=(year >= years[0]) & (year <= years[1]))
    return t.load()


def steps_per_day(t):
    """Number of time steps per day (1 for daily data, 4 for 6-hourly data, ...)."""
    return int(round(t.sizes["time"] / np.unique(t.time.dt.floor("D")).size))


def extreme_anomaly(t, event_type, half_window_years, percentile, window_days, smooth_days, smooth_min_days):
    """
    Anomaly of every time step, only where it is extreme (NaN elsewhere). Done for every time of
    the day separately:
      1. transient climatology: mean of the same time of day and day of the year over the years
         within +/- half_window_years (the year itself included);
      2. anomaly = temperature - this climatology;
      3. threshold for every day of the year: the percentile of the anomalies within +/- window_days;
      4. the threshold is smoothed with a running mean over smooth_days days of the year;
      5. extreme = anomaly above (heatwave) or below (coldwave) the threshold.
    """
    sign = EVENT_SIGN[event_type]
    years_all = t.time.dt.year.values
    hours_all = t.time.dt.hour.values
    doy_all = t.time.dt.dayofyear.values
    years, doys, hours = np.unique(years_all), np.unique(doy_all), np.unique(hours_all)
    year_pos, doy_pos = np.searchsorted(years, years_all), np.searchsorted(doys, doy_all)
    data = t.values
    grid = data.shape[1:]
    out = np.full(data.shape, np.nan, dtype=data.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)          # all-NaN (ocean) points
        for h in hours:
            idx = np.flatnonzero(hours_all == h)
            arr = np.full((len(years), len(doys)) + grid, np.nan, dtype=data.dtype)
            arr[year_pos[idx], doy_pos[idx]] = data[idx]
            anom = np.empty_like(arr)
            for yi, year in enumerate(years):
                anom[yi] = arr[yi] - np.nanmean(arr[np.abs(years - year) <= half_window_years], axis=0)
            threshold = np.empty((len(doys),) + grid, dtype=data.dtype)
            for di, day in enumerate(doys):
                window = np.abs(doys - day) <= window_days
                threshold[di] = np.nanpercentile(anom[:, window].reshape((-1,) + grid), 100 * percentile, axis=0)
            threshold = (xr.DataArray(threshold, dims=("dayofyear", "lat", "lon"))
                         .rolling(dayofyear=smooth_days, center=True, min_periods=smooth_min_days)
                         .mean().values)
            a = anom[year_pos[idx], doy_pos[idx]]
            extreme = sign * (a - threshold[doy_pos[idx]]) > 0
            out[idx] = np.where(extreme, a, np.nan)
            print(f"  hour {int(h):02d}: done", flush=True)
    return xr.DataArray(out, coords=t.coords, dims=t.dims, name="anomaly")


def remove_small_objects(mask, min_size):
    """Remove the regions (points touching along an edge) with fewer than min_size points."""
    labels, _ = ndimage.label(mask, structure=CROSS)
    keep = np.bincount(labels.ravel()) >= min_size
    keep[0] = False
    return keep[labels]


def clean_extreme_anomaly(extreme, land, min_size):
    """Keep the extreme points on land that belong to a region of at least min_size points."""
    valid = extreme.notnull().values & land.values
    clean = np.zeros(valid.shape, dtype=bool)
    for i in range(valid.shape[0]):
        clean[i] = remove_small_objects(valid[i], min_size)
    return extreme.where(xr.DataArray(clean, coords=extreme.coords, dims=extreme.dims))


def mask_year_boundaries(x):
    """Set the last time step of every year and the first of the next to NaN, so that no event
    continues from one summer to the next."""
    values = x.values.copy()
    for idx in np.where(np.diff(x.time.dt.year.values) != 0)[0]:
        values[idx] = np.nan
        values[idx + 1] = np.nan
    return x.copy(data=values)


def run_contrack(anomaly, event_type, threshold, overlap, persistence_steps, twosided=True):
    """
    ConTrack (https://github.com/steidani/ConTrack) on the extreme anomaly: regions above
    +threshold for a heatwave, below -threshold for a coldwave, tracked in time.
    Returns the flag field (id of the event at every time step and grid point, 0 = none) and the
    life cycle table of ConTrack.
    """
    try:
        from contrack import contrack
    except ImportError:
        raise ImportError("ConTrack is not installed. See the README.")
    sign = EVENT_SIGN[event_type]
    tracker = contrack()
    tracker.read_xarray(anomaly.to_dataset(name="anomaly"))
    tracker.set_up(force=True, write=False)
    tracker.run_contrack(variable="anomaly", threshold=sign * threshold, gorl=">" if sign > 0 else "<",
                         overlap=overlap, persistence=persistence_steps, twosided=twosided)
    lifecycle = tracker.run_lifecycle(flag="flag", variable="anomaly")
    return tracker["flag"], lifecycle