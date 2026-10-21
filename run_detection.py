"""
detect heatwaves or coldwaves.

Two methods (METHOD):
  "track_obj"  daily maximum temperature, coherence filter, tracking, events and episodes
  "contrack"   all time steps of the day (hourly, 3-hourly, 6-hourly, ...), transient climatology, tracking with ConTrack
"""
from pathlib import Path

import numpy as np
import xarray as xr

from heatwave_tools import (build_event_mask, clean_extreme_anomaly, coherence_filter,
                            compute_climatology, episodes_without_merging, extreme_anomaly,
                            filter_events, get_land_mask, load_climatology, load_temperature,
                            load_timesteps, make_anomaly, mask_year_boundaries, merge_into_episodes,
                            run_contrack, save_climatology, save_results, steps_per_day,
                            summarise_events, track_obj)

# =========================================================================
#                              USER SETTINGS
# =========================================================================

# ---- 1. DATA (both methods) ---------------------------------------------
INPUT_FILE = "data/tmax_daily_sample.nc"   # NetCDF file with the temperature data 
VARIABLE = "2t"                            # name of the temperature variable in the file, e.g. "2t" or "t2m"
LAT_RANGE = (30.0, 60.0)                   # domain: (south, north) in degrees north; None = whole file
LON_RANGE = None                           # domain: (west, east) in degrees east, -180 to 180; None = all longitudes
DETECTION_YEARS = None                     # (first year, last year) to search for events; None = all years in the file
LAND_MASK = True                           # True : events only on land. False: all grid points
LAND_MASK_FILE = None                      # only if LAND_MASK = True: None = land is where the temperature file has data
LAND_MASK_VARIABLE = None                  #   (ocean = NaN). Or a file (same grid) and the name of its variable (points
                                           #   with data are land), e.g. "land_mask.nc" and "fbupinx"

# ---- 2. WHAT TO DETECT (both methods) -----------------------------------
EVENT_TYPE = "heatwave"                    # "heatwave": temperature above the threshold
                                           # "coldwave": temperature below the threshold
SEASON_MONTHS = [6, 7, 8]                  # months in which events are searched

# ---- 3. METHOD ----------------------------------------------------------
METHOD = "track_obj"                       # "track_obj" or "contrack". Settings: part 4 or part 5 below

OUTPUT_DIR = "output"                      # all results are written here

# ---- 4. SETTINGS FOR METHOD = "track_obj" -------------------------------
# Input: any file; several values per day (hourly, 3-hourly, ...) are reduced to the daily maximum.
OBJ_PERCENTILE = 0.95                      # threshold: heatwave 0.95 (hottest 5 % of days), coldwave 0.05
OBJ_WINDOW_DAYS = 8                        # the percentile of a calendar day uses the days within +/- this many days
OBJ_COMPUTE_CLIMATOLOGY = False            # True : compute the thresholds now (slow, needs many years of data)
                                           # False: read the thresholds from OBJ_CLIMATOLOGY_FILE
OBJ_CLIMATOLOGY_FILE = "data/climatology_p95.nc"   # written if True, read if False
OBJ_BASELINE_FILE = None                   # only if True: file used to compute the thresholds; None = INPUT_FILE
OBJ_BASELINE_START = "1992-06-01"          # only if True: period used to compute the thresholds
OBJ_BASELINE_END = "2025-08-31"
OBJ_COHERENCE_WINDOW_DEG = 1.5             # a hot point is kept only if enough of the land within about this many
OBJ_COHERENCE_FRACTION = 0.70              # degrees around it is hot too (this share). 0 = filter off
OBJ_OVERLAP = 0.60                         # overlap needed to continue an event from one day to the next
OBJ_MIN_DURATION_DAYS = 5                  # events must last at least this many days
OBJ_MIN_GRIDCELLS = 70                      # events must cover MORE than this many grid points on their largest day
OBJ_CLOSING_ITERATIONS = 1                 # closing of small gaps in the daily mask (0 = off)
OBJ_MERGE_EVENTS = True                    # True : events of the same year that are close in time and space become
                                           #        one episode. False: every event is its own episode
OBJ_MERGE_GAP_DAYS = 2                     # ... start less than this many days after the end of the previous event
OBJ_MERGE_CENTROID_DEG = 8.0               # ... and their boxes overlap or their centroids are this close (degrees)
                                           # (The merging compares bounding boxes. On a large domain use False.)
OBJ_SAVE_MASK = True                       # also write tracked_mask.nc (event id at every day and grid point)
OBJ_SAVE_ANOMALY = False                   # also write anomaly.nc (large for big domains)

# ---- 5. SETTINGS FOR METHOD = "contrack" --------------------------------
# Input: the file with all its time steps (hourly, 3-hourly, 6-hourly, ...). Needs ConTrack installed.
CT_HALF_WINDOW_YEARS = 4                   # transient climatology: mean over the years within +/- this many years
CT_PERCENTILE = 0.90                       # threshold of the anomalies: heatwave 0.90, coldwave 0.10
CT_WINDOW_DAYS = 10                        # the percentile of a day of the year uses the days within +/- this many days
CT_SMOOTHING_DAYS = 21                     # the threshold is smoothed with a running mean over this many days
CT_SMOOTHING_MIN_DAYS = 11                 # (at least this many days needed, so the ends of the season are kept)
CT_MIN_OBJECT_SIZE = 5                     # regions with fewer grid points are removed before the tracking
CT_THRESHOLD = 0.001                       # ConTrack: tracks the anomaly above +this (heatwave) or below -this (coldwave)
CT_OVERLAP = 0.40                          # ConTrack: overlap of a region between two time steps
CT_PERSISTENCE_DAYS = 4                    # ConTrack: minimum life time in days (16 steps for 6-hourly data)
CT_TWOSIDED = True                         # ConTrack: overlap test forward and backward in time
CT_SAVE_HOT_ANOMALY = True                 # write hot_anomaly.nc, the field that is given to ConTrack
CT_REUSE_HOT_ANOMALY = False               # True: read hot_anomaly.nc instead of computing it again (the slow part)

# =========================================================================
#                         END OF USER SETTINGS
# =========================================================================


def check_percentile(percentile):
    if EVENT_TYPE == "heatwave" and percentile <= 0.5:
        raise ValueError("A heatwave needs a percentile above 0.5, for example 0.95")
    if EVENT_TYPE == "coldwave" and percentile >= 0.5:
        raise ValueError("A coldwave needs a percentile below 0.5, for example 0.05")


def print_table(df):
    df = df.copy()
    for column in df.columns:
        if column.endswith("_date") or column == "peak_day":
            df[column] = df[column].dt.strftime("%Y-%m-%d")
    df["source_event_ids"] = df["source_event_ids"].apply(lambda ids: ";".join(map(str, ids)))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def run_track_obj():
    check_percentile(OBJ_PERCENTILE)
    print("1. Reading the temperature ...")
    t = load_temperature(INPUT_FILE, VARIABLE, LAT_RANGE, LON_RANGE, DETECTION_YEARS)
    land = get_land_mask(t, LAND_MASK, LAND_MASK_FILE, LAND_MASK_VARIABLE)
    print(f"   {t.sizes['time']} days, {t.sizes['lat']} x {t.sizes['lon']} grid points")

    print("2. Threshold ...")
    if OBJ_COMPUTE_CLIMATOLOGY:
        baseline = t if OBJ_BASELINE_FILE is None else load_temperature(OBJ_BASELINE_FILE, VARIABLE, LAT_RANGE, LON_RANGE)
        clim = compute_climatology(baseline, OBJ_BASELINE_START, OBJ_BASELINE_END, OBJ_PERCENTILE,
                                   OBJ_WINDOW_DAYS, SEASON_MONTHS)
        save_climatology(clim, OBJ_CLIMATOLOGY_FILE)
    else:
        clim = load_climatology(OBJ_CLIMATOLOGY_FILE, OBJ_PERCENTILE, OBJ_WINDOW_DAYS)

    print("3. Anomaly, hot (cold) points and coherence filter ...")
    t_season, anomaly, hot = make_anomaly(t, clim, SEASON_MONTHS, EVENT_TYPE, land)
    mask = coherence_filter(hot, land, OBJ_COHERENCE_WINDOW_DEG, OBJ_COHERENCE_FRACTION)

    print("4. Tracking ...")
    events = track_obj(mask, land, OBJ_OVERLAP, OBJ_CLOSING_ITERATIONS)

    print("5. Events and episodes ...")
    events = filter_events(events, OBJ_MIN_DURATION_DAYS, OBJ_MIN_GRIDCELLS)
    events_df = summarise_events(events, t_season, anomaly, EVENT_TYPE)
    if OBJ_MERGE_EVENTS:
        episodes_df = merge_into_episodes(events_df, OBJ_MERGE_GAP_DAYS, OBJ_MERGE_CENTROID_DEG, EVENT_TYPE)
    else:
        episodes_df = episodes_without_merging(events_df)
    mask_ds = build_event_mask(events, t_season, EVENT_TYPE) if OBJ_SAVE_MASK else None
    save_results(OUTPUT_DIR, events_df, episodes_df, mask_ds, anomaly if OBJ_SAVE_ANOMALY else None)

    print(f"\n{len(events_df)} events, {len(episodes_df)} episodes ({EVENT_TYPE}, track_obj)")
    print_table(episodes_df)


def run_contrack_workflow():
    check_percentile(CT_PERCENTILE)
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    hot_file = out / "hot_anomaly.nc"

    print("1. Reading the temperature (all time steps) ...")
    t = load_timesteps(INPUT_FILE, VARIABLE, LAT_RANGE, LON_RANGE, SEASON_MONTHS, DETECTION_YEARS)
    land = get_land_mask(t, LAND_MASK, LAND_MASK_FILE, LAND_MASK_VARIABLE)
    steps = steps_per_day(t)
    print(f"   {t.sizes['time']} time steps ({steps} per day), {t.sizes['lat']} x {t.sizes['lon']} grid points")

    if CT_REUSE_HOT_ANOMALY and hot_file.exists():
        print(f"2. Reading the extreme anomaly from {hot_file} ...")
        hot_anomaly = xr.open_dataarray(hot_file).load()
    else:
        print("2. Transient climatology, anomaly and extreme points (the slow part) ...")
        extreme = extreme_anomaly(t, EVENT_TYPE, CT_HALF_WINDOW_YEARS, CT_PERCENTILE, CT_WINDOW_DAYS,
                                  CT_SMOOTHING_DAYS, CT_SMOOTHING_MIN_DAYS)
        print("3. Land mask, removal of small objects, year boundaries ...")
        hot_anomaly = mask_year_boundaries(clean_extreme_anomaly(extreme, land, CT_MIN_OBJECT_SIZE))
        if CT_SAVE_HOT_ANOMALY:
            hot_anomaly.to_dataset().to_netcdf(hot_file, encoding={"anomaly": {"zlib": True, "complevel": 5}})

    persistence_steps = int(round(CT_PERSISTENCE_DAYS * steps))
    print(f"4. ConTrack (persistence {CT_PERSISTENCE_DAYS} days = {persistence_steps} time steps) ...")
    flag, lifecycle = run_contrack(hot_anomaly, EVENT_TYPE, CT_THRESHOLD, CT_OVERLAP, persistence_steps, CT_TWOSIDED)
    flag.to_dataset().to_netcdf(out / "contrack_flag.nc", encoding={"flag": {"zlib": True, "complevel": 5}})
    lifecycle.to_csv(out / "contrack_lifecycle.csv", index=False)

    print(f"\nResults written to {out}/  (contrack_flag.nc, contrack_lifecycle.csv)")
    print(f"{len(np.unique(flag.values[flag.values > 0]))} events ({EVENT_TYPE}, contrack)")
    if len(lifecycle):
        flag_col, date_col, size_col = lifecycle.columns[[0, 1, 5]]
        print(lifecycle.groupby(flag_col).agg(first=(date_col, "first"), last=(date_col, "last"),
                                              steps=(date_col, "size"), max_size=(size_col, "max")).to_string())


def main():
    if EVENT_TYPE not in ("heatwave", "coldwave"):
        raise ValueError("EVENT_TYPE must be 'heatwave' or 'coldwave'")
    if METHOD == "track_obj":
        run_track_obj()
    elif METHOD == "contrack":
        run_contrack_workflow()
    else:
        raise ValueError("METHOD must be 'track_obj' or 'contrack'")


if __name__ == "__main__":
    main()