# quasi-lagrangian-heatwaves

Detect and track heatwaves or coldwaves in temperature data, with two methods:

| `METHOD` | Input | Tracking |
|---|---|---|
| `"track_obj"` | any time resolution, reduced to the daily maximum | own tracking code, events and episodes |
| `"contrack"` | all time steps of the day (hourly, 3-hourly, 6-hourly, ...) | [ConTrack](https://github.com/steidani/ConTrack) |

Both work for any domain and for heatwaves and coldwaves.


## Files

| File | Purpose |
|---|---|
| `run_detection.py` | **The file you run.** All settings are at the top. |
| `heatwave_tools.py` | The functions that do the work. |
| `make_sample_data.py` | Makes the sample files in `data/` from a large ERA5 file. |
| `data/tmax_daily_sample.nc` | Sample: daily maximum temperature, mid-latitudes, 2015-2024. Not in this repository: download it from Zenodo (see Quick start). |
| `data/climatology_p95.nc`, `data/climatology_p05.nc` | Thresholds for `track_obj` (95th and 5th percentile). |

## Install

```bash
conda env create -f environment.yml
conda activate heatwaves
```

ConTrack is only needed for `METHOD = "contrack"`:
`pip install git+https://github.com/steidani/ConTrack.git` (check the ConTrack README).



```bash
python run_detection.py
```

Runs `track_obj` on the sample data. Results are written to `output/`.
Expected result: 
|event_id start_date |  end_date|  max_gridcells  |centroid_lat|  centroid_lon | peak_day_max_temp|
|6103 2021-06-30 2021-07-09     |      2031        |  47.5       |  70.6         |      48.5|
|8023 2023-06-01 2023-06-09     |      1988        | 50.5       |  75.8         |     42.8
|9638 2024-07-02 2024-07-11     |     1596         |49.9         | -116.5        |   49.0

## Settings (top of `run_detection.py`)
Read it carefully

## How the methods work

### `track_obj`

1. **Reading.** Several values per day are reduced to the daily maximum. Kelvin is converted to
   degC. 29 February is removed.
2. **Threshold.** For every calendar day and grid point, the percentile of the temperature
   within +/- `OBJ_WINDOW_DAYS` of that day, over the baseline period. It needs many years
   (decades). With the 10-year sample use `OBJ_COMPUTE_CLIMATOLOGY = False` and the supplied
   climatology file. The file records the percentile and window it was made with, and the code
   stops if the settings do not match. A climatology file is only valid for the grid and the
   months it was made for.
3. **Hot (cold) points.** Land points with temperature above (below) the threshold.
   Anomaly = temperature minus threshold.
4. **Coherence filter.** A hot point stays only if enough of the land around it is hot too.
5. **Tracking.** Small gaps are closed (cross-shaped structure). The points that touch (also
   diagonally) form one region per day. An event continues into the region with the largest
   overlap fraction (overlap divided by the smaller of the two areas) if that fraction is larger
   than `OBJ_OVERLAP`. Events choose in the order they were created. Regions that continue no
   event start a new event. Events never continue from one summer to the next.
6. **Events.** At least `OBJ_MIN_DURATION_DAYS` long and more than `OBJ_MIN_GRIDCELLS` grid points
   on the largest day. Event ids count the events in the order they were created.
7. **Episodes.** An event is added to the episode built so far if it is in the same year, starts
   less than `OBJ_MERGE_GAP_DAYS` days after the episode ends, and its box overlaps the box of
   the episode or the centroids are less than `OBJ_MERGE_CENTROID_DEG` degrees apart.

### `contrack`

Done for every time of the day separately, for any time resolution:

1. **Transient climatology.** Mean of the same time of day and day of the year over the years
   within +/- `CT_HALF_WINDOW_YEARS` (the year itself included). The anomaly is the temperature
   minus this climatology.
2. **Threshold.** For every day of the year, the percentile of the anomalies within
   +/- `CT_WINDOW_DAYS`, then a running mean over `CT_SMOOTHING_DAYS` days (centred; the ends of the
   season use at least `CT_SMOOTHING_MIN_DAYS` days).
3. **Extreme points.** Anomaly above (below) the threshold. The anomaly is kept there, NaN elsewhere.
4. **Cleaning.** Only land points. Regions (points touching along an edge) with fewer than
   `CT_MIN_OBJECT_SIZE` points are removed. This follows scikit-image's `remove_small_objects`
   up to version 0.25. Newer versions changed the meaning of the size argument, so this
   repository uses its own implementation.
5. **Year boundaries.** The last time step of each year and the first of the next are set to NaN,
   so that no event continues into the next summer.
6. **ConTrack.** The extreme anomaly is tracked with the ConTrack settings above.

## Output (in `OUTPUT_DIR`)


## Things to know

- The peak day of an event is the day with the largest area.
- The episode merging of `track_obj` compares bounding boxes. It is meant for a region. On a
  large domain use `OBJ_MERGE_EVENTS = False`.
- Events across the dateline: `lon_min` is the western edge and `lon_max` the eastern edge, so
  `lon_min` can be larger than `lon_max`.
- ConTrack with `CT_TWOSIDED = True` removes the first and last time step of an event, so
  events are shorter and fewer than with `False`.
- With daily data, `contrack` has one time step per day and `CT_PERSISTENCE_DAYS = 4` means 4 steps.
  relates to the scripts archived with the papers (Zenodo DOI).>>



## Sample data
`make_sample_data.py` reads a large ERA5 file (2 m temperature, 0.5 degrees, land only), takes the
daily maximum for the mid-latitudes, writes the sample years and computes the thresholds from the
baseline period. Edit the settings at its top and run it. It reads the file year by year, so it
also works for large files. Source: ERA5 (Hersbach et al., 2020), The sample is daily, which `contrack` also accepts. For the full workflow of `contrack` use a file with several time steps per day. 

## Citation and license



Please cite ConTrack as requested in its README when you use `METHOD = "contrack"`.

