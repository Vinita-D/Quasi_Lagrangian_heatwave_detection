# Quasi_Lagrangian_heatwave_detection
Code to detect and track heatwaves in daily maximum temperature and to detect atmospheric blocks with ConTrack. 

Two independent workflows:

| Workflow | Script | Needs ConTrack |
|---|---|---|
| Heatwave detection and tracking | `detect_heatwaves.py` | no |
| Block detection | `detect_blocks_contrack.py` | yes |

## Install

```bash
conda env create -f environment.yml
conda activate heatwaves
```

ConTrack is only needed for block detection. If the line in `environment.yml`
does not work, follow the installation instructions in the ConTrack README.

## Quick start (no data needed)

```bash
python examples/make_synthetic_example.py      # synthetic Tmax with two planted heatwaves
python detect_heatwaves.py config/greece.yaml
```

Expected result: two episodes, about 2005-06-19 to 2005-06-28 and 2010-07-09 to
2010-07-16. Results are in `output/greece/`.

## Heatwave detection with your own data

**Input.** A NetCDF file of *daily* maximum 2 m temperature with dimensions `time`,
`lat`, `lon`. Ocean points must be NaN (land points have data).
Kelvin is converted to degC automatically. From hourly data, make the daily maximum with
`cdo daymax hourly.nc daily_tmax.nc`. The file must also cover the baseline years of the
threshold and a margin of `window_days` around the summer months.

**Settings.** Copy `config/greece.yaml` and edit it. All settings are described below.

**Run.**

```bash
python detect_heatwaves.py config/greece.yaml
```

**Output** (in `output_dir`)

| File | Content |
|---|---|
| `events.csv` | one row per tracked event: dates, duration, peak day, largest area, bounding box, centroid, peak Tmax and peak anomaly |
| `episodes.csv` | events that overlap in time merged into episodes (the heatwaves of the paper) |
| `heatwave_periods.json` | start and end date of every episode, by year, for use in later analyses |
| `tmax_anomaly.nc` | daily Tmax minus threshold, summer months (if `save_anomaly: true`) |

### How it works

1. **Threshold.** For every calendar day and grid point, the `percentile` of Tmax within
   +/- `window_days` of that day, over the baseline years.
2. **Hot points.** Land points in the summer months with Tmax above the threshold.
   The anomaly is Tmax minus this threshold.
3. **Regions.** Hot points that touch (also diagonally) form one region per day.
4. **Tracking.** A region continues an event if it has the largest overlap with the
   event's region of the previous day, and that overlap divided by the smaller of the two
   areas is larger than `overlap_threshold`. Events never continue across summers.
5. **Filtering.** Events must last at least `min_duration_days` and cover more than
   `min_gridcells` grid points on their largest day.
6. **Episodes.** Events that overlap in time, or are less than `merge_gap_days` apart,
   are merged into one episode. The peak of an episode is the peak of the member event
   with the largest anomaly.

### Settings

| Setting | Meaning | Value used in the paper |
|---|---|---|
| `variable` | variable name in the file | <<>> |
| `lat_range`, `lon_range` | domain | <<>> |
| `summer_months` | months searched for heatwaves | <<>> |
| `climatology_years` | baseline years of the threshold | <<>> |
| `percentile` | percentile of the threshold | <<>> |
| `window_days` | days on each side in the threshold window | <<>> |
| `overlap_threshold` | overlap needed to continue an event | <<>> |
| `min_duration_days` | minimum duration | <<>> |
| `min_gridcells` | minimum area (grid points, more than) | <<>> |
| `merge_gap_days` | gap that still counts as one episode | <<>> |

### Things to know

- The peak day is the day with the largest area. The bounding box and centroid describe
  the region on that day.
- An event can start from a very small region that later grows, and end in a small
  remnant. Start and end dates can therefore be one or two days wider than the core of
  the event.
- An event follows one region per day. Splitting and merging of regions are not
  treated separately.
- The anomaly is relative to a moving percentile threshold, not to the mean climate.
- <<Say here how this version relates to the scripts archived with the paper
  (Zenodo DOI), and whether the heatwaves of Table 2 are reproduced.>>

## Small example from real data

To put a small example file in `data/`, cut it from a large daily Tmax file:

```bash
python make_example_data.py big_daily_tmax.nc data/tmax_daily_example.nc
```

Example data from ERA5 are described in `data/README.md`. <<add the source and version>>

## Block detection with ConTrack

```bash
python detect_blocks_contrack.py config/blocks.yaml
```

Edit `config/blocks.yaml`: the input file, the variable name and its units, an optional
`scale_factor`, and the ConTrack settings. `persistence` is counted in time steps of the
input file. Output: `block_flag.nc` and `block_lifecycle.csv` in `output_dir`.

Please cite ConTrack as requested in its README.

## Citation and license

<<citation and Zenodo DOI of this repository>>

Input data: ERA5 (Hersbach et al., 2020).

<<license>>
