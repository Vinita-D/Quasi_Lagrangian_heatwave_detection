"""
make_sample_data.py -- make the sample files of this repository from the large ERA5 file.

Writes
  data/tmax_daily_sample.nc   daily maximum temperature of SAMPLE_YEARS (the data of the examples)
  data/climatology_p95.nc     thresholds computed from the whole record (BASELINE_START to BASELINE_END)
  data/climatology_p05.nc     the same for coldwaves

Edit the settings, then run:   python make_sample_data.py
"""
from pathlib import Path

from heatwave_tools import compute_climatology, load_temperature, save_climatology

# ------------------------------------------------------------------ settings
INPUT_FILE = "path/to/ERA5_t2m_1979_2025_0p5_land_MJJA.nc"      # the large ERA5 file
VARIABLE = "2t"
LAT_RANGE = (30.0, 60.0)                  # mid-latitudes
LON_RANGE = None                          # None = all longitudes
SAMPLE_YEARS = (2015, 2024)               # years of the sample file
BASELINE_START = "1992-06-01"             # period used for the thresholds
BASELINE_END = "2025-08-31"
SEASON_MONTHS = [6, 7, 8]
WINDOW_DAYS = 8
CLIMATOLOGIES = {0.95: "data/climatology_p95.nc",    # percentile: file
                 0.05: "data/climatology_p05.nc"}
SAMPLE_FILE = "data/tmax_daily_sample.nc"
# ---------------------------------------------------------------------------

print("Reading the whole record (daily maximum, year by year) ...")
t = load_temperature(INPUT_FILE, VARIABLE, LAT_RANGE, LON_RANGE)
print(f"  {t.sizes['time']} days, {t.sizes['lat']} x {t.sizes['lon']} grid points")

print(f"Sample file, years {SAMPLE_YEARS[0]}-{SAMPLE_YEARS[1]} ...")
year = t.time.dt.year
sample = t.sel(time=(year >= SAMPLE_YEARS[0]) & (year <= SAMPLE_YEARS[1])).rename(VARIABLE)
Path(SAMPLE_FILE).parent.mkdir(parents=True, exist_ok=True)
sample.to_dataset().to_netcdf(SAMPLE_FILE, encoding={VARIABLE: {"zlib": True, "complevel": 5, "dtype": "float32"}})
print(f"  {SAMPLE_FILE}: {sample.sizes['time']} days, {Path(SAMPLE_FILE).stat().st_size / 1e6:.1f} MB")

for percentile, file in CLIMATOLOGIES.items():
    print(f"Thresholds, percentile {percentile}, baseline {BASELINE_START} to {BASELINE_END} ...")
    clim = compute_climatology(t, BASELINE_START, BASELINE_END, percentile, WINDOW_DAYS, SEASON_MONTHS)
    save_climatology(clim, file)