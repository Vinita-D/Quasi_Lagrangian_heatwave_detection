"""
Step 1: download ERA5 2 m temperature (and the land-sea mask) from the Copernicus Climate Data Store, and save it in the format use to run other files.

you will need an account in  https://cds.climate.copernicus.eu and install cdsapi


Output: one file with variable t2m and coordinates time, lat, lon (+ land_sea_mask.nc).
"""
from pathlib import Path

import cdsapi
import xarray as xr

# ============================================================================ settings
YEARS = range(1979, 2022)                  # 1979-2021
MONTHS = ['06', '07', '08']                # June, July, August
TIMES = ['00:00', '06:00', '12:00', '18:00']   # 6-hourly
AREA = [90, -180, 20, 180]                 # north, west, south, east
GRID = [1.0, 1.0]                          # degrees
OUT_DIR = Path('data/era5')
OUT_FILE = Path('data/ERA5_t2m_JJA_6hourly.nc')


# ============================================================================ functions
def standardise(ds):
    """Same names whatever the CDS gives: variable t2m, coordinates time, lat, lon."""
    names = {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'}
    ds = ds.rename({k: v for k, v in names.items() if k in ds.variables})
    for name in ['2t', 'VAR_2T', 'var167', 't2m']:          # names used for 2 m temperature
        if name in ds.data_vars:
            ds = ds.rename({name: 't2m'})
            break
    ds = ds.drop_vars([c for c in ['number', 'expver'] if c in ds.variables])
    return ds.sortby('lat')                                  # south to north


def download(client, variable, year, target):
    """Download one year of one variable (skipped if the file already exists)."""
    if target.exists():
        return
    client.retrieve('reanalysis-era5-single-levels', {
        'product_type': ['reanalysis'],
        'variable': [variable],
        'year': [str(year)],
        'month': MONTHS,
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': TIMES,
        'area': AREA,
        'grid': GRID,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }, str(target))


# ============================================================================ main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    # temperature, one file per year
    files = []
    for year in YEARS:
        f = OUT_DIR / f't2m_{year}.nc'
        print(f'downloading {year}')
        download(client, '2m_temperature', year, f)
        files.append(f)

    # one standardised file for the detection
    ds = xr.concat([standardise(xr.open_dataset(f)) for f in files], dim='time')
    ds.to_netcdf(OUT_FILE)
    print(f'written {OUT_FILE}')

    # land-sea mask (constant, one time step is enough)
    lsm = OUT_DIR / 'land_sea_mask_raw.nc'
    download(client, 'land_sea_mask', YEARS[0], lsm)
    standardise(xr.open_dataset(lsm)).isel(time=0).to_netcdf('data/land_sea_mask.nc')
    print('written data/land_sea_mask.nc')


if __name__ == '__main__':
    main()