import pandas as pd

def getHydroyear(date, start_month):
    """Determine hydrological year of a date. 'start_month' is the first month of the hydrological year. In Greenland, hydrological year is defined as September 1 through August 31 (Ettema et al 2009)."""
    pdate = pd.to_datetime(date)
    if pdate.month >= start_month:
        hydroyear = pdate.year
    elif pdate.month < start_month:
        hydroyear = pdate.year - 1
    return hydroyear


def filterGeographic(ds, bounds):
    """Filter xarray dataset down to a specific region defined by lat/lon bounds. 'bounds' is a dictionary where the keys are cardinal directions (NESW) and the values are the boundary eges in degrees."""
    ds_filter = ds.where((ds.latitude >= bounds['S']) &
                         (ds.latitude <= bounds['N']) &
                         (ds.longitude >= bounds['W']) &
                         (ds.longitude <= bounds['E']), drop=True)
    return ds_filter