# Read and analyze NetCDF4 climate data

#%% -- Imports
import sys
sys.path.append('/mnt/e/')

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
# from icetools import climplotlib as cpl


#%% -- Functions

def checkType(dtype):
    dtype_list = ['ecco4', 'ecco5', 'noaa', 'hadley', 'icesdk', 'mar']
    if dtype not in dtype_list:
        raise ValueError('Invalid data type. Expected one of: {}'.format(dtype_list))


def loadData(dir, dtype):
    checkType(dtype)
    if dtype is 'ecco5':
        data = xr.open_mfdataset(dir+'*.nc', parallel=True, decode_times=False)
        units, ref_date = data.TIME.attrs['units'].split('since')
        data['TIME'] = pd.date_range(start=ref_date, periods=data.sizes['TIME'], freq='MS')
        data = data.rename({'TIME': 'time'})
    elif dtype is 'icesdk':
        data = pd.read_table(dir, sep=',')
        new_cols = {'Latitude [degrees_north]': 'latitude',
            'Longitude [degrees_east]': 'longitude',
            'yyyy-mm-ddThh:mm': 'time',
            'PRES [db]': 'pressure',
            'TEMP [deg C]': 'temperature'}
        data.rename(columns=new_cols, inplace=True)
        # convert pressure (reported in decibars) to depth (1 db = 10,000 Pa)
        data['depth'] = 10 * data.pressure / 9.81
        data['time'] = pd.to_datetime(data['time'])
    else:
        data = xr.open_mfdataset(dir+'*.nc', parallel=True)
    return data


# def loadDKData(file):
#     data = pd.read_table(file, sep=',')
#     new_cols = {'Latitude [degrees_north]': 'latitude',
#                 'Longitude [degrees_east]': 'longitude',
#                 'yyyy-mm-ddThh:mm': 'time',
#                 'PRES [db]': 'pressure',
#                 'TEMP [deg C]': 'temperature'}
#     data.rename(columns=new_cols, inplace=True)
#     # convert pressure (reported in decibars) to depth (1 db = 10,000 Pa)
#     data['depth'] = 10 * data.pressure / 9.81
#     data['time'] = pd.to_datetime(data['time'])
#     return data


def eccoDepth(data, dtype):
    checkType(dtype)
    if dtype is 'ecco4':
        # pass actual ecco data
        t0 = data.isel(time=0)
        z = list(zip(t0.k.values, t0.Z.values))
    elif dtype is 'ecco5':
        z = list(zip(list(range(len(data.DEPTH_T))), data.DEPTH_T.values))
    return z


def eccoTemperature(data, dtype):
    """Convert potential temperature THETA to absolute temperature T."""
    checkType(dtype)
    P0 = 100000 # Pa, or 1 bar
    rhow = 1027 # kg/m3, density of seawater
    g = 9.81 #m/s2, gravity
    Rcp = 0.12 # gas constant / specific heat capacity for seawater
    if dtype is 'ecco4':
        data = data.assign(temperature=lambda x: x.THETA / ((P0 / (rhow*g*abs(x.Z)))**Rcp))
    if dtype is 'ecco5':
        data = data.assign(temperature=lambda x: x.THETA / ((P0 / (rhow*g*abs(x.DEPTH_T)))**Rcp))
    return data


# def eccoNCGridNearestCoord(ncgrid, coordinate):
#     latitude, longitude = coordinate
#     min_distance = 1
#     min_gridcoord = (0, 0, 0)
#     for t, tile in enumerate(ncgrid.tile.values):
#         stack = np.dstack((ncgrid.sel(tile=tile).XC, ncgrid.sel(tile=tile).YC))
#         for j, row in enumerate(stack):
#             for i, coord in enumerate(row):
#                 distance = ((coord[0]-longitude)**2 + (coord[1]-latitude)**2)**0.5
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_gridcoord = (tile, j, i)
#     return min_gridcoord


def subsetGeographic(data, bounds, dtype):
    """Subset data to geographic boundaries. Bounds are listed in order N S E W in decimal degrees."""
    checkType(dtype)
    N, S, E, W = bounds
    if dtype is 'ecco4':
        Ni = N*2 + 179.5
        Si = S*2 + 179.5
        Ei = E*2 + 359.5
        Wi = W*2 + 359.5
        data_subset = data.sel(j=slice(Si, Ni)).sel(i=slice(Wi, Ei))
    elif dtype is 'ecco5':
        data_subset = data.sel(LATITUDE_T=slice(S, N)).sel(LONGITUDE_T=slice(W, E))
    elif dtype is 'hadley':
        if W < 0:
            Wi = W+360
        if E < 0:
            Ei = E+360
        data_subset = data.sel(lat=slice(S, N)).sel(lon=slice(Wi, Ei))
    elif dtype is 'noaa':
        data_subset = data.where((data.latitude > S) & (data.latitude < N) & (data.longitude > W) & (data.longitude < E), drop=True)
    elif dtype is 'mar':
        data_subset = data.where((data.LAT > S) & (data.LAT < N) & (data.LON > W) & (data.LON < E), drop=True)
    elif dtype is 'icesdk':
        data_subset = data.where((data.latitude > S) & (data.latitude < N) & 
                                 (data.longitude > W) & (data.longitude < E)).dropna()
    return data_subset


def subsetTime(data, times, dtype):
    checkType(dtype)
    start, end = times
    if dtype is 'icesdk':
        data_subset = data.where((data.time > start) & (data.time < end)).dropna()
    else:
        data_subset = data.sel(time=slice(start, end))
    return data_subset


def subsetYear(data, year, dtype):
    checkType(dtype)
    year_begin = '{}-01-01'.format(year)
    year_end = '{}-12-31'.format(year)
    if dtype is 'icesdk':
        data_year = data.where((data.time > year_begin) & (data.time < year_end)).dropna()
    else:
        data_year = data.sel(time=slice(year_begin, year_end))
    return data_year


def subsetTopography(data, topobounds, dtype='mar'):
    """Subset data to between topographic levels (lower and upper). For MAR data only."""
    checkType(dtype)
    lower, upper = topobounds
    if dtype == 'mar':
        data_subset = data.where((data.SH >= lower) & (data.SH <= upper), drop=True)
    else:
        data_subset = data
        print('Subset topography only for MAR data')
    return data_subset


def subsetIce(data, threshold=100, dtype='mar'):
    """Subset data to areas that meet or exceed threshold for ice area percentage. For MAR data only."""
    checkType(dtype)
    if dtype == 'mar':
        data_subset = data.where(data.MSK >= threshold, other=np.nan)
    else:
        data_subset = data
        print('Subset topography only for MAR data')
    return data_subset


def convertCoordinates(coordinate, dtype):
    """credit to T Sutterley (github.com/tsutterley/SMBcorr)"""
    latitude, longitude = coordinate
    if dtype is 'mar':
        # MAR model projection: Polar Stereographic (Oblique)
        #-- Earth Radius: 6371229 m
        #-- True Latitude: 0
        #-- Center Longitude: -40
        #-- Center Latitude: 70.5
        proj_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
        "+a=6371229 +no_defs units=km")
        #-- pyproj transformer for converting from latitude/longitude into projected coordinates
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
        crs2 = pyproj.CRS.from_string(proj_params)
        transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        #-- convert projection from latitude/longitude to projected
        gridX, gridY = transformer.transform(longitude, latitude)
    elif dtype is 'noaa':
        proj_params = ("epsg:3411")
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
        crs2 = pyproj.CRS.from_string(proj_params)
        transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        gridX, gridY = transformer.transform(longitude, latitude)
    elif dtype is 'ecco4':
        gridX = longitude*2 + 359.5
        gridY = latitude*2 + 179.5
    return (gridX, gridY)


def returnCoordinates(gridpoint, dtype):
    """Reverse operation of convertCoordinates: takes in gridpoint and returns lat/lon"""
    x, y = gridpoint
    if dtype is 'mar':
        proj_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
        "+a=6371229 +no_defs units=km")
        crs1 = pyproj.CRS.from_string(proj_params)
        crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
        transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        longitude, latitude = transformer.transform(x, y)
    elif dtype is 'noaa':
        proj_params = ("epsg:3411")
        crs1 = pyproj.CRS.from_string(proj_params)
        crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
        transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        longitude, latitude = transformer.transform(x, y)
    elif dtype is 'ecco4':
        longitude = (x - 359.5) / 2
        latitude = (y - 179.5) / 2
    return (latitude, longitude)


def dataAtCoord(data, coordinate, dtype, tolerance=None):
    """Return data at grid point nearest to a geographic coordinate [latitude, longitude]."""
    checkType(dtype)
    latitude, longitude = coordinate
    if dtype is 'noaa':
        X, Y = convertCoordinates(coordinate, dtype)
        data_point = data.sel(xgrid=X, ygrid=Y, method='nearest')
    elif dtype is 'mar':
        X, Y = convertCoordinates(coordinate, dtype)
        data_point = data.sel(X12_251=X, Y20_465=Y, method='nearest')
    elif dtype is 'ecco4':
        X, Y = convertCoordinates(coordinate, dtype)
        data_point = data.sel(i=X, j=Y, method='nearest')
    elif dtype is 'ecco5':
        data_point = data.sel(LATITUDE_T=latitude, LONGITUDE_T=longitude, method='nearest')
    elif dtype is 'hadley':
        if longitude < 0:
            longitude = longitude + 360
        data_point = data.sel(lat=latitude, lon=longitude, method='nearest')
    elif dtype is 'icesdk':
        if tolerance is None:
            raise ValueError('Must specify a coordinate tolerance for icesdk data')
        N = latitude+tolerance
        S = latitude-tolerance
        E = longitude+tolerance
        W = longitude-tolerance
        bounds = [N, S, E, W]
        data_point = subsetGeographic(data, bounds, dtype)
    return data_point


def dataAtGridPoint(data, gridpoint, dtype):
    checkType(dtype)
    x, y = gridpoint
    if dtype == 'noaa':
        data_point = data.sel(xgrid=x, ygrid=y, method='nearest')
    elif dtype == 'ecco4':
        # TODO: check that this case works
        data_point = data.sel(j=y, i=x, method='nearest')
    return data_point


def dataAtDepth(data, depth, dtype, tolerance=None):
    checkType(dtype)
    # if depth entered is positive, make it negative
    if dtype is 'ecco4':
        if depth > 0:
            depth = -1 * depth
        z = eccoDepth(data, dtype='ecco4')
        closest_k = min(range(len(z)), key=lambda i: abs(z[i][1]-depth))
        data_depth = data.isel(k=closest_k)
    elif dtype is 'ecco5':
        if depth < 0:
            depth = -1 * depth
        data_depth =  data.sel(DEPTH_T=depth, method='nearest')
    elif dtype is 'icesdk':
        if tolerance is None:
            raise ValueError('When dtype is icesdk, must set a depth tolerance.')
        upper = depth + tolerance
        lower = depth - tolerance
        data_depth = data.where((data.depth < upper) & (data.depth > lower)).dropna()
    else:
        raise ValueError('Depth data only available for dtypes ecco and icesdk.')
    return data_depth


def selectPointData(data, dtype, depth=None, depth_tolerance=None, point=None, coordinate=None, coordinate_tolerance=None, ncgrid=None):
    checkType(dtype)
    if depth is not None:
        data_depth = dataAtDepth(data, depth, dtype, depth_tolerance)
    else:
        data_depth = data
    if point is not None:
        data_point = dataAtGridPoint(data_depth, point, dtype)
    elif coordinate is not None:
        data_point = dataAtCoord(data_depth, coordinate, dtype, coordinate_tolerance)
    else:
        data_point = data_depth
    return data_point


def trueGridPoint(data_point, dtype):
    checkType(dtype)
    if dtype is 'noaa':
        true_gridpoint = (data_point.xgrid.values, data_point.ygrid.values)
    elif dtype is 'mar':
        true_gridpoint = (data_point.X12_251.values, data_point.Y20_465.values)
    elif dtype is 'ecco4':
        true_gridpoint = (data_point.i.values, data_point.j.values)
    elif dtype is 'hadley':
        true_gridpoint = (data_point.lat.values, data_point.lon.values)
    else:
        true_gridpoint = None
    return true_gridpoint


def icesAnnualMean(data):
    annual_years = pd.to_datetime(data.time.values).year.unique()
    annual_mean = np.empty(0)
    for y in annual_years:
        data_year = subsetYear(data, y, dtype='icesdk')
        mean_var = data_year.temperature.values.mean()
        annual_mean = np.append(annual_mean, mean_var)
    annual_stats = pd.DataFrame({'avg': annual_mean}, index=annual_years)
    return annual_stats


def icesDecadalMean(data):
    decadal_mean = np.empty(0)
    years = pd.to_datetime(data.time.values).year.unique()
    decades = range(years[0]-years[0]%10, years[-1], 10)
    for d in decades:
        dyears = range(d, d+10)
        data_decade = np.empty(0)
        for y in dyears:
            data_year = subsetYear(data, y, dtype='icesdk')
            data_decade = np.append(data_decade, data_year.temperature)
        decade_mean = data_decade.mean()
        decadal_mean = np.append(decadal_mean, decade_mean)
    decadal_stats = pd.DataFrame({'avg': decadal_mean}, index=[d for d in decades])
    return decadal_stats


def seasonMean(data, dvar, quarter_start):
    # data_months = pd.to_datetime(data.time.values).month
    month_dict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 
                  7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
    # data_season = data.where([m in month_ints for m in data_months]).dropna(dim='time')
    # season_mean = data_season.resample(time='AS-{}'.format(month_dict[month_ints[0]])).mean()
    quarterly_mean = data.resample(time='QS-{}'.format(month_dict[quarter_start])).mean()
    season_mean = quarterly_mean.where(pd.to_datetime(quarterly_mean.time.values).month == quarter_start)
    season_mean = season_mean[dvar].dropna(dim='time')
    return season_mean


def meanDatasets(ecco=None, hadley=None, noaa=None):
    mean_datasets = pd.concat([ecco, hadley, noaa]).groupby(level=0).mean()
    return mean_datasets


def seaIceScale(data, dtype):
    checkType(dtype)
    if dtype in ['ecco4', 'ecco5', 'noaa']:
        scaled_data = data * 100
    elif dtype is 'hadley':
        scaled_data = data
    else:
        scaled_data = data
    return scaled_data


def seaIcePresence(data, dvar, dtype, minimum):
    checkType(dtype)
    data = data[dvar].dropna(dim='time')
    scaled_data = seaIceScale(data, dtype)
    ice_mask = data.where(scaled_data>=minimum, other=0)
    ice_presence = ice_mask.where(ice_mask==0, other=1)
    return ice_presence


def seaIceSeasonLength(data, dvar, dtype, minimum, start='SEP', coordinate=None):
    checkType(dtype)
    if coordinate is not None:
        data_point = dataAtCoord(data, coordinate, dtype)
    else:
        data_point = data
    ice_presence = seaIcePresence(data_point, dvar, dtype, minimum)
    ice_season_length = ice_presence.resample(time='AS-{}'.format(start)).sum()
    return ice_season_length


def seaIceSeasonOnset(data, dvar, dtype, minimum, start='SEP', coordinate=None):
    checkType(dtype)
    if coordinate is not None:
        data_point = dataAtCoord(data, coordinate, dtype)
    else: 
        data_point = data
    ice_presence = seaIcePresence(data_point, dvar, dtype, minimum)
    ice_onset = ice_presence.resample(time='AS-{}'.format(start)).apply(lambda x: x.idxmax())
    return ice_onset


def mmweday2myr(data):
    """Convert data units from mmWE/day to m/yr"""
    new_data = data * 365 / 1000
    return new_data


def bulkValuesMAR(data, coords, dvar):
    bulk_values = pd.DataFrame(
        columns=pd.to_datetime(data.TIME.values),
        index=range(len(coords)))
    
    for c in range(len(coords)):
        coord = coords[c]
        data_coord = dataAtCoord(data, coord, dtype='mar')
        data_values = data_coord[dvar].values
        data_values = mmweday2myr(data_values)
        bulk_values.loc[c] = data_values.flatten()
    
    return bulk_values


def bulkMeanMAR(data, coords, dvar):
    bulk_values = bulkValuesMAR(data, coords, dvar)
    bulk_mean = bulk_values.mean(axis=1).mean()
    return bulk_mean


def bulkAnomalyMAR(data, coords, dvar):
    bulk_anomaly = pd.DataFrame(
        columns=pd.to_datetime(data.TIME.values), 
        index=range(len(coords)))
    # for c in range(len(coords)):
    #     coord = coords[c]
    #     data_coord = dataAtCoord(data, coord, dtype='mar')
    #     total_mean = data_coord[dvar].mean().values
    #     anomaly = data_coord[dvar].values - total_mean
    #     total_mean = mmweday2myr(anomaly)
    #     anomaly = mmweday2myr(anomaly)
    #     bulk_anomaly.loc[c] = anomaly.flatten()
    bulk_values = bulkValuesMAR(data, coords, dvar)
    coord_means = bulk_values.mean(axis=1)
    for c in range(len(coords)):
        anomaly = bulk_values.loc[c] - coord_means[c]
        bulk_anomaly.loc[c] = anomaly
    return bulk_anomaly


def decadalAnomalyMAR(data, coords, dvar):
    bulk_anomaly = bulkAnomalyMAR(data, coords, dvar)
    decadal_mean_anomaly = bulk_anomaly.mean()[1:].resample('10AS').mean()
    return decadal_mean_anomaly