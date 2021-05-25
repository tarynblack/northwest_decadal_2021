# Classes and methods for intake and management of glacier terminus data.

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely import ops
import numpy as np


class Glacier:
    def __init__(self, gid):
        self.gid = gid
        self.refline = LineString()
        self.refbox = LineString()
        self.centerline = LineString()
        self.officialname = ''
        self.greenlandicname = ''
        self.alternativename = ''
        self.obsseries = []
        # Observed values
        self.dates = []
        self.years = []
        self.hydroyears = []
        self.seasons = []
        self.daysofyear = []
        self.daysofhydroyear = []
        self.areas = []
        self.widths = []
        self.meanwidth = []
        self.lengths = []
        self.termareas = []
        # Derived values
        self.name = self.getGlacierName()
        self.missingyears = []
        self.interpyears = []
        self.datayears = []
        self.interpareas = []
        self.interplengths = []
        self.interptermareas = []

    # Internal methods
    
    def sortDates(self):
        self.obsseries = sorted(self.obsseries, key=lambda k: k.date)
    
    def extract(self, attr):
        """Extract list of a given attribute for each TerminusObservation in the Glacier observation series"""
        data_list = eval('[obs.{} for obs in self.obsseries]'.format(attr))
        data_list = pd.Series(data_list)
        return data_list

    def getGlacierName(self):
        """Determine which glacier name to use."""
        if self.officialname.strip() and self.officialname == self.greenlandicname:
            name = '{} ({})'.format(self.officialname, self.alternativename)
        elif self.officialname.strip():
            name = self.officialname
        elif self.greenlandicname.strip():
            name = self.greenlandicname
        elif self.alternativename.strip():
            name = self.alternativename
        else:
            name = 'Glacier #{}'.format(self.gid)
        return name

    def getMissingYears(self, year_list):
        """Identify years with missing data for an attribute"""
        observed_hydroyears = self.extract('hydroyear')
        missing_years = list(set(year_list) - set(observed_hydroyears))
        return missing_years
    
    def interpolateMeasurements(self, attr, year_list):
        """Interpolate values between observations of area or length. Do not interpolate prior to first observation."""
        obs_attr = getattr(self, attr).values.astype('float32')
        observations = pd.DataFrame(data={attr: obs_attr},\
                                    index=self.hydroyears)
        missing_years = self.getMissingYears(year_list)
        missing_data = pd.DataFrame(data={attr: None}, \
                                    index=missing_years)
        interpolated_data = observations.append(missing_data, sort=True).sort_index()
        interpolated_data = interpolated_data.interpolate(method='linear', \
            limit_direction='forward')
        return interpolated_data

    def getInterpolatedYears(self, attr, year_list):
        """Identify years in which data have been interpolated."""
        interpolated_data = self.interpolateMeasurements(attr, year_list)
        interpolated_years_index = interpolated_data.dropna().index
        observed_years = self.hydroyears
        interpolated_years = list(
            set(interpolated_years_index) - set(observed_years))
        return interpolated_years
    
    def getDataYears(self, attr, year_list):
        """Identify all years for which an observation or interpolation is present."""
        observed_years = self.hydroyears
        interpolated_years = self.getInterpolatedYears(attr, year_list)
        data_years = list(set(observed_years) | set(interpolated_years))
        data_years.sort()
        return data_years

    # External methods
    
    def addObservation(self, observation):
        if observation.gid != self.gid:
            print('Cannot add glacier %s observation to glacier %s observation series' % (observation.gid, self.gid))
        self.obsseries.append(observation)
        self.sortDates()
    
    def updateObservedValues(self):
        self.dates = self.extract('date')
        self.years = self.extract('year')
        self.hydroyears = self.extract('hydroyear')
        self.seasons = self.extract('season')
        self.daysofyear = self.extract('dayofyear')
        self.daysofhydroyear = self.extract('dayofhydroyear')
        self.areas = self.extract('area')
        self.widths = self.extract('width')
        self.meanwidth = self.widths.mean()
        # self.lengths = self.extract('length')
        self.lengths = self.areas / self.meanwidth
        self.termareas = self.extract('termarea')

    def updateDerivedValues(self, year_list):
        self.name = self.getGlacierName()
        self.missingyears = self.getMissingYears(year_list)
        self.interpyears = self.getInterpolatedYears('areas', year_list)
        self.datayears = self.getDataYears('areas', year_list)
        self.interpareas = self.interpolateMeasurements('areas', year_list)
        self.interplengths = self.interpolateMeasurements('lengths', year_list)
        self.interptermareas = self.interpolateMeasurements('termareas', year_list)

    def filterDates(self, attr, startdate, enddate):
        """Filter data to between selected dates."""
        attrs = getattr(self, attr)
        if attr in ['lengths', 'areas', 'termareas', 'widths']:
            dates = pd.Series(pd.to_datetime(self.dates))
            if startdate is not None:
                startdate = pd.to_datetime(startdate)
                dates = dates.loc[dates[dates >= startdate].index]
                attrs = attrs.loc[dates[dates >= startdate].index]
            if enddate is not None:
                enddate = pd.to_datetime(enddate)
                dates = dates.loc[dates[dates <= enddate].index]
                attrs = attrs.loc[dates[dates <= enddate].index]
        elif attr in ['interplengths', 'interpareas', 'interptermareas']:
            dates = pd.Series(pd.to_datetime(self.datayears, format='%Y'))
            if startdate is not None:
                startdate = pd.to_datetime(startdate)
                dates = dates.loc[dates[dates >= startdate].index]
            if enddate is not None:
                enddate = pd.to_datetime(enddate)
                dates = dates.loc[dates[dates <= enddate].index]
            attrs = attrs.loc[[d.year for d in dates]]
        return attrs, dates
    
    def filterSeasons(self, attr, season=None, startdate=None, enddate=None):
        """Filter data to specified season and dates."""
        attrs, dates = self.filterDates(attr, startdate, enddate)
        if season is not None:
            season = season.upper()
            season_dates = dates.where(self.seasons == season).dropna()
            season_attrs = attrs.where(self.seasons == season).dropna()
        else:
            season_dates = dates
            season_attrs = attrs
        return season_dates, season_attrs

    def cumulativeChange(self, attr, startdate=None, enddate=None):
        """Calculate net change of attribute between start and end dates."""
        attrs, change_dates = self.filterDates(attr, startdate, enddate)
        num_observations = len(attrs)

        if num_observations == 0:
            print('No observations for {} (#{}) between {} and {}.'.format(
                self.name, self.gid, startdate, enddate))
            attrs = pd.Series(0.0)
            change_dates = (startdate, enddate)
        
        elif change_dates.index[0] != 0:
            """In order to calculate attr difference in time subsets, there needs to be one attr measurement prior to the beginning of the subset. If there is only one measurement in the subset, then the area change is relative to the previous measurement."""
            new_attr_index = attrs.index[0] - 1
            new_date_index = change_dates.index[0] - 1
            # Get attribute value from one timestep before startdate
            prev_attr = getattr(self, attr).loc[new_attr_index]
            attrs.loc[new_attr_index] = prev_attr
            attrs.sort_index(inplace=True)
            # Get date from one timestep before startdate
            if attr in ['lengths', 'areas', 'termareas']:
                dates = pd.Series(pd.to_datetime(self.dates))
            elif attr in ['interplengths', 'interpareas', 'interptermareas']:
                dates = pd.Series(pd.to_datetime(self.datayears, format='%Y'))
            prev_date = dates.iloc[new_date_index]
            change_dates.loc[new_date_index] = prev_date
            change_dates.sort_index(inplace=True)
            # change_dates = (dates.iloc[0].date(), dates.iloc[-1].date())
        
        # Calculate inter-measurement, cumulative, and net change
        attr_change = attrs.diff()
        cumulative_change = attr_change.cumsum()
        cumulative_change.iloc[0] = 0.0

        return cumulative_change, change_dates, num_observations
    
    def rateChange(self, attr, startdate=None, enddate=None):
        """Calculate rate of change in units per day."""
        cumulative_change, change_dates, _ = self.cumulativeChange(attr, \
            startdate, enddate)
        interobs_change = cumulative_change.diff().values
        interobs_days = [d.days for d in change_dates.diff()]
        interobs_change_rate = interobs_change / interobs_days
        return interobs_change_rate
    
    def netRateChange(self, attr, startdate=None, enddate=None):
        """Calculate rate of change (units per year) of attribute between start and end dates."""
        cumulative_change, change_dates, _ = self.cumulativeChange(attr, \
            startdate, enddate)
        net_change = cumulative_change.iloc[-1]
        date_first = change_dates[0]
        date_last = change_dates[-1]

        # Get length of time range in years (SMS=semi-month start frequency (1st and 15th), slightly more precise than just months)
        years_diff = len(pd.date_range(date_first, date_last, freq='SMS'))/24

        # Average rate of area change per year = net change / years
        net_rate_change = net_change / years_diff
        return net_rate_change

    def normChange(self, attr, startdate=None, enddate=None):
        """Calculate normalized attribute change over a time period. "Normalized" such that 0 is scaled to the smallest value of the attribute in the time period, and 1 is scaled to the largest value, with all other values linearly scaled in between."""
        cumulative_change, scaled_dates, _ = self.cumulativeChange(
            attr, startdate, enddate)
        max_val = cumulative_change.values.max()
        min_val = cumulative_change.values.min()
        scaled_attr = (cumulative_change - min_val) / (max_val - min_val)
        return scaled_attr, scaled_dates


class TerminusObservation:
    def __init__(self, gid, qflag, termination, imageid, sensor, date, \
        terminus, referencebox):
        # Attributes that must be defined on instantiation
        self.gid = gid
        self.qflag = qflag
        self.termination = termination
        self.imageid = imageid
        self.sensor = sensor
        self.date = pd.to_datetime(date)
        self.terminus = terminus
        self.referencebox = referencebox
        # Optional additional attributes
        self.centerline = LineString()
        # Attributes that are determined from initial instance attributes
        self.year = self.date.year
        self.hydroyear = self.getHydrologicalYear()
        self.season = self.getSeason()
        self.dayofyear = self.getDayOfYear('01-01')
        self.dayofhydroyear = self.getDayOfYear('09-01')
        # Derived values
        self.area = self.getArea()
        self.width = self.getTerminusWidth()
        self.length = self.area / self.width
        self.termarea = 0.0
        # self.centerlineintersection = self.getCenterlineIntersection()
        # self.centerlinelength = self.getCenterlineLength()
    
    def getHydrologicalYear(self):
        """Determine hydrological year of a given date. Greenland hydrological year
        is defined as September 1 through August 31. Returns the starting year of 
        the hydrological year (aligned with September)."""
        date = pd.to_datetime(self.date)
        if pd.notnull(date):
            if date.month >= 9:
                hydroyear = date.year
            elif date.month < 9:
                hydroyear = date.year - 1
            return hydroyear

    def getSeason(self):
        """Determine Northern Hemisphere season based on date."""
        date = pd.to_datetime(self.date)
        month = date.month
        if month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        elif month in [9, 10, 11]:
            season = 'autumn'
        elif month in [12, 1, 2]:
            season = 'winter'
        return season
    
    def getDayOfYear(self, startMMDD):
        """Convert date to day of year relative to a given start month and day ('MM-DD')."""
        date = pd.to_datetime(self.date)
        if pd.notnull(date):
            startdate = pd.to_datetime(str(self.date.year)+'-'+startMMDD)
            if not date > startdate:
                startdate = pd.to_datetime(str(self.date.year - 1)+'-'+startMMDD)
            dayofyear = (date - startdate).days
        return dayofyear

    def getArea(self):
        """Calculate glacier area (in km2) relative to reference box."""
        outline = self.terminus.union(self.referencebox)
        glacierpoly = ops.polygonize_full(outline)[0]
        if glacierpoly.is_empty:
            print('{}: Glacier {} does not fully intersect box'.format(
                self.date, self.gid))
            area_km = np.nan
        else:
            area_km = glacierpoly.area / 10**6
        return area_km
    
    def getTerminusWidth(self):
        """Compute box width at points where terminus intersects box, and return average width in km."""
        if self.referencebox.geom_type == 'MultiLineString':
            box = ops.linemerge(ops.MultiLineString(self.referencebox))
        else:
            box = ops.LineString(self.referencebox)
        # Split box into two halves
        half1 = ops.substring(box, 0, 0.5, normalized=True)
        half2 = ops.substring(box, 0.5, 1, normalized=True)
        # Get terminus intersections with box halves
        tx1 = self.terminus.intersection(half1)
        tx2 = self.terminus.intersection(half2)
        if tx1.is_empty or tx2.is_empty:
            print('{}: Glacier {} does not intersect box'.format(self.date, self.gid))
            average_width = np.nan
        else:
            # Get distance from intersection point to other half
            tx1_dist = tx1.distance(half2)
            tx2_dist = tx2.distance(half1)
            # Get average distance across box, i.e. average terminus width
            average_width = (tx1_dist + tx2_dist) / 2 / 10**3
        return average_width
    
    def getCenterlineIntersection(self):
        """Locate intersection between glacier outline and glacier centerline. Split centerline at the intersection and calculate the length of the substring. Return intersection point and substring length."""
        intersection_point = self.centerline.intersection(self.terminus)
        return intersection_point
    
    def getCenterlineLength(self):
        """Split centerline at intersection point and calculate the length of the substring."""
        # TODO: fix error "Splitting GeometryCollection geometry is not supported"
        split_centerline = ops.split(self.centerline, self.terminus)
        substring_length = split_centerline[0].length
        return substring_length


# def shp2gdf(file, epsg=3574):
#     """Reads a shapefile of glacier data and reprojects to a specified EPSG (default is EPSG:3574 - WGS 84 / North Pole LAEA Atlantic, unit=meters)
#     Result is a geodataframe containing the shapefile data.
#     Also reindex the gdf so that index=GlacierID for direct selection by ID."""
#     gdf = gpd.read_file(file)
#     gdf = gdf.to_crs(epsg=epsg)
#     # TODO: check whether glacier has multiple entries
#     # (then will have multiple indices of that value, which screws up indexing)
#     gdf = gdf.sort_values(by='GlacierID').set_index('GlacierID', drop=False)
#     return gdf


# def glacierInfo(termini_gdf, box_gdf, gid):
#     """Get terminus and metadata for a given glacier ID in the geodataframe of
#     termini data, as well as the glacier's reference box."""
#     terminus = termini_gdf.loc[gid]
#     box = box_gdf.loc[gid]
#     return terminus, box
