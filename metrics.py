#!/usr/bin/env python3
# Calculate metrics for glacier change
# Taryn Black, August 2020

import pandas as pd
from shapely import ops
import ruptures as rpt
import pwlf
import numpy as np
from GPyOpt.methods import BayesianOptimization
import scipy.stats

def addDecade(start_date):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.date_range(start_date, periods=2, freq='9Y')[-1]
    end_date = end_date.date()
    return end_date


# def filterByDates(glacier, measure, date_start, date_end):
#     measures = glacier.extract(measure)

#     # Filter to data between selected dates
#     if date_start:
#         date_start = pd.to_datetime(date_start)
#         measures = measures.where(glacier.dates >= date_start).dropna()
#         dates = dates.where(glacier.dates >= date_start).dropna()
#     if date_end: 
#         date_end = pd.to_datetime(date_end)
#         measures = measures.where(glacier.dates <= date_end).dropna()
#         dates = dates.where(glacier.dates <= date_end).dropna()
#     return measures, dates


def firstFullYear(all_glaciers):
    """Identify the first year in which all glaciers have either an observed or an interpolated data point."""
    first_full_year = 0
    for g in all_glaciers:
        g_firstyear = all_glaciers[g].datayears[0]
        if g_firstyear > first_full_year:
            first_full_year = g_firstyear
    return first_full_year


def hydrologicalYear(date, month):
    """Determine hydrological year of a given date, for hydrological year starting in month."""
    date = pd.to_datetime(date)
    if pd.notnull(date):
        if date.month >= month:
            hydroyear = date.year
        elif date.month < month:
            hydroyear = date.year - 1
        return hydroyear


def finalNetChange(glaciers, attr, startdate=None, enddate=None):
    """Calculate final (cumulative) net change in attribute (e.g. area)."""
    final_net_change = pd.Series(index=glaciers.keys())
    for g in glaciers:
        glacier = glaciers[g]
        cumul_change, _, _ = glacier.cumulativeChange(attr, startdate, enddate)
        final_net_change.loc[g] = cumul_change.iloc[-1].values
    return final_net_change


def stdevChange(glaciers, attr, startdate=None, enddate=None):
    """Test whether glacier net change is outside one standard deviation of its variability."""
    change_stdev = pd.Series(index=glaciers.keys())
    for g in glaciers:
        glacier = glaciers[g]
        # change_stdev.at[g] = getattr(glacier, attr).std(skipna=True)
        series = glacier.filterDates(attr=attr, startdate=startdate, enddate=enddate)[0]
        change_stdev.at[g] = series.std(skipna=True)
    return change_stdev


def significantChange(glaciers, attr, startdate=None, enddate=None):
    """Identify glaciers that have experienced significant change, defined as a net change greater than one standard deviation of the total variability."""
    final_net_change = finalNetChange(glaciers, attr, startdate, enddate)
    change_stdev = stdevChange(glaciers, attr, startdate, enddate)
    significance = abs(final_net_change) > (2 * change_stdev)
    return significance


def stableGlaciers(glaciers, attr, startdate=None, enddate=None):
    """Identify glaciers that have been stable in an attribute since startdate."""
    final_net_change = finalNetChange(glaciers, attr, startdate, enddate)
    significance = significantChange(glaciers, attr, startdate, enddate)
    stable_glaciers = final_net_change.where(significance==False).dropna()
    return stable_glaciers
    

def advancingGlaciers(glaciers, startdate=None, enddate=None):
    """Identify glaciers which have had significant net increase in length since startdate."""
    net_length_change = finalNetChange(glaciers, 'interplengths', startdate, enddate)
    significance = significantChange(glaciers, 'interplengths', startdate, enddate)
    advancing_glaciers = net_length_change.where(net_length_change > 0).dropna()
    advancing_glaciers = advancing_glaciers.where(significance==True).dropna()
    return advancing_glaciers


def growingGlaciers(glaciers, startdate=None, enddate=None):
    """Identify glaciers which have had significant net increase in area since startdate."""
    net_area_change = finalNetChange(glaciers, 'interpareas', startdate, enddate)
    significance = significantChange(glaciers, 'interpareas', startdate, enddate)
    growing_glaciers = net_area_change.where(net_area_change > 0).dropna()
    growing_glaciers = growing_glaciers.where(significance==True).dropna()
    return growing_glaciers


def retreatingGlaciers(glaciers, startdate=None, enddate=None):
    """Identify glaciers which have had significant net decrease in length since startdate."""
    net_length_change = finalNetChange(glaciers, 'interplengths', startdate, enddate)
    significance = significantChange(glaciers, 'interplengths', startdate, enddate)
    retreating_glaciers = net_length_change.where(net_length_change < 0).dropna()
    retreating_glaciers = retreating_glaciers.where(significance==True).dropna()
    return retreating_glaciers


def shrinkingGlaciers(glaciers, startdate=None, enddate=None):
    """Identify glaciers which have had significant net decrease in area since startdate."""
    net_area_change = finalNetChange(glaciers, 'interpareas', startdate, enddate)
    significance = significantChange(glaciers, 'interpareas', startdate, enddate)
    shrinking_glaciers = net_area_change.where(net_area_change < 0).dropna()
    shrinking_glaciers = shrinking_glaciers.where(significance==True).dropna()
    return shrinking_glaciers


def lengthType(glacier, startdate=None, enddate=None):
    net_length_change = glacier.cumulativeChange('interplengths', startdate, enddate)[0].iloc[-1].lengths
    change_stdev = glacier.filterDates(attr='interplengths', startdate=startdate, enddate=enddate)[0].std(skipna=True).lengths
    significance = abs(net_length_change) > change_stdev
    if not significance:
        length_type = 'stable'
    elif significance and net_length_change > 0:
        length_type = 'advancing'
    elif significance and net_length_change < 0:
        length_type = 'retreating'
    return length_type


def areaType(glacier, startdate=None, enddate=None):
    net_area_change = glacier.cumulativeChange('interpareas', startdate, enddate)[0].iloc[-1].areas
    change_stdev = glacier.filterDates(attr='interpareas', startdate=startdate, enddate=enddate)[0].std(skipna=True).areas
    significance = abs(net_area_change) > change_stdev
    if not significance:
        area_type = 'stable'
    elif significance and net_area_change > 0:
        area_type = 'growing'
    elif significance and net_area_change < 0:
        area_type = 'shrinking'
    return area_type


def dominantGlaciers(glaciers, attr, startdate=None, enddate=None):
    """Identify glaciers that dominate in an attribute since startdate. Dominance defined as greater than two standard deviations of the population mean."""
    final_net_change = finalNetChange(glaciers, attr, startdate, enddate)
    mean_net_change = final_net_change.mean()
    std_net_change = final_net_change.std()
    dominant_threshold_pos = mean_net_change + 2*std_net_change
    dominant_threshold_neg = mean_net_change - 2*std_net_change
    dominant_glacier_pos = final_net_change.where(
        final_net_change > dominant_threshold_pos).dropna()
    dominant_glacier_neg = final_net_change.where(
        final_net_change < dominant_threshold_neg).dropna()
    dominant_glaciers = dominant_glacier_pos.append(dominant_glacier_neg)
    return dominant_glaciers

def filterGlaciers(glaciers, ids, idtype='remove'):
    """Filter glacier dataset by glacier IDs to keep or remove."""
    glacier_dict_copy = glaciers.copy()
    glacier_ids = glaciers.keys()
    if idtype == 'remove':
        remove_ids = ids
    elif idtype == 'keep':
        remove_ids = glacier_ids - ids
    [glacier_dict_copy.pop(id) for id in remove_ids]
    return glacier_dict_copy


def normChangeStats(glaciers, attr, startdate=None, enddate=None):
    """Calculate population mean value of normalized attribute change."""
    glacier_norms = pd.DataFrame(columns=glaciers[list(glaciers.keys())[0]].datayears,
                          index=list(glaciers.keys()))
    for g in glaciers:
        scaled_attr, _ = glaciers[g].normChange(attr, startdate=startdate, enddate=enddate)
        data = scaled_attr.__getattr__(scaled_attr.columns[0])
        glacier_norms.loc[g] = data
    
    norm_mean = glacier_norms.mean()
    norm_std = glacier_norms.std()
    
    return norm_mean, norm_std


def getLinearFit(glacier, attr, startdate=None, enddate=None):
    attrs, dates = glacier.filterDates(attr, startdate, enddate)
    attrs = attrs[~np.isnan(attrs)]
    dates = attrs.index.values
    attrs = [a for item in attrs.values for a in item]
    # Calculate linear regression using scipy.stats
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(dates, attrs)
    r_sq = r_value**2
    return slope, intercept, r_sq, p_value, std_err


def getBreakPoints(glacier, attr, n_segs=3, startdate=None, enddate=None):
    attrs, dates = glacier.filterDates(attr, startdate, enddate)
    attrs = attrs[~np.isnan(attrs)]
    dates = attrs.index.values
    attrs = [a for item in attrs.values for a in item]
    pwlf_fun = pwlf.PiecewiseLinFit(dates, attrs)
    breaks = pwlf_fun.fit(n_segs)
    breaks = [y for y in breaks if y != dates[0]]
    breaks = [y for y in breaks if y != dates[-1]]
    breaks = [round(y) for y in breaks]
    return breaks, pwlf_fun


def f_test(glacier, attr, n_segs=3, startdate=None, enddate=None):
    # Get piecewise fit
    breaks, pwlf_fun = getBreakPoints(glacier, attr, n_segs, startdate, enddate)
    # Get linear fit
    slope, intercept, r_sq, p_value, std_err = getLinearFit(glacier, attr, startdate, enddate)
    # Get dates for x
    attrs, _ = glacier.filterDates(attr, startdate, enddate)
    dates = attrs[~np.isnan(attrs)].index.values
    # Get predicted y values for linear regression
    lin_y = (slope*dates) + intercept
    # Get predicted y values for piecewise regression
    pw_y = pwlf_fun.predict(dates)
    # Get sum of squares error for full (piecewise) and reduced (linear) regressions
    sse_f = np.sum((pw_y - attrs.areas.values) ** 2)
    sse_r = np.sum((lin_y - attrs.areas.values) ** 2)
    # Get degrees of freedom for reduced and full regressions
    dfr = len(dates) - 2
    dff = len(dates) - (n_segs+1)
    # Compute f-statistic
    f = ((sse_r - sse_f)/(dfr - dff)) / (sse_f/dff)
    # Compute p-value
    p = 1 - scipy.stats.f.cdf(f, (dfr-dff), dff)
    # Return f and p values
    return f, p


def avgAnnualChangeRate(glacier, attr, time_bins, n_segs=3, startdate=None, enddate=None):
    breaks, pwlf_fun = getBreakPoints(glacier, attr, n_segs, startdate, enddate)
    attr_fit = pwlf_fun.predict(time_bins)
    # attr_fit = [a for item in attr_fit for a in item]
    rates = np.diff(attr_fit) / np.diff(time_bins)
    return breaks, rates



