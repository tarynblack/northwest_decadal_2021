# Library of plotting functions for climate analysis

import sys

from numpy.lib.histograms import _get_bin_edges, histogramdd
from pandas.core.groupby.generic import NamedAgg
sys.path.append('/mnt/e/')

import matplotlib.pyplot as plt
import icetools.climate as clm
import icetools.plotting.plotutils as plu
import pandas as pd


def monthCycler(months, start):
    cycled_months = [(i-(start-1) if i>=start else (i+(12-start+1))) for i in months]
    return cycled_months


def subplotGridLabels(fig, title, xlabel, ylabel):
    fig.add_subplot(111, frameon=False)
    plt.grid(None)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def getColor(name):
    colors = {'ecco': '#0073FF', #blue
              'icesdk': '#73FF00', #green
              'hadley': '#FF8C00', #orange
              'noaa': '#8C00FF', #purple
              'default': 'royalblue'}
    color = colors[name]
    return color

def getMarker(name):
    markers = {'ecco': 'o',
               'icesdk': '^',
               'hadley': 's',
               'noaa': 'v',
               'default': '.'}


def subSurfaceTemperatureAnomaly(ax, depth, coordinate, ecco=None, ecco_dtype='ecco5', icesdk=None, dec_mean=False, spgrid=False, style='pub-jog'):
    if ecco is not None:
        ecco_depth_point = clm.selectPointData(ecco, dtype=ecco_dtype, depth=depth, coordinate=coordinate)
        ecco_mean = ecco_depth_point.temperature.mean().values
        ecco_anomaly = ecco_depth_point.temperature.values - ecco_mean
        time = ecco.temperature.time.values
        eag, = ax.plot(time, ecco_anomaly, marker=getMarker('ecco'), color=getColor('ecco'), alpha=0.7, label='ECCO annual mean')
        ax.annotate(text='ECCO mean={:.2f} $^oC$'.format(ecco_mean), xy=(0.05, 0.9), xycoords='axes fraction')
        # plu.designProperties(ax, [eag], style)
        if dec_mean is True:
            df = pd.DataFrame(data={'anomaly': ecco_anomaly}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            ecco_df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(ecco_df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(ecco_df_decade.index.values + 10, format='%Y')
            edg = ax.hlines(ecco_df_decade.anomaly.values, xmin=decade_starts, xmax=decade_ends, color=getColor('ecco'), linewidth=5, alpha=0.5, zorder=2, label='ECCO decadal mean')
            # plu.designProperties(ax, [edg], style)
    else:
        eag, = ax.plot(0,0, visible=False)
        edg, = ax.plot(0,0, visible=False)
        ecco_df_decade = []
    if icesdk is not None:
        icesdk_depth_point = clm.selectPointData(icesdk, dtype='icesdk', depth=depth, depth_tolerance=25, coordinate=coordinate, coordinate_tolerance=0.51)
        icesdk_mean = clm.icesAnnualMean(icesdk_depth_point)
        icesdk_anomaly = icesdk_mean.avg.values - icesdk_mean.avg.mean()
        time = pd.to_datetime(icesdk_mean.index.values, format='%Y')
        iag, = ax.plot(time, icesdk_anomaly, marker=getMarker('icesdk'), color=getColor('icesdk'), alpha=0.7, label='ICES annual mean')
        ax.annotate(text='ICES mean={:.2f} $^oC$'.format(icesdk_mean.avg.mean()), xy=(0.05, 0.8), xycoords='axes fraction')
        # plu.designProperties(ax, [iag], style)
        if dec_mean is True:
            ices_decadal_mean = clm.icesDecadalMean(icesdk_depth_point)
            ices_decadal_anomaly = ices_decadal_mean.avg.values - ices_decadal_mean.avg.mean()
            decade_starts = pd.to_datetime(ices_decadal_mean.index, format='%Y')
            decade_ends = pd.to_datetime(ices_decadal_mean.index+10, format='%Y')
            ices_df_decade = pd.DataFrame(data={'decade': decade_starts, 'anomaly': ices_decadal_anomaly})
            idg = ax.hlines(ices_decadal_anomaly, xmin=decade_starts, xmax=decade_ends, color=getColor('icesdk'), linewidth=5, alpha=0.5, zorder=2, label='ICES decadal mean')
            # plu.designProperties(ax, [idg], style)
    else:
        iag, = ax.plot(0,0, visible=False)
        idg, = ax.plot(0,0, visible=False)
        ices_df_decade = []
    if spgrid is True:
        ax.set_title('{} N, {} W'.format(coordinate[0], -coordinate[1]))
        # ax.legend()
    elif spgrid is False:
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature anomaly ($^oC$)')
        ax.set_title('Ocean Temperature Anomaly at {}m ({} N, {} W)'.format(depth, coordinate[0], -coordinate[1]))
    # plu.designProperties(ax, [], style)
    return eag, edg, iag, idg, ecco_df_decade, ices_df_decade


def seaSurfaceTemperatureAnomaly(ax, coordinate, ecco=None, ecco_dtype='ecco5', hadley=None, dec_mean=False, spgrid=False, style='pub-jog'):
    if ecco is not None:
        ecco_sfc_point = clm.selectPointData(ecco, dtype=ecco_dtype, depth=0, coordinate=coordinate)
        ecco_mean = ecco_sfc_point.temperature.mean().values
        ecco_anomaly = ecco_sfc_point.temperature.values - ecco_mean
        time = ecco.temperature.time.values
        eag, = ax.plot(time, ecco_anomaly, marker=getMarker('ecco'), color=getColor('ecco'), alpha=0.7, label='ECCO annual mean')
        ax.annotate(text='ECCO mean={:.2f} $^oC$'.format(ecco_mean), xy=(0.05, 0.9), xycoords='axes fraction')
        # plu.designProperties(ax, [eag], style)
        if dec_mean is True:
            df = pd.DataFrame(data={'anomaly': ecco_anomaly}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            ecco_df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(ecco_df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(ecco_df_decade.index.values + 10, format='%Y')
            edg = ax.hlines(ecco_df_decade.anomaly.values, xmin=decade_starts, xmax=decade_ends, color=getColor('ecco'), linewidth=5, alpha=0.5, zorder=2, label='ECCO decadal mean')
            # plu.designProperties(ax, [edg], style)
    else:
        eag, = ax.plot(0,0, visible=False)
        edg, = ax.plot(0,0, visible=False)
        ecco_df_decade = []
    if hadley is not None:
        hadley_sfc_point = clm.selectPointData(hadley, dtype='hadley', coordinate=coordinate)
        hadley_mean = hadley_sfc_point.SST.mean().values
        hadley_anomaly = hadley_sfc_point.SST.values - hadley_mean
        time = hadley.SST.time.values
        hag, = ax.plot(time, hadley_anomaly, marker=getMarker('hadley'), color=getColor('hadley'), alpha=0.7, label='Hadley-OI annual mean')
        ax.annotate(text='Hadley-OI mean={:.2f} $^oC$'.format(hadley_mean), xy=(0.05, 0.8), xycoords='axes fraction')
        # plu.designProperties(ax, [hag], style)
        if dec_mean is True:
            df = pd.DataFrame(data={'anomaly': hadley_anomaly}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            hadley_df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(hadley_df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(hadley_df_decade.index.values + 10, format='%Y')
            hdg = ax.hlines(hadley_df_decade.anomaly.values, xmin=decade_starts, xmax=decade_ends, color=getColor('hadley'), linewidth=5, alpha=0.5, zorder=2, label='Hadley-OI decadal mean')
            # plu.designProperties(ax, [hdg], style)
    else:
        hag, = ax.plot(0,0, visible=False)
        hdg, = ax.plot(0,0, visible=False)
        hadley_df_decade = []
    if spgrid is True:
        ax.set_title('{} N, {} W'.format(coordinate[0], -coordinate[1]))
    elif spgrid is False:
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature anomaly ($^oC$)')
        ax.set_title('Sea Surface Temperature Anomaly ({} N, {} W)'.format(coordinate[0], -coordinate[1]))
    # plu.designProperties(ax, [], style)
    return eag, edg, hag, hdg, ecco_df_decade, hadley_df_decade


def seaIceConcentration(ax, coordinate, season_start, ecco=None, hadley=None, noaa=None, noaa_var=None, dec_mean=False, spgrid=False, style='pub-jog'):
    if ecco is not None:
        ecco_coord = clm.dataAtCoord(ecco, coordinate, dtype='ecco4')
        ecco_season = clm.seasonMean(ecco_coord, 'SIarea', season_start)
        ecco_sic = 100 * ecco_season.values
        time = ecco_season.time.values
        eg, = ax.plot(time, ecco_sic, '.-', color=getColor('ecco'), alpha=0.7, label='ECCO seasonal mean')
        # plu.designProperties(ax, [graph], style)
    else: eg, = ax.plot(0,0, visible=False)
    if hadley is not None:
        hadley_coord = clm.dataAtCoord(hadley, coordinate, dtype='hadley')
        hadley_season = clm.seasonMean(hadley_coord, 'SEAICE', season_start)
        hadley_sic = hadley_season.values
        time = hadley_season.time.values
        hag, = ax.plot(time, hadley_sic, marker=getMarker('hadley'), color=getColor('hadley'), alpha=0.7, label='Hadley-OI seasonal mean')
        # plu.designProperties(ax, [graph], style)
        if dec_mean is True:
            df = pd.DataFrame(data={'sic': hadley_sic}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(df_decade.index.values + 10, format='%Y')
            # decadal_hadley_sic = df.resample('10AS', loffset='-2A').mean()
            # decade_starts = pd.to_datetime(decadal_hadley_sic.index.year, format='%Y')
            # decade_ends = pd.to_datetime(decadal_hadley_sic.index.year+10, format='%Y')
            hdg = ax.hlines(df_decade.sic.values, xmin=decade_starts, xmax=decade_ends, color=getColor('hadley'), linewidth=5, alpha=0.5, zorder=2, label='Hadley-OI decadal seasonal mean')
            # plu.designProperties(ax, [hdg], style)
    else: 
        hag, = ax.plot(0,0, visible=False)
        hdg, = ax.plot(0,0, visible=False)
    if noaa is not None and noaa_var is not None:
        noaa_coord = clm.dataAtCoord(noaa, coordinate, dtype='noaa')
        noaa_season = clm.seasonMean(noaa_coord, noaa_var, season_start)
        noaa_sic = 100 * noaa_season.values
        time = noaa_season.time.values
        nag, = ax.plot(time, noaa_sic, marker=getMarker('noaa'), color=getColor('noaa'), alpha=0.7, label='NOAA seasonal mean')
        # plu.designProperties(ax, [graph], style)
        if dec_mean is True:
            df = pd.DataFrame(data={'sic': noaa_sic}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(df_decade.index.values + 10, format='%Y')
            # decadal_noaa_sic = df.resample('10AS', loffset='+1A').mean()
            # decade_starts = pd.to_datetime(decadal_noaa_sic.index.year, format='%Y')
            # decade_ends = pd.to_datetime(decadal_noaa_sic.index.year+10, format='%Y')
            ndg = ax.hlines(df_decade.sic.values, xmin=decade_starts, xmax=decade_ends, color=getColor('noaa'), linewidth=5, alpha=0.5, zorder=2, label='NOAA decadal seasonal mean')
            # plu.designProperties(ax, [ndg], style)
    else: 
        nag, = ax.plot(0,0, visible=False)
        ndg, = ax.plot(0,0, visible=False)
    if spgrid is True:
        ax.set_title('{} N, {} W'.format(coordinate[0], -coordinate[1]))
    elif spgrid is False:
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('% concentration')
        ax.set_title('Sea Ice Concentration ({} N, {} W)'.format(coordinate[0], -coordinate[1]))
    # plu.designProperties(ax, [eg, hag, hdg, nag, ndg], style)
    return eg, hag, hdg, nag, ndg


def seaIceSeason(ax, coordinate, sic_minimum, ecco=None, hadley=None, noaa=None, noaa_var=None, spgrid=False, style='pub-jog'):
    if ecco is not None:
        try:
            ecco_ice_season_length = clm.seaIceSeasonLength(ecco, 'SIarea', dtype='ecco4', minimum=sic_minimum, coordinate=coordinate)
            ecco_ice_onset = clm.seaIceSeasonOnset(ecco, 'SIarea', dtype='ecco5', minimum=sic_minimum, coordinate=coordinate)
            months = [d.month for d in pd.to_datetime(ecco_ice_onset.values)][:-1]
            months = [m+12 if m<9 else m for m in months]
            lengths = ecco_ice_season_length.values[:-1]
            years = pd.to_datetime(ecco_ice_season_length.time.values).year.values[:-1]
            graph = ax.vlines(years-0.25, ymin=months, ymax=months+lengths-1, color=getColor('ecco'), alpha=0.8, label='ECCO')
            # plu.designProperties(ax, [graph], style)
        except: None
    if hadley is not None:
        try:
            hadley_ice_season_length = clm.seaIceSeasonLength(hadley, 'SEAICE', dtype='hadley', minimum=sic_minimum, coordinate=coordinate)
            hadley_ice_onset = clm.seaIceSeasonOnset(hadley, 'SEAICE', dtype='hadley', minimum=sic_minimum, coordinate=coordinate)
            months = [d.month for d in pd.to_datetime(hadley_ice_onset.values)][:-1]
            months = [m+12 if m<9 else m for m in months]
            lengths = hadley_ice_season_length.values[:-1]
            years = pd.to_datetime(hadley_ice_season_length.time.values).year.values[:-1]
            graph = ax.vlines(years, ymin=months, ymax=months+lengths-1, color=getColor('hadley'), alpha=0.8, label='Hadley-OI')
            # plu.designProperties(ax, [graph], style)
        except: None
    if noaa is not None and noaa_var is not None:
        try:
            noaa_ice_season_length = clm.seaIceSeasonLength(noaa, noaa_var, dtype='noaa', minimum=sic_minimum, coordinate=coordinate)
            noaa_ice_onset = clm.seaIceSeasonOnset(noaa, noaa_var, dtype='noaa', minimum=sic_minimum, coordinate=coordinate)
            months = [d.month for d in pd.to_datetime(noaa_ice_onset.values)][:-1]
            months = [m+12 if m<9 else m for m in months]
            lengths = noaa_ice_season_length.values[:-1]
            years = pd.to_datetime(noaa_ice_season_length.time.values).year.values[:-1]
            graph = ax.vlines(years+0.25, ymin=months, ymax=months+lengths-1, color=getColor('noaa'), alpha=0.8, label='NOAA')
            # plu.designProperties(ax, [graph], style)
        except: None
    ax.set_yticks([9,11,13,15,17,19,21])
    ax.set_yticklabels(['Sep', 'Nov', 'Jan', 'Mar', 'May', 'Jul', 'Sep'])
    if spgrid is True:
        ax.set_title('{} N, {} W'.format(coordinate[0], -coordinate[1]))
    elif spgrid is False:
        ax.legend()
        ax.set_xlabel('Years')
        ax.set_ylabel('Months')
        ax.set_title('Sea Ice Season Onset and Length ({} N, {} W)'.format(coordinate[0], -coordinate[1]))
    # plu.designProperties(ax, [graph], style)
    

def seaIceSeasonLength(ax, coordinate, sic_minimum, ecco=None, hadley=None, noaa=None, noaa_var=None, dec_mean=False, spgrid=False, style='pub-jog'):
    if ecco is not None:
        ecco_ice_season_length = clm.seaIceSeasonLength(ecco, 'SIarea', dtype='ecco4', minimum=sic_minimum, coordinate=coordinate)
        ecco_time = ecco_ice_season_length.time.values[:-1]
        ecco_vals = ecco_ice_season_length.values[:-1]
        eg, = ax.plot(ecco_time, ecco_vals, '.-', color=getColor('ecco'), alpha=0.7, label='ECCO')
        # plu.designProperties(ax, [graph], style)
        # ax.bar(ecco_time-width, ecco_vals, width, color=getColor('ecco'), label='ECCO')
        ecco_df = pd.DataFrame(data=ecco_vals, index=ecco_time)
    else: 
        ecco_df = None
        eg, = ax.plot(0,0, visible=False)
    if hadley is not None:
        hadley_ice_season_length = clm.seaIceSeasonLength(hadley, 'SEAICE', dtype='hadley', minimum=sic_minimum, coordinate=coordinate)
        time = hadley_ice_season_length.time.values[:-1]
        lengths = hadley_ice_season_length.values[:-1]
        hag, = ax.plot(time, lengths, marker=getMarker('hadley'), color=getColor('hadley'), alpha=0.7, label='Hadley-OI annual mean')
        # plu.designProperties(ax, [graph], style)
        # ax.bar(hadley_time, hadley_vals, width, color=getColor('hadley'), label='Hadley-OI')
        if dec_mean is True:
            df = pd.DataFrame(data={'lengths': lengths}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(df_decade.index.values + 10, format='%Y')
            # decadal_hadley_length = hadley_df.resample('10AS', loffset='-2A').mean()
            # decade_starts = pd.to_datetime(decadal_hadley_length.index.year, format='%Y')
            # decade_ends = pd.to_datetime(decadal_hadley_length.index.year+10, format='%Y')
            hdg = ax.hlines(df_decade.lengths.values, xmin=decade_starts, xmax=decade_ends, color=getColor('hadley'), linewidth=5, alpha=0.5, zorder=2, label='Hadley-OI decadal mean')
            # plu.designProperties(ax, [hdg], style)
    else: 
        hag, = ax.plot(0,0, visible=False)
        hdg, = ax.plot(0,0, visible=False)
    if noaa is not None and noaa_var is not None:
        noaa_ice_season_length = clm.seaIceSeasonLength(noaa, noaa_var, dtype='noaa', minimum=sic_minimum, coordinate=coordinate)
        time = noaa_ice_season_length.time.values[:-1]
        lengths = noaa_ice_season_length.values[:-1]
        nag, = ax.plot(time, lengths, marker=getMarker('noaa'), color=getColor('noaa'), alpha=0.7, label='NOAA annual mean')
        # plu.designProperties(ax, [graph], style)
        # ax.bar(noaa_time+width, noaa_vals, width, color=getColor('noaa'), label='NOAA')
        if dec_mean is True:
            df = pd.DataFrame(data={'lengths': lengths}, index=time)
            df['decade'] = df.index.year - (df.index.year % 10)
            df_decade = df.groupby('decade').mean()
            decade_starts = pd.to_datetime(df_decade.index.values, format='%Y')
            decade_ends = pd.to_datetime(df_decade.index.values + 10, format='%Y')
            # decadal_noaa_length = noaa_df.resample('10AS', loffset='+1A').mean()
            # decade_starts = pd.to_datetime(decadal_noaa_length.index.year, format='%Y')
            # decade_ends = pd.to_datetime(decadal_noaa_length.index.year+10, format='%Y')
            ndg = ax.hlines(df_decade.lengths.values, xmin=decade_starts, xmax=decade_ends, color=getColor('noaa'), linewidth=5, alpha=0.5, zorder=2, label='NOAA decadal mean')
            # plu.designProperties(ax, [ndg], style)
    else: 
        nag, = ax.plot(0,0, visible=False)
        ndg, = ax.plot(0,0, visible=False)
    if spgrid is True:
        ax.set_title('{} N, {} W'.format(coordinate[0], -coordinate[1]))
    elif spgrid is False:
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Months')
        ax.set_title('Duration of sea ice season ({} N, {} W)'.format(coordinate[0], -coordinate[1]))
    # plu.designProperties(ax, [eg, hag, nag], style)
    return eg, hag, hdg, nag, ndg


def seaIceSeasonOnset(ax, coordinate, sic_minimum, ecco=None, hadley=None, noaa=None, noaa_var=None, spgrid=False, style='pub-jog'):
    shift = 0.15
    if ecco is not None:
        ecco_ice_onset = clm.seaIceSeasonOnset(ecco, 'SIarea', dtype='ecco4', minimum=sic_minimum, coordinate=coordinate)
        ecco_years = pd.to_datetime(ecco_ice_onset.values).year
        ecco_months = pd.to_datetime(ecco_ice_onset.values).month
        ecco_years = pd.to_datetime(
            [ecco_years[yi]-1 if ecco_months[yi]<9 else 
            ecco_years[yi] for yi in range(len(ecco_years))], 
            format='%Y')
        ecco_months = [m+12-shift if m<9 else m-shift for m in ecco_months]
        eg = ax.scatter(ecco_years, ecco_months, marker='o', color=getColor('ecco'), alpha=0.7, edgecolor=None, label='ECCO')
        # plu.designProperties(ax, [graph], style)
    else: eg, = ax.plot(0,0, visible=False)
    if hadley is not None:
        hadley_ice_onset = clm.seaIceSeasonOnset(hadley, 'SEAICE', dtype='hadley', minimum=sic_minimum, coordinate=coordinate)
        hadley_years = pd.to_datetime(hadley_ice_onset.values).year
        hadley_months = pd.to_datetime(hadley_ice_onset.values).month
        hadley_years = pd.to_datetime(
            [hadley_years[yi]-1 if hadley_months[yi]<9 else 
            hadley_years[yi] for yi in range(len(hadley_years))], 
            format='%Y')
        hadley_months = [m+12 if m<9 else m for m in hadley_months]
        hg = ax.scatter(hadley_years, hadley_months, marker='o', color=getColor('hadley'), alpha=0.7, edgecolor=None, label='Hadley-OI')
        # plu.designProperties(ax, [graph], style)
    else: hg, = ax.plot(0,0, visible=False)
    if noaa is not None and noaa_var is not None:
        noaa_ice_onset = clm.seaIceSeasonOnset(noaa, noaa_var, dtype='noaa', minimum=sic_minimum, coordinate=coordinate)
        noaa_years = pd.to_datetime(noaa_ice_onset.values).year
        noaa_months = pd.to_datetime(noaa_ice_onset.values).month
        noaa_years = pd.to_datetime(
            [noaa_years[yi]-1 if noaa_months[yi]<9 else 
            noaa_years[yi] for yi in range(len(noaa_years))], 
            format='%Y')
        noaa_months = [m+12+shift if m<9 else m+shift for m in noaa_months]
        ng = ax.scatter(noaa_years, noaa_months, marker='o', color=getColor('noaa'), alpha=0.7, edgecolor=None, label='NOAA')
        # plu.designProperties(ax, [graph], style)
    else: ng, = ax.plot(0,0, visible=False)
    ax.set_yticks([9,11,13,15,17,19,21])
    ax.set_ylim(bottom=8, top=21)
    ax.set_yticklabels(['Sep', 'Nov', 'Jan', 'Mar', 'May', 'Jul', 'Sep'])
    if spgrid is True:
        ax.set_title('{} N, {} W'.format(coordinate[0], -coordinate[1]))
    elif spgrid is False:
        ax.legend()
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')
        ax.set_title('Onset of sea ice formation ({} N, {} W)'.format(coordinate[0], -coordinate[1]))
    # plu.designProperties(ax, [eg, hg, ng], style)
    return eg, hg, ng


def marAverage(ax, data, dvar, spgrid=False, style='pub-jog'):
    dvars = {'SMB': 'Surface Mass Balance',
             'RU':  'Runoff',
             'SF':  'Snowfall',
             'RF':  'Rainfall',
             'ME':  'Meltwater Production'}
    time = data[dvar].TIME.values
    vals = clm.mmweday2myr(data[dvar].values)
    graph, = ax.plot(time, vals, color=getColor('default'))
    if spgrid is True:
        ax.set_title('{}'.format('[region, TBD]'))
    elif spgrid is False:
        ax.set_xlabel('Year')
        ax.set_ylabel('{} (m/yr)'.format(dvars[dvar]))
        ax.set_title('{} at {}'.format(dvars[dvar], '[region, TBD]'))
    # plu.designProperties(ax, [graph], style)


def marAnomaly(ax, data, dvar, spgrid=False, style='pub-jog'):
    dvars = {'SMB': 'Surface Mass Balance',
             'RU':  'Runoff',
             'SF':  'Snowfall',
             'RF':  'Rainfall',
             'ME':  'Meltwater Production'}
    time = data[dvar].TIME.values
    anomaly = data[dvar].values - data[dvar].mean().values
    anomaly = clm.mmweday2myr(anomaly)
    graph, = ax.plot(time, anomaly, color=getColor('default'))
    if spgrid is True:
        ax.set_title('{}'.format('[region, TBD]'))
    elif spgrid is False:
        ax.set_xlabel('Year')
        ax.set_ylabel('{} anomaly (m/yr)'.format(dvars[dvar]))
        ax.set_title('{} Anomaly at {}'.format(dvars[dvar], '[region, TBD]'))
    # plu.designProperties(ax, [graph], style)


def marAnomalyBulk(ax, data, coords, dvar, individual=False, ann_mean=False, error=False, dec_mean=False):
    dvars = {'SMB': 'Surface mass balance',
             'RU':  'Runoff',
             'SF':  'Snowfall',
             'RF':  'Rainfall',
             'ME':  'Meltwater production'}
    
    # bulk_anomaly = pd.DataFrame(
        # columns=pd.to_datetime(data.TIME.values), 
        # index=range(len(coords)))
    bulk_anomaly = clm.bulkAnomalyMAR(data, coords, dvar)
    time = bulk_anomaly.keys()
      
    # plot annual mean variable for each coordinate (nearest on grid)
    if individual is True:
        for c in range(len(coords)):
            anomaly = bulk_anomaly.loc[c].values
            time = bulk_anomaly.loc[c].index
            ax.plot(time, anomaly, color='silver', alpha=0.3, zorder=1, label='individual glaciers')
    # plot overall annual mean(+std) and decadal means
    if ann_mean is True:
        bulk_mean = clm.bulkMeanMAR(data, coords, dvar)
        ax.plot(time, bulk_anomaly.mean(), '.-', color=getColor('default'), zorder=3, label='annual anomaly')
        bulk_anomaly.mean().mean()
        ax.annotate(text='mean={:.2f} m/yr'.format(bulk_mean), xy=(0.05, 0.9), xycoords='axes fraction')
    if error is True:
        ax.fill_between(time, bulk_anomaly.mean()-bulk_anomaly.std(), bulk_anomaly.mean()+bulk_anomaly.std(), color=getColor('default'), edgecolor=None, alpha=0.2, label='annual std')
    if dec_mean is True:
        decadal_mean_anomaly = clm.decadalAnomalyMAR(data, coords, dvar)
        decade_starts = pd.to_datetime(decadal_mean_anomaly.index.year[:-1], format='%Y')
        decade_ends = pd.to_datetime(decadal_mean_anomaly.index.year[:-1]+10, format='%Y')
        ax.hlines(decadal_mean_anomaly[:-1], xmin=decade_starts, xmax=decade_ends, color=getColor('default'), linewidth=5, alpha=0.6, zorder=4, label='decadal mean anomaly')

    ax.set_xlabel('Year')
    ax.set_ylabel('Anomaly (m a$^{-1}$)')
    ax.set_title('{}'.format(dvars[dvar]))


def marMeltDaysAnomaly(ax, data, coords, individual=False, ann_mean=False, error=False, dec_mean=False):
    bulk_melt = pd.DataFrame(
        columns=pd.to_datetime(data.resample(TIME='AS').count().TIME.values), 
        index=range(len(coords)))
    bulk_melt_anomaly = pd.DataFrame(
        columns=pd.to_datetime(data.resample(TIME='AS').count().TIME.values),
        index=range(len(coords)))

    for c in range(len(coords)):
        coord = coords[c]
        # -- get annual number of melt days
        daily_melt_coord = clm.dataAtCoord(data, coord, dtype='mar').ME
        annual_melt_days = daily_melt_coord.where(daily_melt_coord != 0).resample(TIME='AS').count(dim='TIME')
        bulk_melt.loc[c] = annual_melt_days.values.flatten()
        # -- calculate melt days anomaly
        anomaly = bulk_melt.loc[c] - bulk_melt.loc[c].mean()
        bulk_melt_anomaly.loc[c] = anomaly
    
    time = bulk_melt_anomaly.loc[c].index
    
    if individual is True:
        for c in range(len(coords)):
            annual_melt_anomaly = bulk_melt_anomaly.loc[c].values
            ig = ax.plot(time, annual_melt_anomaly, color='silver', alpha=0.3, zorder=1, label='individual glaciers')
    else: ig, = ax.plot(0,0, visible=False)
    if ann_mean is True:
        bulk_mean = bulk_melt.mean(axis=1).mean()
        am, = ax.plot(time, bulk_melt_anomaly.mean(), '.-', color=getColor('default'), zorder=3, label='annual anomaly')
        ax.annotate(text='mean={:.0f} days'.format(bulk_mean), xy=(0.05, 0.9), xycoords='axes fraction')
    else: am, = ax.plot(0, 0, visible=False)
    if error is True:
        er = ax.fill_between(time, bulk_melt_anomaly.mean()-bulk_melt_anomaly.std(), bulk_melt_anomaly.mean()+bulk_melt_anomaly.std(), color=getColor('default'), edgecolor=None, alpha=0.2, label='annual std')
    else: er, = ax.plot(0,0, visible=False)
    if dec_mean is True:
        decadal_mean_melt = bulk_melt_anomaly.mean()[1:].resample('10AS').mean()
        decade_starts = pd.to_datetime(decadal_mean_melt.index.year[:-1], format='%Y')
        decade_ends = pd.to_datetime(decadal_mean_melt.index.year[:-1]+10, format='%Y')
        dm = ax.hlines(decadal_mean_melt[:-1], xmin=decade_starts, xmax=decade_ends, color=getColor('default'), linewidth=5, alpha=0.6, zorder=2, label='decadal mean anomaly')
    else: dm, = ax.plot(0,0, visible=False)

    ax.set_xlabel('Year')
    ax.set_ylabel('Anomaly (days)')
    ax.set_title('Melt days')

    return ig, er, am, dm, decadal_mean_melt


def marRunoffDaysAnomaly(ax, data, coords, individual=False, ann_mean=False, error=False, dec_mean=False):
    bulk_runoff = pd.DataFrame(
        columns=pd.to_datetime(data.resample(TIME='AS').count().TIME.values), 
        index=range(len(coords)))
    bulk_runoff_anomaly = pd.DataFrame(
        columns=pd.to_datetime(data.resample(TIME='AS').count().TIME.values),
        index=range(len(coords)))

    for c in range(len(coords)):
        coord = coords[c]
        # -- get annual number of runoff days
        daily_runoff_coord = clm.dataAtCoord(data, coord, dtype='mar').RU
        annual_runoff_days = daily_runoff_coord.where(daily_runoff_coord != 0).resample(TIME='AS').count(dim='TIME')
        bulk_runoff.loc[c] = annual_runoff_days.values.flatten()
        # -- calculate runoff days anomaly
        anomaly = bulk_runoff.loc[c] - bulk_runoff.loc[c].mean()
        bulk_runoff_anomaly.loc[c] = anomaly
    
    time = bulk_runoff_anomaly.loc[c].index
        # daily_runoff_mask = daily_runoff_coord.where(daily_runoff_coord==0,  other=1)
        # annual_runoff_days = daily_runoff_mask.resample(TIME='AS').sum()
        # total_mean = annual_runoff_days.mean().values
        # annual_runoff_anomaly = annual_runoff_days.values - total_mean
        # time = annual_runoff_days.TIME.values
    if individual is True:
        for c in range(len(coords)):
            annual_runoff_anomaly = bulk_runoff_anomaly.loc[c].values
            ig, = ax.plot(time, annual_runoff_anomaly, color='silver', alpha=0.3, zorder=1, label='individual glaciers')
    else: ig, = ax.plot(0,0, visible=False)
        # bulk_runoff.loc[c] = annual_runoff_anomaly.flatten()
    
    if ann_mean is True:
        bulk_mean = bulk_runoff.mean(axis=1).mean()
        am, = ax.plot(time, bulk_runoff_anomaly.mean(), '.-', color=getColor('default'), zorder=3, label='annual anomaly')
        ax.annotate(text='mean={:.0f} days'.format(bulk_mean), xy=(0.05, 0.9), xycoords='axes fraction')
    else: am, = ax.plot(0, 0, visible=False)
    if error is True:
        er = ax.fill_between(time, bulk_runoff_anomaly.mean()-bulk_runoff_anomaly.std(), bulk_runoff_anomaly.mean()+bulk_runoff_anomaly.std(), color=getColor('default'), edgecolor=None, alpha=0.2, label='annual std')
    else: er, = ax.plot(0,0, visible=False)
    if dec_mean is True:
        decadal_mean_runoff = bulk_runoff_anomaly.mean()[1:].resample('10AS').mean()
        decade_starts = pd.to_datetime(decadal_mean_runoff.index.year[:-1], format='%Y')
        decade_ends = pd.to_datetime(decadal_mean_runoff.index.year[:-1]+10, format='%Y')
        dm = ax.hlines(decadal_mean_runoff[:-1], xmin=decade_starts, xmax=decade_ends, color=getColor('default'), linewidth=5, alpha=0.6, zorder=2, label='decadal mean anomaly')
    else: dm, = ax.plot(0,0, visible=False)

    ax.set_xlabel('Year')
    ax.set_ylabel('Anomaly (days)')
    ax.set_title('Runoff days')

    return ig, er, am, dm, decadal_mean_runoff
