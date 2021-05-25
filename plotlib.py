# Functions for creating various glacier data plots.

import sys
sys.path.append('/mnt/e/')

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import icetools.metrics as met
import icetools.plotting.plotutils as plu
import pwlf
import random
import numpy as np

# Design parameters
attr_units = {'lengths'         : 'km',
              'areas'           : '$km^2$',
              'termareas'       : '$km^2$',
              'interplengths'   : 'km',
              'interpareas'     : '$km^2$',
              'interptermareas' : '$km^2$'}

attr_names = {'lengths'         : 'Length',
              'areas'           : 'Area',
              'termareas'       : 'Terminus Area',
              'interplengths'   : 'Length',
              'interpareas'     : 'Area',
              'interptermareas' : 'Terminus Area'}


def getGlacierStyles(GIDS):
    glacier_cmap = mpl.cm.get_cmap('rainbow', len(GIDS))
    base_markers = ['o', 'v', '^', 's', 'X', 'D']
    # random_list = random.sample(list(np.arange(max(GIDS)+1)), max(GIDS)+1)
    # glacier_colors = {key: glacier_cmap(random_list[key]) for key in sorted(dict.fromkeys(GIDS))}
    glacier_colors = {key: glacier_cmap(key) for key in sorted(dict.fromkeys(GIDS))}
    glacier_markers = {key: base_markers[key % len(base_markers)] for key in sorted(dict.fromkeys(GIDS))}
    glacier_styles = {key: {'m': glacier_markers[key], 'c': glacier_colors[key]} for key in glacier_colors}
    return glacier_styles


def getColor(name):
    colors = {'default': 'royalblue'}
    color = colors[name]
    return color


def getCmap(name):
    cmaps = {'default': 'viridis'}
    cmap = cmaps[name]
    return cmap


def checkAttribute(attr):
    attribute_types = ['lengths',
                       'areas',
                       'termareas',
                       'interplengths',
                       'interpareas',
                       'interptermareas']
    if attr not in attribute_types:
        raise ValueError('Invalid attribute type. Expected one of: {}'.format(
            attribute_types))


def alignYScale(ax1, ax2):
    ax1min, ax1max = ax1.get_ylim()
    ax2min, ax2max = ax2.get_ylim()
    axmin = min(ax1min, ax2min)
    axmax = max(ax1max, ax2max)
    ax1.set_ylim(axmin, axmax)
    ax2.set_ylim(axmin, axmax)


def annotateBars(ax, anno, x, y):
    for i in range(len(anno)):
        sign = y[i]/abs(y[i])
        ax.annotate(anno[i],
                    xy=(x[i], 0),#y[i]), 
                    xytext = (0, -sign * 8),
                    textcoords = 'offset points',
                    ha='center', va='center')


def pickTimeLabel(glacier, attr):
    checkAttribute(attr)
    if attr in ['lengths', 'areas', 'termareas']:
        # time = glacier.dates
        timelabel = 'Date'
    elif attr in ['interplengths', 'interpareas', 'interptermareas']:
        # time = glacier.datayears
        timelabel = 'Hydrological Year'
    return timelabel


# Plots

def annualObservations(ax, glaciers, years, show_firstyear=True, style='pub-jog'):
    for g in glaciers:
        glacier = glaciers[g]
        
        obsv = ax.scatter([glacier.gid]*len(glacier.hydroyears), glacier.hydroyears, \
            marker='o', c=glacier.daysofhydroyear, cmap='twilight_shifted', label='observed')
        intp = ax.scatter([glacier.gid]*len(glacier.interpyears), glacier.interpyears, \
            marker='o', edgecolors='gray', facecolors='none', label='interpolated')
        
        # plu.designProperties(ax, obsv, style)
    
    if show_firstyear:
        first_full_year = met.firstFullYear(glaciers)
        ax.axhline(y=first_full_year, linewidth=3.0, alpha=0.5, color='red', zorder=0.5)
    
    ax.set_title('Observation time series')
    ax.set_xlabel('Glacier ID')
    ax.set_ylabel('Hydrological year')
    ax.set_ylim(bottom=years[0]-1, top=years[-1]+1)

    # Add colorbar with season labels
    cbar = plt.colorbar(obsv, label='Day of hydrological year', values=list(range(0, 366)))
    tick_locator = mpl.ticker.LinearLocator(numticks=9)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.set_yticklabels(
        ['', 'Autumn', '', 'Winter', '', 'Spring', '', 'Summer', ''], 
        rotation=90, verticalalignment='center')
    
    # Add legend for observed/interpolated points
    ax.figure.legend(
        handles=[obsv, intp], ncol=2, 
        loc='upper center', bbox_to_anchor=(0.45, 0.0))

    # plu.designProperties(ax, graph1)
    # plu.designProperties(ax, [], style)


def dateObservations(ax, glaciers, gids, years, xax='glacier', style='pub-jog'):
    count = 0
    for g in glaciers:
        glacier = glaciers[g]
        if xax not in ['glacier', 'date']:
            raise ValueError('xax must be \'glacier\' or \'date\'')
        if xax == 'glacier':
            graph = ax.scatter([count]*len(glacier.dates), glacier.dates, c=getColor('default'), marker='.')
        elif xax == 'date':
            graph = ax.scatter(glacier.dates, [count]*len(glacier.dates), c=getColor('default'), marker='.')
        # plu.designProperties(ax, [graph], style)
        count = count+1

    ax.set_title('Observation Time Series')
    if xax == 'glacier':
        ax.set_xlabel('Glacier ID')
        ax.set_ylabel('Date')
        ax.set_xticks(range(len(gids)), gids, rotation=90)
        ax.set_ylim(bottom=pd.to_datetime(years[0], format='%Y'), \
                top=pd.to_datetime(years[-1]+1, format='%Y'))
    elif xax == 'date':
        ax.set_xlabel('Date')
        ax.set_ylabel('Glacier ID')
        ax.set_yticks(range(len(gids)), gids)
        ax.set_ylim(bottom=0, top=len(gids))
        ax.set_xlim(left=pd.to_datetime(years[0], format='%Y'), \
                 right=pd.to_datetime(years[-1]+1, format='%Y'))

    # plu.designProperties(ax, [graph], style)
        

def cumulativeChange(ax, glacier, attr, startdate=None, enddate=None, style='pub-jog'):
    checkAttribute(attr)
    
    cumulative_attr, cumulative_dates, _ = glacier.cumulativeChange(
        attr, startdate, enddate)
    graph, = ax.plot(cumulative_dates, cumulative_attr, \
        'o-', color=getColor('default'))

    ax.set_title('{}: {} Change'.format(glacier.name, attr_names[attr]))
    ax.set_xlabel(pickTimeLabel(glacier, attr))
    ax.set_ylabel('Cumulative {} Change ({})'.format(
        attr_names[attr], attr_units[attr]))
    xleft = pd.to_datetime(cumulative_dates.iloc[0].year-1, format='%Y')
    xright = pd.to_datetime(cumulative_dates.iloc[-1].year+1, format='%Y')
    ax.set_xlim(left=xleft, right=xright)
    plu.designProperties(ax, graph, style)


def differentialChange(ax, glacier, attr, startdate=None, enddate=None, style='pub-jog'):
    checkAttribute(attr)

    attrs, dates = glacier.filterDates(attr, startdate, enddate)
    diff_attrs = attrs.diff()

    graph = ax.bar(dates, diff_attrs, width=75, color=getColor('default'))
    ax.set_title('{}: {} Change Between Measurements'.format(
        glacier.name, attr_names[attr]))
    ax.set_xlabel(pickTimeLabel(glacier, attr))
    ax.set_ylabel('{} Change ({})'.format(attr_names[attr], attr_units[attr]))
    xleft = pd.to_datetime(dates.iloc[0].year-1, format='%Y')
    xright = pd.to_datetime(dates.iloc[-1].year+1, format='%Y')
    ax.set_xlim(left=xleft, right=xright)
    plu.designProperties(ax, graph, style)


def decadalChange(ax, glacier, attr, startdecades, style='pub-jog'):
    checkAttribute(attr)

    startdecades = pd.to_datetime(startdecades)
    net_decadal_change = pd.Series(dtype='float64')
    decade_labels = []
    bar_annotations = []
    for startyear in startdecades:
        endyear = met.addDecade(startyear)
        cumul_decadal_change, _, num_obsv = glacier.cumulativeChange(
            attr, startyear, endyear)
        net_decadal_change.loc[midyear] = cumul_decadal_change.iloc[-1]
        midyear = endyear.year - 4
        decade_labels.append('{}-{}'.format(startyear.year, startyear.year+9))
        bar_annotations.append('{} obsv'.format(num_obsv))
    
    rects = ax.bar(net_decadal_change.index.values, net_decadal_change.values,\
        width=5, color=getColor('default'))
    annotateBars(ax, bar_annotations, \
        net_decadal_change.index.values, net_decadal_change.values)
    ax.set_title('{}: Decadal {} Change'.format(glacier.name, attr_names[attr]))
    ax.set_xlabel('Decade')
    ax.set_ylabel('Net {} Change ({})'.format(
        attr_names[attr], attr_units[attr]))
    xtlocs = ax.get_xticks()
    ax.set_xticks(xtlocs+5)
    ax.set_xticklabels(decade_labels)
    if max(ax.get_ylim()) == 0.0:
        yrange, _ = ax.get_ylim()
        ax.set_ylim(top=abs(0.05*yrange))

    plu.designProperties(ax, rects, style)
    ax.set_axisbelow(True)
    ax.grid(axis='x')


def changeSummary(ax, glaciers, attr, GIDS=None, startdate=None, enddate=None):
    checkAttribute(attr)

    for g in glaciers:
        glacier = glaciers[g]
        cumulative_attr, cumulative_dates, _ = glacier.cumulativeChange(
            attr, startdate, enddate)
        graph, = ax.plot(cumulative_dates, cumulative_attr, color=getColor('default'), alpha=0.3)
        if GIDS is not None:
            glacier_styles = getGlacierStyles(GIDS)
            graph.set_marker(glacier_styles[glacier.gid]['m'])
            graph.set_linewidth(1)
            graph.set_markersize(1)
            graph.set_color(glacier_styles[glacier.gid]['c'])
            graph.set_alpha(0.7)
            graph.set_label(glacier.name)
        # plu.designProperties(ax, [graph], style)
    
    ax.set_title('Cumulative {} change'.format(attr_names[attr].lower()))
    ax.set_xlabel(pickTimeLabel(glacier, attr))
    ax.set_ylabel('{} ({})'.format(attr_names[attr], attr_units[attr]))
    xleft = pd.to_datetime(cumulative_dates.iloc[0].year-1, format='%Y')
    xright = pd.to_datetime(cumulative_dates.iloc[-1].year+1, format='%Y')
    ax.set_xlim(left=xleft, right=xright)
    if GIDS is not None:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08))


def changeSummaryNorm(ax, glaciers, attr, startdate=None, enddate=None, showmean=False):
    checkAttribute(attr)

    for g in glaciers:
        glacier = glaciers[g]
        scaled_attr, scaled_dates = glacier.normChange(attr, startdate, enddate)
        scaled_dates = pd.to_datetime(scaled_dates, format='%Y')
        ig, = ax.plot(scaled_dates, scaled_attr, color='silver', alpha=0.3, label='individual glaciers')
        # plu.designProperties(ax, [ig], style)
    if showmean is True:
        norm_mean, norm_std = met.normChangeStats(glaciers, attr, startdate=startdate, enddate=enddate)
        norm_dates = pd.to_datetime(norm_mean.index, format='%Y')
        av, = ax.plot(norm_dates, norm_mean.values, '.-',
            color=getColor('default'), linewidth=3, markersize=8, zorder=10, label='mean')
        sd = ax.fill_between(norm_dates, 
            norm_mean.values+norm_std.values, norm_mean.values-norm_std.values, 
            color=getColor('default'), alpha=0.2, zorder=10, label='std')  
    
    ax.set_title('Normalized cumulative {} change'.format(attr_names[attr].lower()))
    ax.set_xlabel(pickTimeLabel(glacier, attr))
    ax.set_ylabel('Normalized change')
    xleft = pd.to_datetime(scaled_dates.iloc[0].year-1, format='%Y')
    xright = pd.to_datetime(scaled_dates.iloc[-1].year+1, format='%Y')
    ax.set_xlim(left=xleft, right=xright)
    ax.set_ylim(bottom=-0.0, top=1.0)
    ax.legend(handles=[ig, av, sd])



def changePointHistogram(ax, glaciers, attr, startdate=None, enddate=None, n_breakpoints=1, method='window', model='l1', wwidth=5, year_bins=None):
    """Plot histogram of breakpoint years for glacier population."""
    checkAttribute(attr)
    population_breakpoint_years = []
    for g in glaciers:
        glacier = glaciers[g]
        breakpoint_dates, _, _ = met.changePointDetection(glacier, attr, startdate=startdate, enddate=enddate, n_breakpoints=n_breakpoints, method=method, model=model, wwidth=wwidth)
        breakpoint_years = [d.year for d in breakpoint_dates]
        # graph = ax.scatter(breakpoint_years, [g]*len(breakpoint_years), color=getColor('default'))
        # plu.designProperties(ax, graph)
        population_breakpoint_years.extend(breakpoint_years)
    graph = plt.hist(population_breakpoint_years, bins=year_bins, rwidth=0.8, color=getColor('default'))
    _, scaled_dates = glacier.normChange(attr, startdate, enddate)
    
    ax.set_title('Glacier {} Change Points'.format(attr_names[attr]))
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    xleft = scaled_dates.iloc[0].year-1
    xright = scaled_dates.iloc[-1].year+1
    plt.xlim(left=xleft, right=xright)
    plu.designProperties(ax, graph)
        

def piecewiseFit(ax, glacier, attr, time_bins, n_segs=3, startdate=None, enddate=None):
    checkAttribute(attr)
    bp, pwlf_fun = met.getBreakPoints(glacier, attr, startdate, enddate, n_segs)
    attr_fit = pwlf_fun.predict(time_bins)
    ax.plot([d.year for d in glacier.dates], getattr(glacier, attr).values, color=getColor('default'))
    graph = ax.plot(time_bins, attr_fit, '-', color='orange')
    ax.set_title('Piecewise Linear Fit of {} {} Change'.format(glacier.name, attr_names[attr]))
    ax.set_xlabel('Year')
    ax.set_ylabel('{} ({})'.format(attr_names[attr], attr_units[attr]))


def breakPointsHist(ax, glaciers, attr, time_bins, n_segs=3, startdate=None, enddate=None):
    checkAttribute(attr)
    time_bins = pd.to_datetime(time_bins, format='%Y')
    all_breaks = []
    for g in glaciers:
        glacier = glaciers[g]
        breaks, _ = met.getBreakPoints(glacier, attr, n_segs, startdate, enddate)
        all_breaks.extend(pd.to_datetime(breaks, format='%Y'))
    _, scaled_dates = glacier.normChange(attr, startdate, enddate)
    graph = ax.hist(all_breaks, bins=time_bins, align='mid', rwidth=0.8, color=getColor('default'))
    ax.set_title('{} break points'.format(attr_names[attr]))
    ax.set_xlabel(pickTimeLabel(glacier, attr))
    ax.set_ylabel('Count')
    xleft = pd.to_datetime(scaled_dates.iloc[0].year-1, format='%Y')
    xright = pd.to_datetime(scaled_dates.iloc[-1].year+1, format='%Y')
    ax.set_xlim(left=xleft, right=xright)
    # plu.designProperties(ax, graph, style)
