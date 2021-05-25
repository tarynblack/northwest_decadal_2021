# General utilities for plotting
# Taryn Black (1/2021)

import pandas as pd
import matplotlib as mpl
import inspect
import os


def globalDesignProperties(style):
    """Set standardized figure properties"""
    # Load styles template
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    styles = pd.read_csv(os.path.join(filepath, 'styles.csv'), index_col=0).to_dict()
    
    # Get style values
    linewidth = styles['linewidth'][style]
    markersize = styles['markersize'][style]
    titlesize = styles['titlesize'][style]
    labelsize = styles['labelsize'][style]
    ticklabelsize = styles['ticklabelsize'][style]
    legendsize = styles['legendsize'][style]
    fontfamily = styles['fontfamily'][style]
    fontname = styles['fontname'][style]

    # Set text properties
    mpl.rcParams['axes.titlesize'] = titlesize
    mpl.rcParams['axes.labelsize'] = labelsize
    mpl.rcParams['xtick.labelsize'] = ticklabelsize
    mpl.rcParams['ytick.labelsize'] = ticklabelsize
    mpl.rcParams['legend.fontsize'] = legendsize
    mpl.rcParams['legend.title_fontsize'] = labelsize
    mpl.rcParams['font.family'] = fontfamily
    mpl.rcParams['font.{}'.format(fontfamily)] = [fontname]
    mpl.rcParams['font.size'] = labelsize

    # Set line and marker properties
    mpl.rcParams['lines.linewidth'] = linewidth
    mpl.rcParams['lines.markersize'] = markersize

    # Set axes and grid properties
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.color'] = 'lightgray'

    # Set color cycler
    # mpl.rcParams['axes.prop_cycle'] = cycler('color', ['dodgerblue', 'darkorange',  'mediumseagreen', 'darkorchid'])


def designProperties(ax=None, graphs=None, style=None):
    """Set standardized figure properties"""
    # Load styles template
    # filename = inspect.getframeinfo(inspect.currentframe()).filename
    # filepath = os.path.dirname(os.path.abspath(filename))
    # styles = pd.read_csv(os.path.join(filepath, 'styles.csv'), index_col=0).to_dict()
    # Get style values
    # linewidth = styles['linewidth'][style]
    # markersize = styles['markersize'][style]
    # titlesize = styles['titlesize'][style]
    # labelsize = styles['labelsize'][style]
    # ticklabelsize = styles['ticklabelsize'][style]
    # legendsize = styles['legendsize'][style]
    # fontfamily = styles['fontfamily'][style]
    # fontname = styles['fontname'][style]

    # Set line and marker properties
    # if graphs:
    #     for graph in [graphs]:
    #         if type(graph) == mpl.lines.Line2D:
    #             graph.set_zorder(5)
                # graph.set_linewidth(linewidth)
                # graph.set_markersize(markersize)
        
    # Set text properties
    # mpl.rcParams['font.family'] = fontfamily
    # mpl.rcParams['font.{}'.format(fontfamily)] = [fontname]
    # mpl.rcParams['legend.fontsize'] = legendsize
    # ax.axes.title.set_fontsize(titlesize)
    # ax.xaxis.label.set_fontsize(labelsize)
    # ax.yaxis.label.set_fontsize(labelsize)
    # ax.tick_params(labelsize=ticklabelsize)
    # if ax.get_legend() is not None:
    #     ax.legend(fontsize=legendsize)
    # if ax.get_children():
    #     for child in ax.get_children():
    #         if isinstance(child, mpl.text.Annotation):
    #             child.set_fontsize(labelsize)

    # Set other properties
    # ax.grid(color='lightgray')
    # ax.axhline(linewidth=0.75, color='black')
    # ax.set_axisbelow(True)


def figureWidth(style):
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    styles = pd.read_csv(os.path.join(filepath, 'styles.csv'), index_col=0).to_dict()
    figure_column_width = styles['colwidth'][style]
    return figure_column_width


def zeroLine(ax):
    ax.axhline(linewidth=0.75, color='black')