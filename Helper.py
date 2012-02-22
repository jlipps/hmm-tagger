######### Helper.py #########

from __future__ import division # use floating-point division
import sys # for logging to stderr

def progress_bar(complete, total, elapsed_time=0):
    """
    Output a progress bar to the screen.

    :param complete: int number of items completed
    :param total: int total number of items
    :param elapsed_time: elapsed time to show, default 0
    """

    bar_width = 50 # how wide should our progress bar be?
    pct_complete = complete / total

    # get how many ticks we need to print for pct_complete
    ticks = int(bar_width * pct_complete)

    # get how many spaces to fill out rest of bar with
    spaces = bar_width - ticks

    output = "\r["

    # print each tick, but turn last tick into arrowhead
    for i in range(ticks):
        if i == ticks-1 and pct_complete < 1:
            output += ">"
        else:
            output += "="

    # print spaces
    for i in range(spaces):
        output += " "

    # print stats
    output += "] %0.2f%% (%d / %d)" % (pct_complete*100, complete, total)
    if elapsed_time > 0:
        output += " %0.2fs" % elapsed_time

    # write to screen
    msg(output)

def indices_of_max(array):
    """
    Return the index/indices in an array which have the highest value.

    :param array: list to search
    """

    indices = [] # intialize index list

    # for each item in array, append index if it has max value
    for i in range(len(array)):
        if array[i]==max(array):
            indices.append(i)

    return indices

def msg(text):
    """
    Write arbitrary text to stderr

    :param text: string text to write
    """

    sys.stderr.write(text)
