######### CLASS-EXTERNAL HELPER FUNCTIONS #########

from __future__ import division # use floating-point division
import sys # for logging to stderr

def progress_bar(complete, total, elapsed_time=0):
    bar_width = 50
    pct_complete = complete / total
    ticks = int(bar_width * pct_complete)
    spaces = bar_width - ticks
    output = ''
    output += "\r["
    for i in range(ticks):
        if i == ticks-1 and pct_complete < 1:
            output += ">"
        else:
            output += "="
    for i in range(spaces):
        output += " "
    output += "] %0.2f%% (%d / %d)" % (pct_complete*100, complete, total)
    if elapsed_time > 0:
        output += " %0.2fs" % elapsed_time
    msg(output)
    
def indices_of_max(array):
    indices = []
    for i in range(len(array)):
        if array[i]==max(array):
            indices.append(i)
    return indices
    
def msg(text):
    sys.stderr.write(text)
        