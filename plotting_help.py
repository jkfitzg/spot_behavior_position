orange = (0.85, 0.54, 0.24)
purple = (0.5, 0.5, 1)
blue = (0, 0, 1)
dark_blue = (.1,.1,.44)
bblue = (0.51, 0.67, 0.89)
green = (0.5, 0.75, 0.42)
bgreen = (.34, .63, .56)
red = (0.82, 0.32, 0.22)
grey = (0.7, 0.7, 0.7)
black = (0., 0., 0.)
white = (1., 1., 1.)
yellow = (.95, .74, .22)
magenta = (.8, 0, .8)


def keep_axes(ax, axes_to_keep, color='black'):
    all_axes = ['left', 'right', 'top', 'bottom']
    ax.xaxis.label.set_color(color)
    ax.tick_params(axis='x', colors=color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y', colors=color)
    for axis in all_axes:
        if axis not in axes_to_keep:
            ax.spines[axis].set_color('none')
        else:
            ax.spines[axis].set_color(color)
            
    if 'left' in axes_to_keep:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in axes_to_keep:
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right') 
    else:
        ax.yaxis.set_ticks([])

    if 'top' in axes_to_keep:
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top') 
    elif 'bottom' in axes_to_keep:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])
        

def set_legend(ax, facecolor='none', edgecolor='black', textcolor='black', loc='upper right'):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    legend = ax.legend(loc=loc)
    frame = legend.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_edgecolor(edgecolor)
    for text in legend.get_texts():
        text.set_color(textcolor)