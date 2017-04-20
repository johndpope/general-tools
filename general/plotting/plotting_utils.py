import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


__all__ = [ 'gridplot', 'gridplot_sidebyside' ]


imshow_plot = lambda ax, data: ax.imshow(data, interpolation='nearest', cmap=plt.cm.gray)

def gridplot(data, num_cols=6, plot=imshow_plot, titles=None, base_figure_size=15):
    num_rows = data.shape[0] // num_cols + 1

    fig = plt.figure(figsize=(base_figure_size, base_figure_size*num_rows/num_cols))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.01, hspace=0.15)

    for i in range(data.shape[0]):
        curr_data = data[i]
        ax = plt.Subplot(fig, gs[i])
        imshow_plot(ax,curr_data)
        if titles is not None:
            ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    plt.show()


# comparing results side by side.
# data should be given as array of [X,y]
# i.e. gridplot_sidebyside([X1,y1], [X2,y2], ...)
def gridplot_sidebyside(*data):
    # comparing only the shape of all X's (trusting the user with the lengths of y's)
    if len(set(x[0].shape[0] for x in data)) != 1:
        raise Exception("data sets to compare side-by-side are of different lengths")

    all_X = np.zeros((len(data)*data[0][0].shape[0],) + data[0][0].shape[1:])
    all_titles = [None] * len(data)*data[0][1].shape[0]

    for i,d in enumerate(data):
        x = d[0]
        y = d[1]
        all_X[i::len(data)] = x
        all_titles[i::len(data)] = [",".join("{:.2f}".format(s) for s in ss) for ss in y]

    gridplot(all_X, titles=all_titles, num_cols=len(data))
