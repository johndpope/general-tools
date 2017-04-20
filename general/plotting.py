import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

imshow_plot = lambda ax, data: ax.imshow(data, interpolation='nearest', cmap=plt.cm.gray)

def gridplot(data, num_cols=6, plot=imshow_plot, titles=None):
    FIGSIZE_W = 15
    num_rows = data.shape[0] // num_cols + 1

    fig = plt.figure(figsize=(FIGSIZE_W, FIGSIZE_W*num_rows/num_cols))
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
