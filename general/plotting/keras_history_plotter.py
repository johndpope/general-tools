import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class KerasHistoryPlotter:

    def __init__(self, base_figure_size=7):
        self.base_figure_size = base_figure_size

    def plot(self, results, metric=None, ylim=None):
        gs = gridspec.GridSpec(4,2)
        colors = plt.cm.gist_rainbow(np.linspace(0,1,len(results)))

        plt.figure(figsize=(2*self.base_figure_size,self.base_figure_size))

        plt.subplot(gs[:3,0])
        for idx, data in enumerate(results):
            hist = data.history.history
            plt.plot(hist['loss'], label="train #{}".format(idx), color=colors[idx], linestyle='dashed')
            if "val_loss" in hist:
                plt.plot(hist['val_loss'], label="val #{}".format(idx), color=colors[idx])
        #plt.legend(loc='best')
        plt.title("Loss")
        if ylim:
            plt.ylim(ylim)


        if metric is None:
            all_metrics = [ m for x in results for m in x.history.history.keys() if m!='loss' and not m.startswith('val_') ]
            metric = all_metrics[0] if len(all_metrics)>0 else None

        if metric is not None:
            plt.subplot(gs[:3,1])
            for idx, data in enumerate(results):
                hist = data.history.history
                plt.plot(hist[metric], label="train #{}".format(idx), color=colors[idx], linestyle='dashed')
                if "val_"+metric in hist:
                    plt.plot(hist["val_"+metric], label="val #{}".format(idx), color=colors[idx])
            #plt.legend(loc='best')
            plt.title(metric)
            if ylim:
                plt.ylim(ylim)


        plt.subplot(gs[3,:])
        plt.axis('off')
        plt.ylim(0,len(results))
        for idx, data in enumerate(results):
            plt.text(0, idx, "{} - {}".format(idx, str(data.params)), color=colors[idx],
                        horizontalalignment='left',verticalalignment='center')
