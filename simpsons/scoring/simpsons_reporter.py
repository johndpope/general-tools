import copy
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import io
import ipywidgets as widgets
from IPython.display import display


import logging
log = logging.getLogger(__name__)


class SimpsonsReporter:

    def __init__(self, train_dev_X, train_dev_y, dev_X, dev_y,
            generator_fn=None, train_dev_num_images=1000, dev_preprocess=None,
            base_figure_size=7):
        self.train_dev_X = train_dev_X
        self.train_dev_y = train_dev_y
        self.train_dev_num_images = train_dev_num_images
        self.dev_X = dev_X
        self.dev_y = dev_y
        self.base_figure_size = base_figure_size
        self.generator_fn = generator_fn

        # check if preprocess is valid.
        # valid = all transformers
        self.dev_preprocess = dev_preprocess
        if self.dev_preprocess:
            if not all(hasattr(step[1], 'transform') for step in self.dev_preprocess.steps):
                raise Exception("all steps in preprocess_pipeline must implement transform")

        self.report_ = None


    def report(self, results):
        panels = []
        for result in results:
            fig = plt.figure(figsize=(self.base_figure_size, self.base_figure_size))
            self._report_single_result(result, fig)
            plt.close()

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            im = widgets.Image(value=buf.getvalue(), format='png')
            buf.close()
            panels.append(im)

        cbs = [ widgets.Checkbox(value=False) for r in results ]
        cbs_and_labels = [ widgets.HBox([
            cb,
            widgets.Label(str(r.params))
        ]) for (cb,r) in zip(cbs,results)]

        button = widgets.Button(description="Compare")

        cb_container = widgets.VBox(cbs_and_labels + [button])
        plots_container = widgets.HBox()

        display(cb_container,plots_container)

        def compare(button):
            res = [ p for p,cb in zip(panels,cbs) if cb.value ]
            plots_container.children = res

        button.on_click(compare)



    def _report_single_result(self, result, fig=None):
        if not fig:
            fig = plt.figure()

        # evaluating train_dev
        train_dev_X, train_dev_y = self._generate_train_dev(result.params)
        train_dev_pred = result.history.model.predict(train_dev_X)

        # evaluating dev
        dev_X, dev_y = self.dev_X, self.dev_y
        if self.dev_preprocess:
            dev_X = self.dev_preprocess.transform(dev_X)
        dev_pred = result.history.model.predict(dev_X)

        max_accs = [
            sorted([ (dev_y[:,char_ind] == (dev_pred[:,char_ind] > t).astype(int)).mean() for t in np.linspace(0.,1.,21) ])[-1]
            for char_ind in range(dev_y.shape[1])
        ]

        suptitle = \
            "\n".join( ",  ".join( "{}: {}".format(k,v) for k,v in list(result.params.items())[i:i+3] ) for i in range(0,len(result.params),3) ) \
            + "\n" + \
            "max(acc): " + ",".join("{:.2f}".format(x) for x in max_accs)

        fig.suptitle(suptitle)
        gs = gridspec.GridSpec(2,2)

        self._plot_roc_curves(gs[0,0], train_dev_pred, train_dev_y, title="Train Dev ROC")
        self._plot_roc_curves(gs[0,1], dev_pred, dev_y, title="Dev ROC")
        self._plot_pr_curves (gs[1,0], train_dev_pred, train_dev_y, title="Train Dev PR")
        self._plot_pr_curves (gs[1,1], dev_pred, dev_y, title="Dev PR")
        gs.tight_layout(fig, rect=[0, 0, 1, 0.87])


    def _plot_pr_curves(self, gs, y_pred, y_true, title=None):
        num_characters = y_true.shape[1]

        pr_curves = [ precision_recall_curve(y_true[:,ind], y_pred[:,ind]) for ind in range(num_characters) ]

        plt.subplot(gs)
        for ind, (prec, recall, _) in enumerate(pr_curves):
            curr_auc = auc(recall, prec)
            plt.plot(recall, prec, label="#{} ({:.2f})".format(ind, curr_auc))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim((0,1))
        plt.ylim((0,1))
        if title:
            plt.title(title)
        plt.legend(loc="best")


    def _plot_roc_curves(self, gs, y_pred, y_true, title=None):
        num_characters = y_true.shape[1]

        roc_curves = [ roc_curve(y_true[:,ind], y_pred[:,ind]) for ind in range(num_characters) ]
        aucs = [ roc_auc_score(y_true[:,ind], y_pred[:,ind]) for ind in range(num_characters) ]

        plt.subplot(gs)
        plt.plot([0,1], [0,1], linestyle='dashed', color='orange') # the "random" line
        for ind, ((fpr, tpr, _), auc) in enumerate(zip(roc_curves, aucs)):
            plt.plot(fpr, tpr, label="#{} ({:.2f})".format(ind, auc))
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.xlim((0,1))
        plt.ylim((0,1))
        if title:
            plt.title(title)
        plt.legend(loc="best")






    def _generate_train_dev(self, res_args):
        # if needed - generate train_dev samples
        if self.generator_fn:
            gen_args = self._filter_params(self.generator_fn, res_args)
            generator = self.generator_fn(self.train_dev_X, self.train_dev_y, **gen_args)
            tmp_train_dev = []
            while sum(x[0].shape[0] for x in tmp_train_dev) < self.train_dev_num_images:
                tmp_train_dev.append(next(generator))
            train_dev_X = np.concatenate([ x[0] for x in tmp_train_dev ])
            train_dev_y = np.concatenate([ x[1] for x in tmp_train_dev ])
        else:
            train_dev_X, train_dev_y = self.train_dev_X, self.train_dev_y

        return train_dev_X[:self.train_dev_num_images], train_dev_y[:self.train_dev_num_images]



    # from keras.wrappers.scikit_learn
    def _filter_params(self, fn, params):
        params = copy.deepcopy(params)

        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name, value in params.items():
            if name in fn_args:
                res.update({name: value})
        return res
