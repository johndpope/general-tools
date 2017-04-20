from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def compare_models(X, y, build_model_func, model_args, nb_epoch=20, batch_size=64, validation_data=None):
    if validation_data:
        val_params = dict(validation_data=validation_data)
    else:
        val_params = dict(validation_split=0.2)


    results = [ (k, model.fit(X, y, **val_params))
          for k,v in model_args.items()
          for model in [ KerasClassifier(build_model_func, nb_epoch=nb_epoch, verbose=2, **v) ]
          ]

    return results

def show_results(results, window_size=1):
    def plt_convolve(X, window_size):
        return np.convolve(X, np.ones(window_size)/window_size, mode='same')[:-window_size]


    plt.figure(figsize=(30,15))

    for c, (label, hist) in zip(np.linspace(0,1,len(results)), results):
        plt.plot(plt_convolve(hist.history['val_acc'], window_size), color=matplotlib.cm.jet(c), linestyle='solid',  linewidth=2.0, label="{}(val)".format(label))
        plt.plot(plt_convolve(hist.history['acc'], window_size), color=matplotlib.cm.jet(c), linestyle='dashed', linewidth=1.0, label="{}(train)".format(label))

        #    plt.plot(np.convolve(hist.history['val_acc'], np.ones(window_size)/window_size, mode='same')[:-window_size], color=matplotlib.cm.jet(c), label="{}(val)".format(label))

    plt.legend(loc='best')
    plt.show()
