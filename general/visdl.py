import keras.models
import keras.layers
import numpy as np
import copy

class ModelVis:
    class LayerVis:
        def __init__(self, model, keras_layer):
            if not isinstance(keras_layer, keras.layers.Convolution2D):
                raise ValueError("Layer must be a Convolution2D layer for now..")
                
            self.model = model
            self.layer = keras_layer
            self.num_filters = self.layer.nb_filter
            self.filter_size_xy = np.array([self.layer.nb_row, self.layer.nb_col]) 

            try:
                self.idx = self.model.layers.index(self.layer)
            except ValueError as e:
                raise ValueError("Can't find layer in model layers", e)


        def find_minimal_input_size_for_activation(self):
            input_shape = np.array([1,1]) # eventually we want to activate the filter once.
            print("input_shape starts with - ", input_shape)
            
            layer_idx = self.idx
            while layer_idx >= 0:
                curr_layer = self.model.layers[layer_idx]
                print("layer - ", curr_layer)
                if isinstance(curr_layer, keras.layers.Dropout):
                    pass
                elif isinstance(curr_layer, keras.layers.MaxPooling2D):
                    input_shape = input_shape * curr_layer.pool_size
                elif isinstance(curr_layer, keras.layers.Convolution2D):
                    curr_layer_filter_size = np.array([curr_layer.nb_row, curr_layer.nb_col])
                    input_shape += curr_layer_filter_size - 1

                print("input_shape = ", input_shape)
                layer_idx -= 1
            
            return input_shape

    # ---------------------------------------------------------
    
    def __init__(self, arg):
        if isinstance(arg, keras.models.Model):
            self.model = arg
        elif isinstance(arg, str):
            self.model = keras.models.load_model(arg)
            
    
    def get_layer(self, layer_id):
        if isinstance(layer_id, int):
            layer = self.model.layers[layer_id]
        else:
            layer = self.model.get_layer(layer_id)
        return self.LayerVis(self.model, layer)
        


    def maximize_activation(self, layer_id, filter_id):
        layer = self.get_layer(layer_id)
        minimal_input_size = layer.find_minimal_input_size_for_activation()
        
        m = copy.copy(self.model)
#        m.layers = m.layers[:layer.idx]
        return m
        # change model input to accept this new input_size
        
        # put a placeholder on the input
        
        # GD when W is fixed to find an optimal input
        
        
        
        

 
