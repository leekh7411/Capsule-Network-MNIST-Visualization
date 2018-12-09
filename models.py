# original source code from https://github.com/XifengGuo/CapsNet-Keras
import numpy as np
from keras import layers, models
from keras import backend as K
from layers import PrimaryCap, CapsuleLayer, CapsLength, Mask
from hyperparams import * # Check the hyper parameters in 'hyperparams.py'

def base_model(input_shape, output_shape):
    input_x = layers.Input(shape=input_shape)
    
    _routings = ROUTING
    _digitcap_num = NUM_CLASSES
    _digitcap_dim = DIGIT_CAP_DIM
    
    x = layers.Conv2D(filters=256, 
                      kernel_size=9,
                      strides=1, 
                      padding='valid', 
                      activation='relu', 
                      name='conv1')(input_x)
    
    x = PrimaryCap(x, 
                   dim_capsule=8, 
                   n_channels=32, 
                   kernel_size=9, 
                   strides=2, 
                   padding='valid')
    
    x = CapsuleLayer(num_capsule = _digitcap_num,
                     dim_capsule = _digitcap_dim,
                     routings = _routings,
                     name='digitcaps')(x)
    digitcaps = x
    
    x = CapsLength(name='capsnet')(x)
    y_pred = x 
    
    # For 'reconstruction' and also as a 'regulerizer', 
    # we get y label as an input as well
    y_label = layers.Input(shape=output_shape)
    
    true_digitcap = Mask()([digitcaps, y_label])
    maxlen_digitcap = Mask()(digitcaps)
    
    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim= _digitcap_dim * _digitcap_num))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    
    # Models for training and evaluation (prediction)
    train_model = models.Model([input_x, y_label], [y_pred, decoder(true_digitcap)])
    eval_model = models.Model(input_x, [y_pred, decoder(maxlen_digitcap)])
    digitcaps_model = models.Model(input_x, digitcaps)
    
    # summary
    train_model.summary()
    
    return train_model, eval_model, digitcaps_model, decoder

if __name__ == "__main__":
    model = base_model(input_shape=(28,28,1), output_shape=(10,))