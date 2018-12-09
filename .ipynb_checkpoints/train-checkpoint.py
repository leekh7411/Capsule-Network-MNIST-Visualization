from models import base_model
from utils import *
from hyperparams import * # Check the hyper parameters in 'hyperparams.py'
from keras import callbacks, optimizers
from keras import backend as K

def margin_loss(y_true, y_pred):
    # original source code from 
    # https://github.com/XifengGuo/CapsNet-Keras
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

if __name__ == "__main__":
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_shape = x_train[0].shape
    y_shape = y_train[0].shape
    
    # Init the baseline model
    train_model, _ , _ , _ = base_model(input_shape = x_shape, output_shape = y_shape)
    
    # Init callback functions
    checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, 
                                           monitor='val_capsnet_acc', 
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           verbose=1)
    earlystopper = callbacks.EarlyStopping(monitor='val_capsnet_acc', 
                                           patience=3, 
                                           verbose=0)
    
    # Compile the training model
    train_model.compile(optimizer=optimizers.Adam(lr=LearningRate), 
                        loss=[margin_loss, 'mse'], # classification error & reconstruction error
                        loss_weights=[1., LAM_RECON], 
                        metrics={'capsnet': 'accuracy'})
    
    # Train model
    train_model.fit([x_train, y_train], 
                    [y_train, x_train], 
                    batch_size = BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=[[x_test, y_test], [y_test, x_test]], 
                    callbacks=[earlystopper, checkpoint])
    