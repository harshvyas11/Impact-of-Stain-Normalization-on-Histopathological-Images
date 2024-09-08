import script.loss as l
from keras_unet_collection import models
import tensorflow as tf




def model_list():
    return [attnet(),pocketnet()]

def attnet(x=8):

    model = models.att_unet_2d((None, None, 3), [64/x, 128/x, 256/x, 512/x], n_labels=1,
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                            batch_norm=True, pool=True, unpool='bilinear', name='attunet')
    model.compile(optimizer='adam', loss=l.dice_loss, metrics=[l.dice_coefficient])
    return model

def pocketnet():

    model = models.att_unet_2d((None, None, 3), [16, 16, 16, 16], n_labels=1,
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                            batch_norm=True, pool=True, unpool='bilinear', name='pocketnet')
    model.compile(optimizer='adam', loss=l.dice_loss, metrics=[l.dice_coefficient])
    return model




def save(model,method,refrence,name):
    model.save(f"Models\\{name}\\method_{method}\\referance_{refrence}")



def load(method,refrence,name):
    return tf.keras.models.load_model(f"Models\\{name}\\method_{method}\\referance_{refrence}", custom_objects={'dice_coefficient': l.dice_coefficient, 'dice_loss': l.dice_loss})

