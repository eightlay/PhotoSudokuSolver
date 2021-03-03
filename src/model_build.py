# import packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def DRbuild(width: int, 
            height: int, 
            depth: int, 
            classes: int = 10)\
        -> Sequential:
    '''
        Builds digit recognition tensoflow model

        Parameters
        ----------
        - width : int
                  width of input data
        - height : int 
                   height of input data
        - depth : int
                  depth (number of color channels) of input data
        - classes : int
                    number of classes

        Returns
        -------
        tf.keras.models.Sequential {built model}
    '''
    inputShape = (height, width, depth)

    return Sequential([
        # 1: CONV => RELU => POOL
        Conv2D(32, (5, 5), padding='same', input_shape=inputShape),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 2: CONV => RELU => POOL
        Conv2D(32, (3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 3: F => D => RELU => DO
        Flatten(),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),

        # 4: softmax classifier
        Dense(classes),
        Activation('softmax')
    ])
