import tensorflow as tf


class LivenessNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        
        
        # Por default viene 'channels_first', por lo cual debemos de colocar los canales RGB al final de la matriz
        INPUT_SHAPE = (height, width, depth)
        chanDim = -1 # Accedemos al último eje de la matriz o tensor, ej nuestro caso (height, width, depth) -1: Corresponde a depth
        
        
        # Si estuvieran los canales RGB al inicio de la matriz, entonces actualiza el INPUT_SHAPE y accedemos a la posición 1, que seguiría siendo depth
        if tf.keras.backend.image_data_format() == 'channels_first':
            INPUT_SHAPE = (depth, height, width)
            chanDim = 1 
        
        # Our CNN exhibits VGGNet-esque qualities. It is very shallow with only a few learned filters. 
        # Ideally, we won’t need a deep network to distinguish between real and spoofed faces.
        #Se utiliza para crear modelos secuenciales de redes neuronales
        model = tf.keras.Sequential([
            
            
                # first set CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu' ,input_shape=INPUT_SHAPE), #conv_1_1, relu 1_1
                tf.keras.layers.BatchNormalization(axis=chanDim),
                
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),#conv_1_2, relu 1_2
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), #max_pool_1
                tf.keras.layers.Dropout(0.25),
                
                
                
                # second set CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),#conv 2_1, relu 2_1
                tf.keras.layers.BatchNormalization(axis=chanDim),
                
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'), #conv 2_2, relu 2_2
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), #max_pool_2
                tf.keras.layers.Dropout(0.25),
                
                
                
                # FullyConnected => BatchNorm => Dropout
                tf.keras.layers.Flatten(), #Flatten
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                # output
                tf.keras.layers.Dense(classes, activation='softmax') #Final softmax
            ])
        
        return model
    
