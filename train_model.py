######################################## SECCIÓN 1 LIBRERÍAS ############################################3######################

import matplotlib
matplotlib.use('Agg') # Agg es usado para escribir archivos png

from livenessnet import LivenessNet # our model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os


############################ SECCIÓN 2 PARÁMETROS ###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True,
                    help='Path to input Dataset')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to output trained model') #le = label_encoder
parser.add_argument('-l', '--le', type=str, required=True,
                    help='Path to output Label Encoder')
parser.add_argument('-p', '--plot', type=str, default='plot.png',
                    help='Path to output loss/accuracy plot')
args = vars(parser.parse_args())

############################## SECCIÓN 3 LISTAR EL DATASET ###########################################################################

print('Obteniendo imágenes del dataset...')

#Creamos una lista con TODOS los archivos en el dataset que se nos pasó en los argumentos
imagePaths = list(paths.list_images(args['dataset'])) 

data = list()
labels = list()


########################### SECCIÓN 4 ITERACIÓN EN CADA IMÁGEN DE LA LISTA ############################################################

# Itera en cada imágen de la lista
for imagePath in imagePaths:
    
    label = imagePath.split(os.path.sep)[-2] #Obtenemos el nombre fake o real
    image = cv2.imread(imagePath) #Leemos la imagen en forma de arreglo
    image = cv2.resize(image, (32,32)) #Redimensionamos la imágen a un arreglo de (32, 32, 3)
    
    data.append(image) #Agregamos la imagen en forma de arreglo a la lista data
    labels.append(label) #Agregamos la etiqueta fake o true de la imagen a la lista labels
    
    
    
    
########################## SECCIÓN 5 CLASIFICACIÓN DE LAS ETIQUETAS "REAL" Y "FAKE" #################################################################################################    
    
#Cambiamos la escala BGR de 0 - 255 a 0 - 1
#Un numpy array (también llamado ndarray) trabaja de una manera super rápida y eficiente con Tensorflow.
data = np.array(data, dtype='float') / 255.0 #cada uno de los arreglos 32x32 los convierte a uno solo de 3 dimensiones

# Clasificamos las etiquetas de (fake, real) a (0, 1)
# Y haz one-shot encoding
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, 2) # one-shot encoding




############################ SECCIÓN 6 PARÁMETROS DE ENTRENAMIENTO #######################################################################################

# train/test split
# we go for traditional 80/20 split
# Theoretically, we have small dataset we need test set to be a bit bigger
# 75/25 or 70/30 split would be ideal, but from the trial and error
# 80/20 gives a better result, so we go for it
# Another thing to consider, since my dataset has only about 14 images of faces from card/solid image
# so 80/20 has a higher chance that those images will be in training set rather than test set (and none on training set)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

# construct the training image generator for data augmentation
# this method from TF will do augmentation at runtime. So, it's quite handy
aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20, 
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# build a model
# define hyperparameters
INIT_LR = 1e-5 # initial learning rate
BATCH_SIZE = 4
EPOCHS = 750


################################### SECCIÓN 7 OPTIMIZADOR, EARLYSTOPPING Y COMPILACIÓN DEL MODELO DE ENTRENAMIENTO ###############################################################

# we don't need early stopping here because we have a small dataset, there is no need to do so
print('[INFO] compiling model...')
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR)
#optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

#Implementación de la Red Neuronal Convolucional (la traemos desde LivenessNet.py)
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
#binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)



########################## SECCIÓN 8 ENTRENAMIENTO, EVALUACIÓN DEL MODELO ENTRENADO Y REPORTE DEL MODELO ENTRENADO #################################################################

# Entrenamiento del modelo
print(f'[INFO] training model for {EPOCHS} epochs...')
history = model.fit(x=aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    epochs=EPOCHS)

# Evaluación del modelo entrenado (qué tan bien predice)
print('[INFO] evaluating network...')
predictions = model.predict(x=X_test, batch_size=BATCH_SIZE)

#Reporte del modelo entrenado (muestra la presición, recall, f1-score, que mientras sea mayor es mejor)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))




############################################### SECCIÓN 9 #################################################################

# Guardamos la red neuronal entrenada en formato binario en liveness.model (argumento que pasamos en 'model' anteriormente)
print(f"[INFO serializing network to '{args['model']}']")
model.save(args['model'], save_format='h5')


# LabelEncoder (le), que contiene las etiquetas Real y Fake codificadas, lo guardamos en Formato Binario en label_encoder
with open(args['le'], 'wb') as file:
    file.write(pickle.dumps(le))
    
# plot training loss and accuract and save
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, EPOCHS), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, EPOCHS), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCHS), history.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, EPOCHS), history.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])