import os
# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from imutils.video import VideoStream
import paho.mqtt.client as mqtt
import tensorflow as tf
import face_recognition
import numpy as np
import imutils
import pickle
import time
import cv2

# OBTENER EL NOMNRE DEL ROSTRO QUE MAS SE DECETÓ A LO LARGO DEL RECONOCIMINETO LISTO
#### AGREGAR, POR EJEMPLO, SI UN INTRUSO SE DETECTÓ POR 15 ITERACIONES, ENTONCES LO REGISTRA EN LA BD LISTO

#-------------------------------------------- TODO ------------------------------------------------
# OPTIMIZAR LA EJECUCIÓN DEL CÓDIGO
# Urge entrenar el modelo Liveness

#-----------OPCIONAL, PRIMERO DEBE CONSIDERARSE
# si solo necesitas reconocer un número limitado de personas, podrías pre-calcular los encodings de las personas y almacenarlos en una base de datos en lugar de calcularlos en 
# tiempo real en cada cuadro del video. De esta manera, solo necesitarías comparar los encodings de los rostros detectados con los encodings almacenados en la base de datos en
# lugar de calcular todos los encodings en cada cuadro. Esto puede ser significativamente más rápido y escalable en situaciones donde hay muchas personas que se deben reconocer.
#Todo lo anterior, en la última parte de la SECCIÓN 2.4

#--------------------------------- IMPORTANTE -------------------------------------------------------------------
#El liveness detection(real / fake) funciona mejor a una distancia de unos 80cm despegado de la cámara
# habrá que ver como funciona con distintas iluminaciones, lo probre con una iluminación blanca enfrente de mi y fondo oscuro
# Con ilúminación potente enfrente de mi me reconoció mejor entre 70cm y 80cm 



#************************************* SECCIÓN 1.- CARGAMOS LOS ARCHIVOS NECESARIOS AL PROGRAMA **********************************

#------------------- Cargamos los rostros codificados

# Esta codificacióon de rostros son vectores numéricos que representan características faciales únicas de personas específicas (de la bd de folders con nombres, como Brandonz).
print('[INFO] loading encodings...')
with open('../face_recognition/encoded_faces.pickle','rb') as file:
    encoded_data = pickle.loads(file.read()) #1


#--------------- Cargar la red neuronal pre-entrenada, la cual solo detecta rostros, NO RECONOCE A LA PERSONA

print('[INFO] loading face detector...')

# Contiene la arquitectura de la red: número de capas, función de activación utilizada en cada capa, tamaño de los filtros convolucionales, entre otros.
proto_path = os.path.sep.join(['face_detector/deploy.prototxt']) 

# Contiene los pesos y los parámetros entrenados del modelo pre-entrenado. Estos parámetros
#son ajustados durante el entrenamiento para que la red neuronal aprenda a detectar rostros en imágenes.
caffemodel_path = os.path.sep.join(['face_detector/res10_300x300_ssd_iter_140000.caffemodel'])

#Se utilizan en conjunto para cargar la red neuronal y utilizarla para hacer predicciones, en este caso se utiliza para realizar la detección de rostros.
detector_net_inicio = cv2.dnn.readNetFromCaffe(proto_path, caffemodel_path) #2
#detector_net_inicio.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
#detector_net_inicio.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)





#*************************** SECCIÓN 1.1.- MODELO REAL / FAKE *************************

#-------------- Carga el modelo que clasifica si un rostro es real o falso
liveness_model = tf.keras.models.load_model('liveness.model') #3

#-------------- Carga la conversión de las etiquetas de texto "real" y "fake" en valores numéricos 0 y 1.
le = pickle.loads(open('label_encoder.pickle', 'rb').read())  #4
    



#**************************************** SECCIÓN 2, RECONOCIMIENTO EN TIEMPO REAL ***************************************

def recognition_liveness(encodings = encoded_data,
                         detector_folder = detector_net_inicio,
                         model_path = liveness_model,
                         le_path = le,
                         confidence=0.7):
        
    args = {'model':model_path, 'le':le_path, 'encodings':encodings, 'confidence':confidence}
    detector_net = detector_folder

    #El While siempre será True por ley de Python
    while True:
        
        try:    #****************************************************************************
                #************* SECCIÓN 2.1.- CONEXIÓN AL SERVER MQTT (RASPBERRY)*************
                
                def on_connect(client, userdata, flags, rc):
                    print("Connected with result code "+str(rc))
                    
                    #Suscribete al siguiente canal del servidor MQTT
                    client.subscribe("/feeds/onoff")
    
                # on_message es un callback, lo que significa que se ejecutará automáticamente cuando un mensaje llege al topic.
                def on_message(client, userdata, msg):
                    print(msg.topic+" "+str(msg.payload))
                    
    
                # Creamos un objeto de la clase mqtt.Client()
                client = mqtt.Client()
                
                # Configuramos las funciones que se ejecutarán cuando el cliente (este pc) se conecte y reciba un mensaje
                client.on_connect = on_connect
                client.on_message = on_message
    
                # Nos conectamos al Servidor, que es la raspbberry, por eso colocamos su IP local.
                client.connect("192.168.43.79", 1883, 60)
                client.publish("/feeds/onoff", "0")        
            
        finally:
            
            #**************************************************************************************
            #***************** SECCIÓN 2.2.- INICIALIZACIÓN DE CÁMARA *****************************
            
            print('[INFO] starting video stream...')
            
            # Iniciar la transmisión de video desde la cámara (src=0)
            vs = VideoStream(src=1, framerate=60).start()
            #vs = VideoStream(src="http://192.168.43.92:81/stream").start() #ESTO ES PARA LA ESP32

            # Inicializar una variable para contar la secuencia en la que aparece la persona reconocida
            sequence_count = 0 
            sequence_count_unknown = 0
            

            name = 'Unknown'
            label_name = 'fake'
            
            #Aquí guardaremos de acuerdo al número de secuencias el nombre de los rostros, por ej, esta lista tendrá 5 "Brandon" o también puede tener 5 "Brandon" y 2 "Ariya"
            nombres = []
            unknown_real_o_falso = []
            preds_mean = []
            
            # Strong_name se refiere al nombre que mas se repite dentro del arreglo "nombres", en otras palabras, guardamos aquí a la persona que mas se reconoció
            strong_name = None
            

            while True:
                
                #*****************************************************************************************************
                # SECCIÓN 2.3.- PREPROCECAMIENTO DE IMÁGENES Y EJECUCIÓN DE LA RED NEURONAL PARA DETECCIÓN DE OBJETOS
                
                # Lee un fotograma del flujo de video y lo redimensiona
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
                
                #cv2.putText(frame, "Press 'q' to quit", (20,35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 2)
                
                # Obtiene la altura y anchura del fotograma del video
                (height, width) = frame.shape[:2]
                
                # la línea preprocesa la imagen para que sea más fácil de procesar para la red neuronal, 1.- se redimensiona el frame
                #2.- Se escala la imágen, como es 1.0, entonces los píxeles van de 0 a 255, 3.- se le dice que la imágen es de (300,300)
                #4.- (104.0, 177.0, 123.0) ayuda a eliminar el efecto de la iluminación en la imagen y normaliza la entrada para la red neuronal.
                # la red neuronal se ha entrenado con el conjunto de datos de FaceNet, y el valor del mean subtraction se ha establecido en (104.0, 177.0, 123.0)
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                
                # Establecemos la entrada a la red neuronal utilizando el objeto blob creado
                detector_net.setInput(blob)
                
                # Ejecuta la red neuronal para realizar inferencias y obtener detecciones y predicciones  
                detections = detector_net.forward()
                
                
                #******************************************************************************************************
                # ***** SECCIÓN 2.4 INTERACIÓN SOBRE LA DETECCIÓN DEL ROSTRO Y EXTRACCIÓN DE REGIONES DE INTERES ******
                
                scale_matrix = np.array([width, height, width, height])
                
                # detections[2] (Tercera dimensión del array) nos da la cantidad de detecciones de rostros encontrado en el video
                for i in range(0, detections.shape[2]):
                    
                    # Se extrae la confianza (probabilidad) de qué la región de interés contiene un rostro i en el video.
                    # "confidence" es una lista de tamaño 1x1xNxD, donde N es el número de detecciones de rostros en el video y D es la dimensión de cada detección, que en este caso es 7.
                    # Las dimensiones de la detección son: 1.- Coordenada del centroide del eje X del rostro detectado, 2.- Coordenada del centroide del eje Y del rostro detectado
                    # 3.- Probabilidad de que la región de interés contenga un rostro, 4.- Coordenada X de arriba a la izq del cuadro de la cara, 5.- Coordenada Y de arriba a la izq del cuadro
                    # 6.- Coordenada en X del punto abajo a la derecha del cuadro de la cara, 7.- Coordenada en Y del punto abajo a la derecha de la cara
                    confidence = detections[0, 0, i, 2]
                    
                    # Se filtran las detecciones con baja confianza.
                    if confidence > args['confidence']:
                        
                        # Convertimos las coordenadas de caja de la localización de la cara a píxeles
                        box = detections[0, 0, i, 3:7] * scale_matrix
                        
                        # Convertimos las coordenadas de caja de la cara en píxeles de Float a Int, esto simplifica cálculos
                        (startX, startY, endX, endY) = box.astype('int')
                        

                        
                        # Recalcando que lo probé con pocos registros, y que con cubrebocas no reconoce
                        # Obtenemos el ancho y alto de la caja delimitadora en píxeles donde box[0], box[1] = X e Y de arripa a la izq, box[2], box[3] = X e Y de abajo a al der de la caja.
                        box_width = box[2] - box[0]
                        box_height = box[3] - box[1]
                        
                        # Estas líneas ajustan las coordenadas de la caja que delimita la cara para que sean un 10% más grandes. 
                        startX = max(0, int(box[0] - 0.1*box_width)) #box[0] = Valor en X arriba izq, - el valor total del ancho de la caja * 0.1 (esto último nos da el 10% del ancho de la caja)
                        startY = max(0, int(box[1] - 0.1*box_height))
                        endX = min(int(box[2] + 0.1*box_width), width)
                        endY = min(int(box[3] + 0.1*box_height), height)
                        
                        face = frame[startY:endY, startX:endX]
                        
                        face_to_recog = face
                         
                    
                        # Si el rostro está fuera del frame y regresa al frame, hay un error, por eso el try except
                        try:
                            # Redimensionamos el rostro que está dentro de la caja delimitadora para adaptarlo a nuestro modelo de livenness detection
                            face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
                        except cv2.error: # Si ocurre un error la intentar redimensionar el rostro, salimos del try except
                            break
                    
                    
                        # Convertimos el rostro recortado a formato RGB para poder utilizar la librería face_recognition
                        rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
                        
                        # utilizamos la librería face_recognition para extraer los encodings del rostro del video
                        encodings = face_recognition.face_encodings(rgb)
                        
                        name = 'Unknown'
                        
                        
                        #********************************************************************************************************
                        #********  SECCIÓN 2.5.- COMPARACIÓN DEL ROSTRO DEL VIDEO Y EL DE LA BD (RECONOCIMIENTO FACIAL) *********
                        
                        # Recorremos los encodigs del rostro o rostros detectados en el video
                        for encoding in encodings:
                            
                            #Compara los encodigs del rostro del video con los almacenados en la BD de rostros previamente entrenada y almacena los resultados en una variable
                            matches = face_recognition.compare_faces(encoded_data['encodings'], encoding)
                            
                            # Si existe coincidencia en el rostro del video y en el de la BD
                            if True in matches:
                                
                                # Crea una lista de índices que indican las posiciones donde se enncontraron las concidencias verdaderas
                                # Sirve para contar el número total de veces que la cara del video hizo "match" con la de la BD
                                matchedIdxs = [i for i, b in enumerate(matches) if b]
                                
                                # Crea un diccionaro para contar la cantidad de veces que coincide la cara del video con la de la BD
                                counts = {}
                                
                                # Recorre los indices SOLAMENTE donde se encontraron coincidencias verdaderas en "matches", si no hay concidencias, no entra en juego este bucle
                                for i in matchedIdxs:
                                    
                                    #Obtenemos del enconded_data (BD) el nombre del rostro detectado en el video, si se reconoció
                                    name = encoded_data['names'][i]
                                    
                                    # Actualizamos le contador de coincidencias entre el rostro del video y el de la BD, 
                                    # Por ej, si "Brandon" ya existe en el diccionario con un valor de 2, el valor "counts" ahorá será 3 si hay coincidencia.
                                    # --------TODO: AQUÍ PODEMOS VALIDAR SI EL NOMBRE CAMBIA EN EL VALOR 2 DE BRANDON A EL VALOR 3 ARIYA, ENTONCES REINCIA EL CONTADOR.
                                    counts[name] = counts.get(name, 0) + 1
                                    
                                # obtenemos el nombre con mayor coincidencias
                                # Iteramos en base al arreglo counts y le decimos que determine el máximo en base a los valores del diccionario "counts" (counts.get)
                                name = max(counts, key=counts.get)
                             
                            
                        #*******************************************************************************************************     
                        #***************** SECCIÓN 2.6.- VERIFICACIÓN REAL/FAKE UTILIZANDO EL LIVENESS.MODEL *******************
                        
                        #face contiene la región dentro de la caja delimitadora, la cual contiene un rostro
                        # Convierte los píxeles de esta imágen/video una escala de 0 a 1
                        face = face.astype('float') / 255.0 
                        
                        # Convierte la imagen del rostro en un array tipo numpy
                        face = tf.keras.preprocessing.image.img_to_array(face)
                        
                        # Agrega una dimensión al array de tipo numpy para que coincida con la entrada requerida por el modelo de liveness
                        face = np.expand_dims(face, axis=0)
                    
                        # Pasa el array de tipo numpy del rostro a través del modelo liveness para determinar si la imagen es real o falsa
                        # predict return 2 value for each example (because in the model we have 2 output classes)
                        # the first value stores the prob of being real, the second value stores the prob of being fake
                        # so argmax will pick the one with highest prob
                        # we care only first output (since we have only 1 input)
                        preds = liveness_model.predict(face)[0]
                        j = np.argmax(preds)
                        
                        # Obtiene el nombre de la clase predicha (real o falsa) a partir del objeto LabelEncoder utilizado para entrenar el modelo.
                        label_name = le.classes_[j]
                        
                        
                        #*************************************************************************************************************
                        #************ SECCIÓN 2.7.- VISUALIZACIÓN DE RESULTADOS DE LA VERIFICACIÓN DE LIVENESS ***********************
                        
                        # Cadena que contiene el nombre de la etiqueta predicha (real o fake) y la probabilidad de esa predicción.
                        label = f'{label_name}: {preds[j]:.4f}'
                        
                        # Si el nombre es "Unknown" o la etiqueta predicha es "fake", se reinicia sequence_count.
                        if name == 'Unknown' or label_name == 'fake':
                            
                            nombres = []
                            sequence_count = 0
                            
                            unknown_real_o_falso.append(label_name)
                            strong_real_falso = max(unknown_real_o_falso, key=unknown_real_o_falso.count)
                            
                            
                            print(unknown_real_o_falso)
                            print(strong_real_falso)
                            
                            sequence_count_unknown += 1
                        
                        # Si el nombre no es "Unknown", la etiqueta predicha es "real" y la probabilidad de esa predicción es mayor o igual a 0.705, se incrementa en +1 secuence_count    
                        elif name != 'Unknown' and label_name == 'real' and preds[j] >= 0.98881:
                            
                            unknown_real_o_falso = []
                            sequence_count_unknown = 0
                            
                            # Agregamos el nombre a la lista "nombres"
                            nombres.append(name)
                            preds_mean.append(preds[j])
                            
                            #mean = np.mean(preds_mean)
                            print(preds_mean)
                            #print(mean)
                            
                            
                            # Extraemos de la lista el nombre que mas se repite en ella, esto hace que si falla el reconocimiento 1 vez con un nombre de 5, entonces no imprime el último
                            # nombre que leyó el reconocimiento facial, sino que se imprime el nombre que mas se reconció a lo largo de la duración del reconocimiento.
                            strong_name = max(nombres, key=nombres.count)
                            
                            sequence_count += 1
                            
                        print(f'[INFO] {name}, {label_name}, seq: {sequence_count}, confidence {preds[j]:.4f}')

                    
                        # Si la etiqueta predicha es "fake", se agrega un mensaje de advertencia en el cuadro del rostro.
                        if label_name == 'fake':
                            cv2.putText(frame, "Rostro Falso", (startX, endY + 25), 
                                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
                        
                        # Agrega el nombre del rostro detectado en el cuadro del rostro.
                        cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
                        
                        # Agrega la etiqueta predicha (Fake o Real) y la probabilidad en el cuadro del rostro.
                        cv2.putText(frame, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # ----------- REVISAR ESTO, CREO QUE YA TENEMOS EL OTRO CUADRO ANTERIOR, DEL FACE RECOGNITION, POR LO QUE PUEDE QUE ESTE NO NOS SIRVA
                        # Agrega un rectángulo en la imagen de salida que rodea la región donde se detectó el rostro.
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
              
                    
                
            #************************************************************************************************************
            #********** SECCIÓN 2.8.- ENVÍO DE RESULTADOS A MQTT, VALIDACIÓN E IMPRESIÓN DE RESULTADOS ****************
                
                
                # name ahora pasa a ser el nombre que mas se reconoció y no el último que se reconoció
                name = strong_name
                
                # Mostramos el frame o frames continuos (o sea el video) y espera que una tecla se presione y obtiene qué tecla se presionó
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(1) & 0xFF
                
                # Si la tecla presioanda es la q
                if key == ord('q'):
                    
                    #Destruye todas las ventanas, retorna un False y rompe el While, pero no la ejecución de la función
                    vs.stop()
                    cv2.destroyAllWindows()
                    return False, False
                    break
                
                if sequence_count==5:
                
                    # Envía el mensaje al canal /feeds/onoff de MQTT y rompe el While, pero no la ejecución de la función
                    mensaje = "1 " + str(name)
                    print(client.publish("/feeds/onoff", mensaje))
                    mean = np.mean(preds_mean)
                    print(mean)
                    break
                
                if sequence_count_unknown == 30:
                
                    # Envía el mensaje al canal /feeds/onoff de MQTT y rompe el While, pero no la ejecución de la función
                    mensaje = "2 Unknown"
                    name = "Unknown"
                    label_name = strong_real_falso
                    
                    if (label_name == "fake"):
                        print("Alguien intentó acceder colocando una fotografía o documento en la cámara")
                    else:
                        print("Alguien desconocido intentó acceder")
                        
                    print(client.publish("/feeds/onoff", mensaje))
                    break
      
            # Si cualquiera de los 2 if de arriba es verdadero y rompe el While, se ejecuta el siguiente bloque de código
            # Destruye todas las ventanas
            vs.stop()
            cv2.destroyAllWindows()

            # (it can f*ck up GPU sometimes if you don't have high performance GPU like me LOL)
            time.sleep(0)
            
            #Despues de tiempo, publica en el canal /feeds/onoff el mensaje 0, (el tiempo de espera se configura en arduino)
            print(client.publish("/feeds/onoff", "0")[1])
            print(name, label_name)
            
            return name, label_name



##################### SECCIÓN 3.- EJECUCIÓN DE LA FUNCIÓN INFINITAMENTE A MENOS QUE SE PRESIONE q ##############################

#Ejecutamos la función infinitamente
while True:
    
    print("Ejecutando de nuevo la función")
    name, label_name = recognition_liveness()

    
    # Si la variable name retorna un False, entonces Rompe el While
    if (name == False):
        print("Salida exitosa")
        break
      