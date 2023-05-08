import os
# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import paho.mqtt.client as mqtt
import tensorflow as tf
import face_recognition
import numpy as np
import warnings
import imutils
import pickle
import time
import cv2


from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from imutils.video import VideoStream
warnings.filterwarnings('ignore')


# OBTENER EL NOMNRE DEL ROSTRO QUE MAS SE DETECTÓ A LO LARGO DEL RECONOCIMINETO LISTO
#### AGREGAR, POR EJEMPLO, SI UN INTRUSO SE DETECTÓ POR 15 ITERACIONES, ENTONCES LO REGISTRA EN LA BD LISTO

#-------------------------------------------- TODO ------------------------------------------------
# OPTIMIZAR LA EJECUCIÓN DEL CÓDIGO
# VALIDAR EN ARDUINO EL MENSAJE 2, YA QUE ENVIAMOS 3 MENSAJES EN TOTAL DEPENDIENDO DE AL CONDICIÓN 0 (VALIDADO), 1 (VALIDADO), 2 (NO VALIDADO)
# MANDAR CORREOS ELECTRÓNICOS SI ALGUIEN INTENTÓ ACCEDER O DE QUIEN ACCEDIÓ
# AUTOMATIZAR EL LABEL ENCODER. TOMANDO LA FOTO DE UNA PERSONA, A LA SIGUIENTE ITERACIÓN DEL CÓDIGO, AGREGA A ESTA NUEVA PERSONA PARA RECONOCERLA, NO HAY PROBLEMA
# POR LA BASE DE DATOS POR EL LABEL ENCODER, YA QUE LA BASE DE DATOS SIMPLEMENTE TOMA EL NOMBRE QUE LE MANDAMOS DESDE AQUÍ


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
    

#************************************************
file = "./resources/anti_spoof_models"


#**************************************** SECCIÓN 2, RECONOCIMIENTO EN TIEMPO REAL ***************************************

def recognition_liveness(encodings = encoded_data,
                         detector_folder = detector_net_inicio,
                         model_path = liveness_model,
                         le_path = le,
                         model_dir = file,
                         device_id = 0):
        
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
            conteo_reales = []
            
            nombre_en_la_foto = []
            conteo_fakes = []
            

            # Strong_name se refiere al nombre que mas se repite dentro del arreglo "nombres", en otras palabras, guardamos aquí a la persona que mas se reconoció
            strong_name = None
            strong_label_name = None
            

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
                            
                            # Agrega el nombre del rostro detectado en el cuadro del rostro.
                            cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
                            
                            # Agrega la etiqueta predicha (Fake o Real) y la probabilidad en el cuadro del rostro.
                            #cv2.putText(frame, label, (startX, startY - 10),
                            #           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # ----------- REVISAR ESTO, CREO QUE YA TENEMOS EL OTRO CUADRO ANTERIOR, DEL FACE RECOGNITION, POR LO QUE PUEDE QUE ESTE NO NOS SIRVA
                            # Agrega un rectángulo en la imagen de salida que rodea la región donde se detectó el rostro.
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
                                
                             


                        image, label, valor_prediccion = test(frame, model_dir, device_id, startX, startY, endX, endY, name)
                        frame = image
                        
                        
                        
                        if name == 'Unknown' or label != 1:
             
                             nombres = []
                             conteo_reales = []
                             sequence_count = 0
                             label_name = "Fake"
                             
                             nombre_en_la_foto.append(name)
                             conteo_fakes.append(label_name)
                             
                             strong_label_name = max(conteo_fakes, key=conteo_fakes.count)
                             strong_name = max(nombre_en_la_foto, key=nombre_en_la_foto.count)
       
                             sequence_count_unknown += 1
                         
                            
                         # Si el nombre no es "Unknown", la etiqueta predicha es "real" y la probabilidad de esa predicción es mayor o igual a 0.705, se incrementa en +1 secuence_count    
                        elif name != 'Unknown' and label == 1 and valor_prediccion >= 0.92:
                             
                             nombre_en_la_foto = []
                             conteo_fakes = []
                             sequence_count_unknown = 0
                             label_name = "Real"
                             
                             # Agregamos el nombre a la lista "nombres"
                             nombres.append(name)
                             conteo_reales.append(label_name)
                             
                             # Extraemos de la lista el nombre que mas se repite en ella, esto hace que si falla el reconocimiento 1 vez con un nombre de 5, entonces no imprime el último
                             # nombre que leyó el reconocimiento facial, sino que se imprime el nombre que mas se reconció a lo largo de la duración del reconocimiento.
                             strong_name = max(nombres, key=nombres.count)
                             strong_label_name = max(conteo_reales, key=conteo_reales.count)
                             
                             sequence_count += 1
                             
                        print(f'[INFO] {name}, {label_name}, seq: {sequence_count}, seq_unk: {sequence_count_unknown} confidence {valor_prediccion}')



                name = strong_name
                label_name = strong_label_name
                
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

                    print("¡HOLA", name.upper(), ", BIENVENIDO!")
                    break
                
                
                if sequence_count_unknown == 30:
                
                    # Envía el mensaje al canal /feeds/onoff de MQTT y rompe el While, pero no la ejecución de la función
                    mensaje = "2 Unknown"
                    
                    
                    
                    if (label_name == "Fake" and name != "Unknown"):
                        
                        print("¡ACCESO DENEGADO! Alguien intentó acceder con una foto o identificación falsa bajo el nombre de", name.upper())
                        
                    elif (label_name == "Fake" and name == "Unknown"):
                        
                        print("!ACCESO DENEGADO¡ Alguien intentó acceder con una foto o identificación de una persona que no está registrada")
                        
                        
                    print(client.publish("/feeds/onoff", mensaje))
                    break
      
        
      
            # Si cualquiera de los 2 if de arriba es verdadero y rompe el While, se ejecuta el siguiente bloque de código
            # Destruye todas las ventanas
            vs.stop()
            cv2.destroyAllWindows()

            # (it can f*ck up GPU sometimes if you don't have high performance GPU like me LOL)
            time.sleep(0)
            
            #Despues de tiempo, publica en el canal /feeds/onoff el mensaje 0, (el tiempo de espera se configura en arduino)
            client.publish("/feeds/onoff", "0")
            print(name, label_name, "\n")
            

            
            return name, label_name





def test(image, model_dir, device_id, start_X, start_Y, end_X, end_Y, name):
    
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    result_text = ""
    color = (255, 0, 0)
    # sum the prediction from single model's result
    
    
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        
        if scale is None:
            param["crop"] = False
            
            
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2 #Value es float de alrededor 10 dígitos
    
    if label == 1:
        
       
        #print("Cara Real. Score: {:.2f}.".format(value) + " Nombre: ", name)
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
        
    else:
        #print("Cara Falsa. Score: {:.2f}.".format(value) + " Nombre: ", name)
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    #print("Prediction cost {:.2f} s".format(test_speed))
    
    cv2.putText(image, name, (start_X, start_Y - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
    cv2.putText(image, result_text, (start_X, start_Y - 10),
               cv2.FONT_HERSHEY_COMPLEX, 0.6, color)
    cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y), color, 2)

    
    return image, label, value





##################### SECCIÓN 3.- EJECUCIÓN DE LA FUNCIÓN INFINITAMENTE A MENOS QUE SE PRESIONE q ##############################

#Ejecutamos la función infinitamente
while True:
    
    print("Ejecutando de nuevo la función")
    name, label_name = recognition_liveness()

    
    # Si la variable name retorna un False, entonces Rompe el While
    if (name == False):
        print("Salida exitosa")
        break
      
        
      
        
      
        