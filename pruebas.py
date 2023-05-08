# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import paho.mqtt.client as mqtt
from PIL import ImageTk, Image
import tensorflow as tf
import face_recognition
import tkinter as tk
import numpy as np
import warnings
import imutils
import pickle
import time
import cv2
import sys
import copy
import os

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from imutils.video import VideoStream

sys.path.append('../face_recognition')
import encode_faces as ef_script
#ef_script.codifica_caras()

warnings.filterwarnings('ignore')


# OBTENER EL NOMNRE DEL ROSTRO QUE MAS SE DETECTÓ A LO LARGO DEL RECONOCIMINETO LISTO
#### AGREGAR, POR EJEMPLO, SI UN INTRUSO SE DETECTÓ POR 15 ITERACIONES, ENTONCES LO REGISTRA EN LA BD LISTO

#-------------------------------------------- TODO ------------------------------------------------
# OPTIMIZAR LA EJECUCIÓN DEL CÓDIGO
# VALIDAR EN ARDUINO EL MENSAJE 2, YA QUE ENVIAMOS 3 MENSAJES EN TOTAL DEPENDIENDO DE AL CONDICIÓN 0 (VALIDADO), 1 (VALIDADO), 2 (NO VALIDADO)
# MANDAR CORREOS ELECTRÓNICOS SI ALGUIEN INTENTÓ ACCEDER O DE QUIEN ACCEDIÓ
# QUE DESPUES DE CIERTO TIEMPO SE LIMPIE LOS ARREGLOS CON LOS NOMBRES ------------------------- PRIORIDAD!!!!!!!!!!!!!!!!!!!!!!!!
#CCOMROBAR COMO FUNCIONA CON MUCHAS FOTOS DE UNA PERSONA EN UNA CARPETA



#-----------OPCIONAL, PRIMERO DEBE CONSIDERARSE
# si solo necesitas reconocer un número limitado de personas, podrías pre-calcular los encodin de las personas y almacenarlos en una base de datos en lugar de calcularlos en 
# tiempo real en cada cuadro del video. De esta manera, solo necesitarías comparar los encodin de los rostros detectados con los encodin almacenados en la base de datos en
# lugar de calcular todos los encodin en cada cuadro. Esto puede ser significativamente más rápido y escalable en situaciones donde hay muchas personas que se deben reconocer.
#Todo lo anterior, en la última parte de la SECCIÓN 2.4
# SIMPLEMENTE CREAR UNA BASE DE DATOS

#--------------------------------- IMPORTANTE -------------------------------------------------------------------
#El liveness detection(real / fake) funciona mejor a una distancia de unos 80cm despegado de la cámara
# habrá que ver como funciona con distintas iluminaciones, lo probre con una iluminación blanca enfrente de mi y fondo oscuro
# Con ilúminación potente enfrente de mi me reconoció mejor entre 70cm y 80cm 


img2 = Image.open('cuadro_facer.png')
img2 = img2.resize((240, 210))

img3 = Image.open('../face_recognition/dataset/Brandon/Brandon.jpg') 
img3 = img3.resize((180, 140))

img4 = Image.open('no_registrado.png')
img4 = img4.resize((180, 140))

img5 = Image.open('foto_falsa.png')
img5 = img5.resize((180, 140))

#start_time = time.time()
#************************************* SECCIÓN 1.- CARGAMOS LOS ARCHIVOS NECESARIOS AL PROGRAMA **********************************

#------------------- Cargamos los rostros codificados

# Esta codificacióon de rostros son vectores numéricos que representan características faciales únicas de personas específicas (de la bd de folders con nombres, como Brandonz).



    
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

     
     
def recognition_liveness(detector_folder = detector_net_inicio,
                         model_path = liveness_model,
                         le_path = le,
                         model_dir = file,
                         device_id = 0):
        
    detector_net = detector_folder
    boton = False
    
    
    global encoded_data
    global encodings
    print('[INFO] loading encodin...')
    with open('../face_recognition/encoded_faces.pickle','rb') as file:
        encoded_data = pickle.loads(file.read()) #1

    caras_en_encode_faces = len(encoded_data['names'])
    print(encoded_data["names"])
    encodings = encoded_data
    

    while not boton:
        
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
            vs = VideoStream(src=0, framerate=60).start()
            #vs = VideoStream(src="http://192.168.43.92:81/stream").start() #ESTO ES PARA LA ESP32
            
            frame3 = vs.read()
            frame3 = imutils.resize(frame3, width=600)
            
            
            sequence_count = 0 
            sequence_count_foto = 0
            sequence_count_unknown = 0
            
            name = 'Unknown'
            label_name = 'Fake'
            text = ""
                    
            nombres = []
            conteo_reales = []
            nombre_en_la_foto = []
            conteo_fakes = []
            
            strong_name = None
            strong_label_name = None
            
            registro_exitoso = False
            no_registrado = False
            foto_falsa = False
            
            start_time = time.time()

            current_time = 0
            elapsed_time = 0
            limit_time = 0 
            
            img3 = None
            i3 = None
            
            milista = [1]
            frame2 = np.array(milista)
            
            #caras_en_encode_faces = num_caras


            
            
            while not boton:
                
                #*****************************************************************************************************
                # SECCIÓN 2.3.- PREPROCECAMIENTO DE IMÁGENES Y EJECUCIÓN DE LA RED NEURONAL PARA DETECCIÓN DE OBJETOS
                root = tk.Tk()
                root.title("Reconocimiento facial")

                # crear canvas para mostrar imagen
                canvas = tk.Canvas(root, width=700, height=600)
                canvas.pack()

                name_label2 = tk.Label(root, text=text)
                

                # crear etiqueta para mostrar nombre
                #name_label = tk.Label(root, text="Desconocido") 
                #name_label.pack()
                
                
                    
                #Mostramos la imágen del cuadro de la cara
                #imgtk2 = ImageTk.PhotoImage(image=img2)
                #canvas.imgtk2 = imgtk2
                #canvas.create_image(200, 115, anchor=tk.CENTER, image=imgtk2)
                
                
                
            
                
                
                #######################################################################################################
                
                def update_image( name=name, label_name=label_name, strong_name=strong_name, strong_label_name=strong_label_name):
                    # leer frame de la cámara
                    # Inicializar una variable para contar la secuencia en la que aparece la persona reconocida

                    nonlocal sequence_count
                    nonlocal sequence_count_foto
                    nonlocal sequence_count_unknown
                    
                    #nonlocal name
                    nonlocal name_label2
                    
                    nonlocal nombres
                    nonlocal conteo_reales
                    nonlocal nombre_en_la_foto
                    nonlocal conteo_fakes
                    
                    nonlocal registro_exitoso
                    nonlocal no_registrado
                    nonlocal foto_falsa
                    
                    nonlocal current_time
                    nonlocal elapsed_time
                    nonlocal start_time
                    nonlocal limit_time
                    
                    nonlocal i3
                    nonlocal img3
                    
                    nonlocal text
                    
                    nonlocal frame2
                    
                    nonlocal caras_en_encode_faces
                    
                    global encoded_data
                    path = "../face_recognition/dataset"
                    num_carpetas = len([dir for dir in os.listdir(path)])
                    
                    if num_carpetas != caras_en_encode_faces:
                        print("El número de registros ha cambiado! actualizando registros...")                        
                        caras_en_encode_faces = ef_script.codifica_caras()
                        with open('../face_recognition/encoded_faces.pickle','rb') as file:
                            encoded_data = pickle.loads(file.read())
                        nombres = []
                        conteo_reales = []
                        nombre_en_la_foto = []
                        conteo_fakes = []
                        sequence_count = 0 
                        sequence_count_foto = 0
                        sequence_count_unknown = 0
                        name = 'Unknown'
                        label_name = 'Fake'



                    current_time = time.time()
                    elapsed_time = current_time - start_time
   
                    
                    frame = vs.read()
                    frame = imutils.resize(frame, width=700)
                    #height_resized, width_resized, channels_resized = frame.shape
                    frame2 = copy.deepcopy(frame)
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
                        
                        # Estas líneas ajustan las coordenadas de la caja que delimita la cara para que sean un 2% más grandes. 
                        startX = max(0, int(box[0] - 0.02*box_width)) #box[0] = Valor en X arriba izq, - el valor total del ancho de la caja * 0.1 (esto último nos da el 10% del ancho de la caja)
                        startY = max(0, int(box[1] - 0.02*box_height))
                        endX = min(int(box[2] + 0.02*box_width), width)
                        endY = min(int(box[3] + 0.02*box_height), height)
                        
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
                        

                        # utilizamos la librería face_recognition para extraer los encodin del rostro del video
                        encodings = face_recognition.face_encodings(rgb)
                        
                        #name = 'Unknown'
                        
                        
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
                                #cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
                                
                                # Agrega la etiqueta predicha (Fake o Real) y la probabilidad en el cuadro del rostro.
                                #cv2.putText(frame, label, (startX, startY - 10),
                                #           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                                
                                # ----------- REVISAR ESTO, CREO QUE YA TENEMOS EL OTRO CUADRO ANTERIOR, DEL FACE RECOGNITION, POR LO QUE PUEDE QUE ESTE NO NOS SIRVA
                                # Agrega un rectángulo en la imagen de salida que rodea la región donde se detectó el rostro.
                                #cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 4)
          
                                
######################################################################################################################


                            image, label, valor_prediccion = test(frame, model_dir, device_id, startX, startY, endX, endY, name)
                            frame = image
                            
                            
                                    
                                
                            
                            print(name, label)
                            
                             # Si el nombre no es "Unknown", la etiqueta predicha es "real" y la probabilidad de esa predicción es mayor o igual a 0.705, se incrementa en +1 secuence_count    
                            if name != 'Unknown' and label == 1 and valor_prediccion >= 0.92:
                                 
                                 nombre_en_la_foto = []
                                 conteo_fakes = []
                                 sequence_count_unknown = 0
                                 sequence_count_foto = 0
                                 label_name = "Real"
                                 
                                 # Agregamos el nombre a la lista "nombres"
                                 nombres.append(name)
                                 conteo_reales.append(label_name)
                                 print(nombres)
                                 print(conteo_reales)
                                 
                                 # Extraemos de la lista el nombre que mas se repite en ella, esto hace que si falla el reconocimiento 1 vez con un nombre de 5, entonces no imprime el último
                                 # nombre que leyó el reconocimiento facial, sino que se imprime el nombre que mas se reconció a lo largo de la duración del reconocimiento.
                                 strong_name = max(nombres, key=nombres.count)
                                 strong_label_name = max(conteo_reales, key=conteo_reales.count)
                                 

                                 if name == strong_name:
                                     sequence_count += 1
                                 else:
                                     nombres = []
                                     conteo_reales = []
                                     sequence_count = 0
                                 
                            if name == 'Unknown' and label == 1 and valor_prediccion >= 0.73: #Si la persona se desconoce y es verdadera
                 
                                 nombres = []
                                 conteo_reales = []
                                 sequence_count = 0
                                 sequence_count_foto = 0
                                 
                                 label_name = "Real"
                                 
                                 nombre_en_la_foto.append(name)
                                 conteo_fakes.append(label_name)
                                 print(nombre_en_la_foto)
                                 print(conteo_fakes)

                                 
                                 strong_label_name = max(conteo_fakes, key=conteo_fakes.count)
                                 strong_name = max(nombre_en_la_foto, key=nombre_en_la_foto.count)
                                 
                                 if "Fake" in conteo_fakes:
                                     conteo_fakes = []
                                     
                                     
                                 if name == strong_name:
                                   sequence_count_unknown += 1
                                 else:
                                   nombre_en_la_foto = []
                                   conteo_fakes = []
                                   sequence_count_unknown = 0
                                     
           
                                 #sequence_count_unknown += 1
                            
                            #Si es diferente de 1 significa que es fake, el problema es que aveces label es 1 aún que label_name
                            #diga que la foto es falsa, aún no entiendo por qué                            
                            if (label != 1 and valor_prediccion >= 0.73) or (label_name == "Fake" and valor_prediccion >= 0.73): 
                 
                                 nombres = []
                                 conteo_reales = []
                                 sequence_count = 0
                                 sequence_count_unknown = 0
                                 label_name = "Fake"
                                 
                                 nombre_en_la_foto.append(name)
                                 conteo_fakes.append(label_name)
                                 print(nombre_en_la_foto)
                                 print(conteo_fakes)
                                 
                                 strong_label_name = max(conteo_fakes, key=conteo_fakes.count)
                                 strong_name = max(nombre_en_la_foto, key=nombre_en_la_foto.count)
                                 print(strong_name)
                                 
                                 if "Real" in conteo_fakes:
                                     conteo_fakes = []
                                     
                                 if name == strong_name:
                                  sequence_count_foto += 1
                                 else:
                                  nombre_en_la_foto = []
                                  conteo_fakes = []
                                  sequence_count_foto = 0
                                     
           
                                 #sequence_count_foto += 1

                                 
                            print(f'[INFO] {name}, {label_name}, seq: {sequence_count}, seq_unk: {sequence_count_unknown}, seq_foto {sequence_count_foto} confidence {valor_prediccion}')
                        
                        name = strong_name
                        label_name = strong_label_name
    
                        if sequence_count == 5:
                        
                            nombres = []
                            conteo_reales = []
                            sequence_count = 0
                            
                            if registro_exitoso == False and no_registrado == False and foto_falsa == False:
                                registro_exitoso = True
                                limit_time = elapsed_time + 10
                            
                                # Envía el mensaje al canal /feeds/onoff de MQTT y rompe el While, pero no la ejecución de la función
                                mensaje = "1 " + str(name)
                                print(client.publish("/feeds/onoff", mensaje))
        
                                text = "¡HOLA " + name.upper() + ", BIENVENIDO!"
                                print(text)
                                
                                name_copy = name[:]
                                #print(name_copy)
                                directorio_imagen = "../face_recognition/dataset/" + name_copy +"/" + name_copy + ".jpg"
                                img3 = Image.open(directorio_imagen)
                                img3 = img3.resize((180, 140))
                            else:
                                print("")
                          
                        
                            #client.publish("/feeds/onoff", "0")
                            break
   
                        
                        if sequence_count_unknown == 12:
                        
                            nombre_en_la_foto = []
                            conteo_fakes = []
                            sequence_count_unknown = 0
                            
                            if registro_exitoso == False and no_registrado == False and foto_falsa == False:
                                no_registrado = True
                                limit_time = elapsed_time + 10
                                
                                text = "La persona que intentó acceder no se encuentra registrada."
                                print(text)
                   
                                mensaje = "2 Unknown"            
                                print(client.publish("/feeds/onoff", mensaje))
                                                          
                                break
                            else:
                                print("")

                        
                        if sequence_count_foto == 12:
                        
                            nombre_en_la_foto = []
                            conteo_fakes = []
                            sequence_count_foto = 0
                            
                            if registro_exitoso == False and no_registrado == False and foto_falsa == False:
                                foto_falsa = True
                                limit_time = elapsed_time + 10
    
                                if (label_name == "Fake" and name != "Unknown"):
                                    
                                    text = "¡ACCESO DENEGADO! Alguien intentó acceder con una foto o identificación falsa bajo el nombre de: " + name.upper()
                                    print(text)
                                    
                                elif (label_name == "Fake" and name == "Unknown"):
                                    
                                    text = "!ACCESO DENEGADO¡ Alguien intentó acceder con una foto o identificación de una persona que no está registrada"
                                    print(text)  
                                    
                                # Envía el mensaje al canal /feeds/onoff de MQTT y rompe el While, pero no la ejecución de la función
                                mensaje = "3 Unknown"
                                print(client.publish("/feeds/onoff", mensaje))
                            else:
                                print("")

                                
                                
                                break


                    
###############################################################################################################     
                    
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    canvas.imgtk = imgtk
                    tag = canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    canvas.tag_lower(tag)
                    
                    #name_label.config(text=name)    
                    


                    #name_label2 = None

                    def imagen(w, h, imagen):
                        imgtk3 = ImageTk.PhotoImage(image=imagen)
                        (height, width) = frame.shape[:2]
                        width_img, height_img = imagen.size
                        canvas.imgtk3 = imgtk3
                        i3 = canvas.create_image(w, height + height_img, anchor=tk.SW, image=imgtk3)
                        canvas.tag_raise(i3)
                        
                        return i3
                    
                    def texto(X, imagen, texto=text):
                        (height, width) = frame.shape[:2]
                        width_img, height_img = imagen.size
                        name_label2.config(text=text, wraplength=270, font=("Helvetica", 9), justify="center")
                        name_label2.place(x=X, y=height+height_img/3)
                        
                        return name_label2
                                              
                    if registro_exitoso == True: 
                        
                        #if name is not None:
                        i3 = imagen(0, 320, img3)
                        name_label2 = texto(200, img3)
                        
                    if no_registrado == True:   
                        i3 = imagen(-32, 355, img4)
                        name_label2 = texto(120, img4)
                        
                    if foto_falsa == True:
                        i3 = imagen(-37, 350, img5)
                        name_label2 = texto(110, img5)

                    
                        
  
                    #print (elapsed_time, limit_time, registro_exitoso)
                    if limit_time < elapsed_time:    
                        start_time = time.time()
                        
                        registro_exitoso = False
                        no_registrado = False
                        foto_falsa = False
                        
                        canvas.delete(i3)
                        
                        name_label2.config(text="", width=0, height=0) 

        

                    root.after(10, update_image)
                    return frame2
                
                #######################################################################################################
                def para():
                    # Crear la ventana secundaria
                    ventana_secundaria = tk.Toplevel()
                    ventana_secundaria.title("Capturar imagen")
                    
                
                    # Función para actualizar la imagen en la etiqueta
                    def update_image2():
                        global frame2
                        imagen_tk = ImageTk.PhotoImage(imagen)
                        label_imagen.configure(image=imagen_tk)
                        label_imagen.image = imagen_tk
                        ventana_secundaria.after(10, update_image2)
                
                    def guardar_imagen():
                        global encoded_data
                        global caras_en_encode_faces
                        nombre_archivo = cuadro_texto.get()
                        ruta_carpeta = "../face_recognition/dataset"

                        carpeta = os.path.join(ruta_carpeta, nombre_archivo)
                        if not os.path.exists(carpeta):
                            os.makedirs(carpeta)
                            print("Carpeta creada con éxito")
                        
                            ruta_imagen = "../face_recognition/dataset/" + nombre_archivo    
                            imagen = os.path.join(ruta_imagen, nombre_archivo + ".jpg")
                            cv2.imwrite(imagen, frame_copy)
                            
                            #caras_en_encode_faces = ef_script.codifica_caras()
                            #with open('../face_recognition/encoded_faces.pickle','rb') as file:
                                #encoded_data2 = pickle.loads(file.read())
                                #encoded_data = encoded_data2
                            
                        else:
                            print("La carpeta que intenta crear, ya existe")
                            
                        ventana_secundaria.destroy()
                
                
                
                    # Mostrar el fotograma en la etiqueta
                    imagen = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    imagen = Image.fromarray(imagen)
                    imagen_tk = ImageTk.PhotoImage(imagen)
                    label_imagen = tk.Label(ventana_secundaria, image=imagen_tk)
                    label_imagen.pack()
                    frame_copy = np.copy(frame2)
                    # Agregar el cuadro de texto para que el usuario ingrese el nombre de la imagen
                    cuadro_texto = tk.Entry(ventana_secundaria, width=30)
                    cuadro_texto.pack()
                
                    # Agregar el botón para guardar la imagen
                    boton_guardar = tk.Button(ventana_secundaria, text="Guardar", command=guardar_imagen, width=17, height=2, font=("Helvetica", 10), bg="lightgray", fg="black", borderwidth=1, relief="ridge")
                    boton_guardar.pack()
                
                    # Función que cierra la ventana y detiene la actualización de la imagen
                    def cerrar_ventana():
                        vs.stop()
                        ventana_secundaria.destroy()
                
                    # Agregar un evento para cerrar la ventana al presionar la tecla Esc
                    ventana_secundaria.bind("<Escape>", lambda event: cerrar_ventana())
                
                    # Llamar a la función para actualizar la imagen
                    update_image2()
                
                    # Iniciar la ventana secundaria
                    ventana_secundaria.mainloop()

    

                button = tk.Button(root, text="Tomar foto", command=para, width=20, height=2, font=("Helvetica", 11), bg="lavender", fg="black", borderwidth=1, relief="ridge")
                button.pack()

                ###############################################################################
                
                update_image()
                root.mainloop()
                break

                

##################### ESTE CÓDIGO NO SE ESTÁ EJECUTANDO DEBIDO AL root.mainloop() #####################################################
      
            # Si cualquiera de los 2 if de arriba es verdadero y rompe el While, se ejecuta el siguiente bloque de código
            # Destruye todas las ventanas
            vs.stop()
            cv2.destroyAllWindows()

            # (it can f*ck up GPU sometimes if you don't have high performance GPU like me LOL)

            print(name, label_name, "\n")       
            
            return False




#########################################################################################################################

def test(image, model_dir, device_id, start_X, start_Y, end_X, end_Y, name):
    
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    #result_text = ""
    #color = (255, 0, 0)
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
    
    #if label == 1:
        
       
        #print("Cara Real. Score: {:.2f}.".format(value) + " Nombre: ", name)
        #result_text = "RealFace Score: {:.2f}".format(value)
        #color = (255, 0, 0)
        
    #else:
        #print("Cara Falsa. Score: {:.2f}.".format(value) + " Nombre: ", name)
        #result_text = "FakeFace Score: {:.2f}".format(value)
        #color = (0, 0, 255)
        
    #print("Prediction cost {:.2f} s".format(test_speed))
    
    #cv2.putText(image, name, (start_X, start_Y - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
    #cv2.putText(image, result_text, (start_X, start_Y - 10),
               #cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
    #cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y), color, 2)
    cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y), (255, 0, 0), 2)

    
    return image, label, value





##################### SECCIÓN 3.- EJECUCIÓN DE LA FUNCIÓN INFINITAMENTE A MENOS QUE SE PRESIONE q ##############################

#Ejecutamos la función infinitamente
while True:
    
    print("Ejecutando de nuevo la función")
    name= recognition_liveness()

    
    # Si la variable name retorna un False, entonces Rompe el While
    if (name == False):
        print("Salida exitosa")
        break






#######################################################################################
"""import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Configurar los detalles del servidor SMTP de Gmail y las credenciales de inicio de sesión
smtp_server = 'smtp.gmail.com'
smtp_port = 465
smtp_username = 'facer.contacto@gmail.com'
smtp_password = ''

# Establecer los detalles del mensaje
sender = 'facer.contacto@gmail.com'
recipient = 'andrikramirez1123@gmail.com'
subject = 'Prueba desde Python'
body = 'Hola, este es un mensaje enviado desde la computadora de Brandon mediante python'

# Construir el mensaje
message = MIMEMultipart()
message['From'] = sender
message['To'] = recipient
message['Subject'] = subject
message.attach(MIMEText(body))

# Adjuntar una imagen al mensaje (opcional)
with open('no_registrado.png', 'rb') as f:
    image = MIMEImage(f.read())
    message.attach(image)

# Conectar con el servidor SMTP y enviar el mensaje
with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
    server.login(smtp_username, smtp_password)
    server.sendmail(sender, recipient, message.as_string())

print('El correo electrónico se ha enviado correctamente.')
"""
        
      
        