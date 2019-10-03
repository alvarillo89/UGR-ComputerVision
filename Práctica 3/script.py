# -*- coding: utf-8 -*-

##############################################################################
# Visión por Computador
# Trabajo 3: Indexación y recuperación de imágenes.
# @author Álvaro Fernández García
##############################################################################


import numpy as np
import cv2
import auxFunc as axf
import matplotlib.pyplot as plt
import pickle


# Muestra una imgen en pantalla. La imagen se recibe como una matriz:
def _showImage(img, title='Imagen'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Concatena varias imágenes en una sola:
def PintaMI(vim, title):
    assert len(vim) > 0
    if len(vim) == 1:
        finalImg = vim
    else:
        finalImg = vim[0]
        vim.pop(0)
    for img in vim:
        finalImg = cv2.hconcat((finalImg, img))
    _showImage(finalImg, title)
    return finalImg


# Muestra en una imagen 10 parches de 24x24 pixeles:
def dibuja10parches(parches):
    f, ax = plt.subplots(2,5)

    for i,a in enumerate(ax.flatten()):
        a.imshow(parches[i], 'gray')
        a.set_xticks([])
        a.set_yticks([])
    
    plt.tight_layout()
    plt.show()        


#######################################################################################
# EJERCICIO 1:
#######################################################################################

# 1 .- Emparejamiento de descriptores [4 puntos]
#  * Mirar las imágenes en imagenesIR.rar y elegir parejas de imágenes
# que tengan partes de escena comunes. Haciendo uso de una máscara
# binaria o de las funciones extractRegion() y clickAndDraw(), seleccionar 
# una región en la primera imagen que esté presente en la segunda imagen. 
# Para ello solo hay que fijar los vértices de un polígono que contenga 
# a la región.
#  * Extraiga los puntos SIFT contenidos en la región seleccionada de la
# primera imagen y calcule las correspondencias con todos los puntos
# SIFT de la segunda imagen (ayuda: use el concepto de máscara con
# el parámetro mask).
#  * Pinte las correspondencias encontrados sobre las imágenes.
#  * Jugar con distintas parejas de imágenes, valorar las correspondencias
# correctas obtenidas y extraer conclusiones respecto a la utilidad de
# esta aproximación de recuperación de regiones/objetos de interés a
# partir de descriptores de una región.

def Ejercicio1():
    # Declaramos los objetos necesarios:
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=True)

    # Crear las parejas de imagenes:
    parejas = [
        (cv2.imread('imagenes/55.png', 1), cv2.imread('imagenes/59.png', 1)),
        (cv2.imread('imagenes/229.png', 1), cv2.imread('imagenes/248.png', 1)),
        (cv2.imread('imagenes/71.png', 1), cv2.imread('imagenes/88.png', 1))
    ]

    for par in parejas:
        img1 = par[0]
        img2 = par[1]

        # Crear la máscara de la región extraída:
        refPts = np.array(axf.extractRegion(img1))
        mask = np.zeros((img1.shape[0], img1.shape[1]), np.uint8)
        mask = cv2.fillConvexPoly(mask, refPts, (255,255,255)) 

        # Extraer los puntos y descriptores:
        kp1, des1 = sift.detectAndCompute(img1, mask)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Extraer las correspondencias:
        matches = bf.match(des1, des2)
            
        # Dibujarlas:
        out = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        _showImage(out)


#######################################################################################
# EJERCICIO 2:
#######################################################################################

# 2. Recuperación de imágenes [4 puntos]
#   • Implementar un modelo de índice invertido + bolsa de palabras para
# las imágenes dadas en imagenesIR.rar usando el vocabulario dado
# en kmeanscenters2000.pkl.
#   • Verificar que el modelo construido para cada imagen permite recu-
# perar imágenes de la misma escena cuando la comparamos al resto
# de imágenes de la base de datos.
#   • Elegir dos imágenes-pregunta en las se ponga de manifiesto que el
# modelo usado es realmente muy efectivo para extraer sus semejantes y
# elegir otra imagen-pregunta en la que se muestre que el modelo puede
# realmente fallar. Para ello muestre las cinco imágenes más semejantes
# de cada una de las imágenes-pregunta seleccionadas usando como
# medida de distancia el producto escalar normalizado de sus vectores
# de bolsa de palabras.
#   • Explicar qué conclusiones obtiene de este experimento.

# Función que construye el índice invertido y la bolsa de palabras:
def Ejercicio2():
    # Cargar los centroides y crear el detector y matcher:
    dicc = axf.loadDictionary('kmeanscenters2000.pkl')[2]
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=False)

    # Inicializar el índice invertido:
    indice_invert = dict()

    for i in range(dicc.shape[0]):
        indice_invert[i] = set()

    # Inicializar la bolsa de palabras:
    bolsa_palabras = dict()

    for i in range(441):
        bolsa_palabras[i] = np.zeros(dicc.shape[0], np.int)

    # Para cada imagen:
    for i in range(441):
        # Leemos la imagen
        img_name = 'imagenes/' + str(i) + '.png'
        img = cv2.imread(img_name, 1)

        # Obtener los descriptores:
        des = sift.detectAndCompute(img, None)[1]

        # Normalizar los descriptores:
        des_normalized = []
        for d in des:
            norm = np.linalg.norm(d)
            # Ignoramos los descriptores nulos:
            if norm != 0:
                des_normalized.append(d * (1/norm))

        des = np.array(des_normalized)

        # Extraer los matches:
        matches = bf.match(des, dicc)

        # Para cada match:
        for m in matches:
            # Actualizar el indice invertido:
            indice_invert[m.trainIdx].add(i)

            # Actualizar los histogramas de la bolsa de palabras:
            bolsa_palabras[i][m.trainIdx] += 1

    # Normalizar los histogramas:
    for i in range(441):
        bolsa_palabras[i] = bolsa_palabras[i] / np.linalg.norm(bolsa_palabras[i])

    return indice_invert, bolsa_palabras


# Dada una imagen img, y la bolsa de palabras calculada con la función anterior,
# muestra las 5 imágenes más cercanas:
def get_5_nearest_images(img, bolsa_palabras):
    # Paso 1: obtener la bolsa de palabras de la imagen:
    # Declaramos los objetos necesarios:
    dicc = axf.loadDictionary('kmeanscenters2000.pkl')[2]
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=False)

    # Inicializar el histograma:
    q = np.zeros(dicc.shape[0])

    # Extraer los descriptores de la imagen:
    des = sift.detectAndCompute(img, None)[1]

    # Normalizar los descriptores:
    des_normalized = []
    for d in des:
        norm = np.linalg.norm(d)
        # Ignoramos los descriptores nulos:
        if norm != 0:
            des_normalized.append(d * (1/norm))

    des = np.array(des_normalized)

    # Extraer los matches:
    matches = bf.match(des, dicc)

    # Actualizar el histograma:
    for m in matches:
        q[m.trainIdx] += 1

    # Normalizarlo:
    q = q / np.linalg.norm(q)

    # Paso 2: Obtener las 5 imágenes más cercanas: 
    # para ello utilizamos la similaridad:
    def sim(I, J):
        # Como están normalizados no es necesario dividir entre el producto de las normas:
        return (I * J).sum()

    # Calcular las similaridades:
    similaridades = []
    for i in range(len(bolsa_palabras)):
        similaridades.append((i, sim(bolsa_palabras[i], q)))

    # Ordenamos las similaridades:
    similaridades = sorted(similaridades, key = lambda x:x[1], reverse=True)

    # Mostrar las 5 imágenes más cercanas: (la posición 0 es la propia imagen, por eso nos la saltamos)
    for i in range(1,6):
        ima = cv2.imread('imagenes/' + str(similaridades[i][0]) + '.png', 1)
        PintaMI([img, ima], "{}ª imagen más cercana: Similaridad = {:.4f}".format(i, similaridades[i][1]))
        

#######################################################################################
# EJERCICIO 3:
#######################################################################################

# Visualización del vocabulario [3 puntos]
#   • Usando las imágenes dadas en imagenesIR.rar se han extraido 600
# regiones de cada imagen de forma directa y se han re-escalado en
# parches de 24x24 píxeles. A partir de ellas se ha construido un 
# vocabulario de 5.000 palabras usando k-means. Los ficheros con los datos
# son descriptorsAndpatches2000.pkl (descriptores de las regiones
# y los parches extraídos) y kmeanscenters2000.pkl (vocabulario 
# extraído).
#   • Elegir al menos dos palabras visuales diferentes y visualizar las 
# regiones imagen de los 10 parches más cercanos de cada palabra visual,
# de forma que se muestre el contenido visual que codifican (mejor en
# niveles de gris).
#   • Explicar si lo que se ha obtenido es realmente lo esperado en términos
# de cercanía visual de los parches.

def Ejercicio3(vocabulario, words):
    # Cargamos los archivos proporcionados:
    dicc = vocabulario
    des, patches = axf.loadAux('descriptorsAndpatches2000.pkl', True)

    # Declaramos el Matcher:
    bf = cv2.BFMatcher(crossCheck=False)

    # Obtenemos los 10 matches más cercanos:
    matches = bf.knnMatch(dicc, des, k=10)

    # Mostrar los ejemplos:
    # en los dos primeros matches los parches son muy parecidos (casi idénticos)
    # en el tercero, hay ligeras variaciones
    # en el último son distintos:
    for i in words:
        print(i)
        myPatches = []
        for m in matches[i]:
            # Convertirlo a escala de grises y hacer el reshape del parche:
            img = cv2.cvtColor(patches[m.trainIdx], cv2.COLOR_BGR2GRAY).reshape(24,24)
            myPatches.append(img)

        dibuja10parches(myPatches)


#######################################################################################
# BONUS:
#######################################################################################

# Ejercicio 2: Creación de un vocabulario [2 puntos]: Calcular desde todas las 
# imagenesIR.rar los ficheros dados en el Ejercicio 2 usando los mismos parámetros. 
# Aplicar con el nuevo diccionario lo pedido con el Ejercicio 3.

def buildVocabulary():
    # Crear el detector SIFT:
    sift = cv2.xfeatures2d.SIFT_create(600)

    img = cv2.imread('imagenes/0.png', 1)
    des = sift.detectAndCompute(img, None)[1]
    descriptors = des

    # Para cada imagen:
    for i in range(1, 441):
        # Leemos la imagen
        img_name = 'imagenes/' + str(i) + '.png'
        img = cv2.imread(img_name, 1)

        # Obtener los descriptores:
        des = sift.detectAndCompute(img, None)[1]

        # Añadirlos: cada fila de la matriz es un descriptor:
        descriptors = np.vstack((descriptors, des))

    # Lo convertimos a float 32 (lo requiere el método kmeans):
    descriptors = np.float32(descriptors)

    # Realizamos el clustering para extraer el vocabulario:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    compactness, labels, centers = cv2.kmeans(descriptors, 2000, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Descomentar esta línea para guardar el vocabulario en un fichero:
    # pickle.dump(centers, open("myVocabulary.pkl", "wb"))

    return centers


#######################################################################################
# PRUEBA DE LAS FUNCIONES:
#######################################################################################

Ejercicio1()

# Ejercicio 2:
invert, bag = Ejercicio2()

# Casos favorables:
img = cv2.imread('imagenes/89.png',1 )
get_5_nearest_images(img, bag)

img = cv2.imread('imagenes/356.png',1 )
get_5_nearest_images(img, bag)

# Caso desfavorable:
img = cv2.imread('imagenes/78.png',1 )
get_5_nearest_images(img, bag)

voc = axf.loadDictionary('kmeanscenters2000.pkl')[2]
# El segundo parámetro son las palabras a visualizar:
Ejercicio3(voc, (76, 32, 36, 80))

#######################################################################################
# PRUEBA DEL BONUS:
#######################################################################################

# Descomentar está línea y comentar la siguiente para volver a calcular el vocabulario:
# voc = buildVocabulary()
voc = pickle.load(open("myVocabulary.pkl", "rb"))
Ejercicio3(voc, (54, 96, 71, 100))
