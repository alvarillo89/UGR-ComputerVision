# -*- coding: utf-8 -*-

"""
##############################################################################
# Visión por Computador
# Trabajo 2: Detección de puntos relevantes y construcción de panoramas
# Álvaro Fernández García
##############################################################################
"""

import numpy as np
import cv2
import random
import math

# Establecer las semillas:
np.random.seed(5)
random.seed(5)


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


# Función para obtener la octava y capa de un punto SIFT:
def unpackSIFTp(kp):
    octava = (kp.octave & 255)
    
    if octava >= 128:
        octava |= -128

    capa = (kp.octave >> 8) & 255

    return octava, capa


##############################################################################
# EJERCICIO 1:
##############################################################################

"""
Detección de puntos SIFT y SURF. Aplicar la detección de
puntos SIFT y SURF sobre las imágenes, representar dichos puntos sobre
las imágenes haciendo uso de la función drawKeyPoints. Presentar los
resultados con las imágenes Yosemite.rar.
    (a) Variar los valores de umbral de la función de detección de puntos
    hasta obtener un conjunto numeroso (≥ 1000) de puntos SIFT y
    SURF que sea representativo de la imagen. Justificar la elección
    de los parámetros en relación a la representatividad de los puntos
    obtenidos.
    (b) Identificar cuántos puntos se han detectado dentro de cada octava.
    En el caso de SIFT, identificar también los puntos detectados en
    cada capa. Mostrar el resultado dibujando sobre la imagen original
    un cı́rculo centrado en cada punto y de radio proporcional al valor de
    sigma usado para su detección (ver circle()) y pintar cada octava
    en un color.
    (c) Mostrar cómo con el vector de keyPoint extraı́dos se pueden calcu-
    lar los descriptores SIFT y SURF asociados a cada punto usando
    OpenCV.
"""

def Ejercicio1(src, hessThrld, edge, contrast):
    # Apartado A):
    # Primero usamos Surf
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessThrld)
    kp_surf = surf.detect(src, None)
    print("Surf point: ", len(kp_surf))
    puntos_surf = cv2.drawKeypoints(src, kp_surf, None, (0,0,255))

    # A continuación usamos sift:
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast, edgeThreshold=edge)
    kp_sift = sift.detect(src, None)
    print("Sift point: ", len(kp_sift))
    puntos_sift = cv2.drawKeypoints(src, kp_sift, None, (255,0,0))

    # Mostramos los resultados:
    PintaMI([puntos_surf, puntos_sift], "Ejercicio 1.a) SUFR y SIFT respectivamente")

    # Apartado B):
    # Extraer los puntos de surf en función de las octavas:
    print("\nPuntos SURF por octavas:")
    surf_octavas = src.copy()
    colors = [(255,0,0), (0,0,255), (0,255,0), (0,0,0)]

    for o, c in zip(range(surf.getNOctaves()), colors):
        # Nos quedamos con los puntos tal que octava == o:
        tmp = list( filter(lambda kp: kp.octave == o, kp_surf) )
        # Generamos un color aleatorio y añadimos los puntos a la imagen:
        surf_octavas = cv2.drawKeypoints(surf_octavas, tmp, None, c, 4)
        print("Octava", str(o), ":", str(len(tmp)), "puntos")

    _showImage(surf_octavas, 'Ejercicio 1.b) Puntos Surf según las octavas')

    # Extraer los puntos SIFT en función de las capas y octavas:  
    print("\nPuntos SIFT por capas y octavas")
    # Calculamos primero las octavas de la imagen:
    octavas_sift = []
    for kp in kp_sift:
        octavaLocal = unpackSIFTp(kp)[0]
        if not octavaLocal in octavas_sift:
            octavas_sift.append(octavaLocal)

    # Ahora filtramos por capas y octavas:
    colors = [(255,0,0), (0,0,255), (0,255,0)]
    for o in octavas_sift:
        sift_octavas = src.copy()
        for c, color in zip(range(1,4), colors):
            tmp = list( filter(lambda kp: unpackSIFTp(kp)[0] == o and unpackSIFTp(kp)[1] == c, kp_sift))           
            sift_octavas = cv2.drawKeypoints(sift_octavas, tmp, None, color, 4)
            print("Octava", str(o), "capa", str(c), ":", str(len(tmp)), "puntos")

        _showImage(sift_octavas, "Ejericicio 1.b) Puntos SIFT, octava " + str(o))


    # Apartado C)
    # Extraemos los descriptores con compute:
    surf_descriptors = surf.compute(src, kp_surf)[1]
    sift_descriptors = sift.compute(src, kp_sift)[1]

    # Vamos a comprobar que los resultados son los mismos que utilizando detectAndCompute()
    surf_check = surf.detectAndCompute(src, None)[1]
    sift_check = sift.detectAndCompute(src, None)[1]

    if np.all(surf_descriptors == surf_check) and np.all(sift_descriptors == sift_check):
        print("\nLos descriptores son iguales")
    else:
        print("\nLos descriptores NO son iguales")
    

##############################################################################
# EJERCICIO 2:
##############################################################################
"""
Usar el detector-descriptor SIFT de OpenCV sobre las imágenes
de Yosemite.rar (cv2.xfeatures2d.SIFT create()). Extraer sus lis-
tas de keyPoints y descriptores asociados. Establecer las corresponden-
cias existentes entre ellos usando el objeto BFMatcher de OpenCV y los
criterios de correspondencias “BruteForce+crossCheck y “Lowe-Average-
2NN”. (NOTA: Si se usan los resultados propios del puntos anterior en
lugar del cálculo de SIFT de OpenCV se añaden 0.5 puntos)
    (a) Mostrar ambas imágenes en un mismo canvas y pintar lı́neas de difer-
    entes colores entre las coordenadas de los puntos en correspondencias.
    Mostrar en cada caso 100 elegidas aleatoriamente.
    2(b) Valorar la calidad de los resultados obtenidos en términos de las corre-
    spondencias válidas observadas por inspección ocular y las tendencias
    de las lı́neas dibujadas.
    (c) Comparar ambas técnicas de correspondencias en términos de la cal-
    idad de sus correspondencias (suponer 100 aleatorias e inspección
    visual).
"""
# Función que obtiene n correspondencias entre las dos imágenes:
# si nbest = True, devuelve las n mejores correspondencias:
def match_fuerza_bruta(img1, img2, n, nbest):
    # Creamos el objeto SIFT:
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtenemos los keyPoints y los descriptores de cada imagen:
    kp_img1 = sift.detect(img1, None)
    kp_img1, desc_img1 = sift.compute(img1, kp_img1)

    kp_img2 = sift.detect(img2, None)
    kp_img2, desc_img2 = sift.compute(img2, kp_img2)

    # Creamos el objeto BFmatcher: (BruteFroce + CrossCheck)
    bf = cv2.BFMatcher(crossCheck=True)

    # Extraemos los matches:
    matches = bf.match(desc_img1, desc_img2)
    
    # Obtener los n:
    if nbest:
        matches = sorted(matches, key = lambda x:x.distance)
        return matches[:n], kp_img1, kp_img2
    else:
        return random.sample(matches, n), kp_img1, kp_img2    


# Igual que la anterior pero para el criterio Lowe-Average-2NN:
def matches_2nn(img1, img2, n, nbest):
    # Creamos el objeto SIFT:
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtenemos los keyPoints y los descriptores de cada imagen:
    kp_img1 = sift.detect(img1, None)
    kp_img1, desc_img1 = sift.compute(img1, kp_img1)

    kp_img2 = sift.detect(img2, None)
    kp_img2, desc_img2 = sift.compute(img2, kp_img2)

    # Utilizamos KNN:
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(desc_img1, desc_img2, k=2)
    
    # Aplicamos el ratio de D. Lowe: 
    matches = [[m] for m,z in matches if m.distance < 0.7 * z.distance]

    # Obtener n puntos
    if nbest:
        matches = sorted(matches, key = lambda x:x[0].distance)
        return matches[:n], kp_img1, kp_img2 
    else:
        return random.sample(matches, n), kp_img1, kp_img2


# Función que implementa las dos anteriores:
def Ejercicio2(img1, img2):
    # Comparación con 100 puntos aleatorios:
    resultados_fuerza, kp1f, kp2f = match_fuerza_bruta(img1, img2, 100, False)
    resultados_2nn, kp1n, kp2n = matches_2nn(img1, img2, 100, False)
    out1 = cv2.drawMatches(img1, kp1f, img2, kp2f, resultados_fuerza, None, flags=2)
    out2 = cv2.drawMatchesKnn(img1, kp1n, img2, kp2n, resultados_2nn, None, flags=2)
    _showImage(out1, "100 aleatorios, fuerza Bruta")
    _showImage(out2, "100 aleatorios, 2nn")


    # Comparación con los 100 mejores puntos:
    resultados_fuerza, kp1f, kp2f = match_fuerza_bruta(img1, img2, 100, True)
    resultados_2nn, kp1n, kp2n = matches_2nn(img1, img2, 100, True)
    out1 = cv2.drawMatches(img1, kp1f, img2, kp2f, resultados_fuerza, None, flags=2)
    out2 = cv2.drawMatchesKnn(img1, kp1n, img2, kp2n, resultados_2nn, None, flags=2)
    _showImage(out1, "100 mejores, fuerza Bruta")
    _showImage(out2, "100 mejores, 2nn")



##############################################################################
# EJERCICIOS 3, 4 y 3 del BONUS:
##############################################################################
"""
BONUS 3.- Implementar de forma eficiente la estimación de una homografı́a
usando RANSAC.
"""
# Para calcular la homografía, aplicaremos  el
# algoritmo "Direct Linear Transformation":
def DLT(pt1, pt2):
    # Creamos la matriz A:
    A = []
    for p1, p2 in zip(pt1, pt2):
        x, y = p1[0], p1[1]
        u, v = p2[0], p2[1]
        eqn1 = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        eqn2 = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
        A.append(eqn1)
        A.append(eqn2)

    # Realizamos la descomposición en valores singulares:
    U, D, Vt = np.linalg.svd(np.array(A))

    # H es el último vector propio:
    H = Vt[-1].reshape(3,3)

    return H


# Implementación de RANSAC:
def RANSAC_Homography(pt1, pt2, theshold):
    iters = 0
    finalH = []
    bestNumberOfInliers = 0

    # Calcula la distancia euclidea entre el punto p1 tras aplicarle H y p2:
    def distance(p1, p2, myH):
        # Añadimos el 1 final y lo colocamos como vector columna:
        p1 = np.append(p1, 1).reshape(3,1)
        p2 = np.append(p2, 1).reshape(3,1)

        # Calculamos H * p1 y hacemos que el ultimo valor sea 1:
        predicted = myH.dot(p1)
        predicted = (1 / predicted[2]) * predicted

        return np.linalg.norm(p2 - predicted)

    # Empezamos la iteraciones:
    while iters < 500:
        # Sacar 4 correspondencias aleatorias:
        randIndex = np.random.choice(len(pt1), 4, replace=False)
        subsample1 = pt1[randIndex]
        subsample2 = pt2[randIndex]

        # Estimar la homografía:
        H = DLT(subsample1, subsample2)

        # Contar la cantidad de inliers para esa H:
        inliers = 0
        for p1, p2 in zip(pt1, pt2):
            if distance(p1, p2, H) < theshold:
                inliers += 1

        # Comprobamos si ha sido la mejor H:
        if inliers > bestNumberOfInliers:
            bestNumberOfInliers = inliers
            finalH = H

        # Incremetar las iteraciones:
        iters += 1

    return finalH


"""
3.- Escribir una función que genere un mosaico de calidad a
partir de N = 3 imágenes relacionadas por homografı́as, sus listas de
keyPoints calculados de acuerdo al punto anterior y las correspondencias
encontradas entre dichas listas. Estimar las homografı́as entre ellas usando
la función cv2.findHomography(p1,p2, CV RANSAC,1). Para el mosaico
será necesario.
    (a) Definir una imagen en la que pintaremos el mosaico.
    (b) Definir la homografı́a que lleva cada una de las imágenes a la imagen
    del mosaico.
    (c) Usar la función cv2.warpPerspective() para trasladar cada imagen
    al mosaico (Ayuda: Usar el flag BORDER TRANSPARENT de warpPers-
    pective).
4.- Lo mismo que en el punto anterior pero para N > 5 (usar las
imágenes para mosaico).
"""

# Contruye un mosaico dado un array de imágenes. Las imágenes deben de
# estar ordenadas y ser a color:
# opencv indica si usar findHomography() o la función del bonus:
def construye_mosaico(imgs, title, opencv=True):
    # Crear la imagen que contendrá el mosaico:
    alto = imgs[0].shape[0] + 400
    ancho = int(np.array([img.shape[1] for img in imgs]).sum()) + 400
    mosaico = np.zeros((alto, ancho, 3), np.uint8)

    # Calculamos las correspondencias utilizando 2nn:
    matches = [tuple(matches_2nn(imgs[i], imgs[i+1], 100, True)) for i in range(len(imgs) - 1)]

    # Preparamos los puntos para findHomography():
    points = []
    for match in matches:
        src = np.float32([match[1][m[0].queryIdx].pt for m in match[0]])
        dts = np.float32([match[2][m[0].trainIdx].pt for m in match[0]])
        points.append((src, dts))

    # Hallamos las homografías:
    if opencv:
        homographies = [cv2.findHomography(pts[0], pts[1], cv2.RANSAC, 1.0)[0] for pts in points]
    else:
        homographies = [RANSAC_Homography(pts[0], pts[1], 1.0) for pts in points]

    # Invertimos las matrices:
    homographies = [np.linalg.inv(h) for h in homographies]

    # Construimos el mosaico: la primera imagen la colocamos donde queramos:
    # t_x = 200, t_y = 200
    M = np.array([[1,0,200], [0,1,200], [0,0,1]], np.float32)
    cv2.warpPerspective(src=imgs[0], dst=mosaico, M=M, dsize=(mosaico.shape[1], mosaico.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
    del imgs[0]

    for img in imgs:
        # Acumulamos:
        M = M.dot(homographies[0])
        # Aplicamos:
        cv2.warpPerspective(src=img, dst=mosaico, M=M, dsize=(mosaico.shape[1], mosaico.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
        del homographies[0]

    # Borramos la parte sobrante del mosaico:
    rows = [i for i in range(mosaico.shape[0]) if np.all(mosaico[i]==0)]
    cols = [j for j in range(mosaico.shape[1]) if np.all(mosaico[:,j]==0)]
    mosaico = np.delete(mosaico, rows, 0)
    mosaico = np.delete(mosaico, cols, 1)

    # Mostramos la imagen:
    _showImage(mosaico, title)
 

##############################################################################
# Prueba de las funciones:
##############################################################################

# Ejercicio 1) A, B:
yosemite = cv2.imread('imagenes/Yosemite1.jpg', 1)
Ejercicio1(yosemite, 700, 2.7, 0.04)

input("\nPulsa ENTER para continuar...")

# Ejercicio 2)
yosemite1 = cv2.imread('imagenes/Yosemite1.jpg', 1)
yosemite2 = cv2.imread('imagenes/Yosemite2.jpg', 1)
Ejercicio2(yosemite1, yosemite2)

# Ejercicio 3)
yosemite1 = cv2.imread('imagenes/yosemite1.jpg', 1)
yosemite2 = cv2.imread('imagenes/yosemite2.jpg', 1)
yosemite3 = cv2.imread('imagenes/yosemite3.jpg', 1)
imgs = [yosemite1, yosemite2, yosemite3]
construye_mosaico(imgs, "Mosaico de tres imágenes")

yosemite5 = cv2.imread('imagenes/yosemite5.jpg', 1)
yosemite6 = cv2.imread('imagenes/yosemite6.jpg', 1)
yosemite7 = cv2.imread('imagenes/yosemite7.jpg', 1)
imgs = [yosemite5, yosemite6, yosemite7]
construye_mosaico(imgs, "Mosaico de tres imágenes")

# Ejercicio 4)
mosaico1 = cv2.imread('imagenes/mosaico002.jpg', 1)
mosaico2 = cv2.imread('imagenes/mosaico003.jpg', 1)
mosaico3 = cv2.imread('imagenes/mosaico004.jpg', 1)
mosaico4 = cv2.imread('imagenes/mosaico005.jpg', 1)
mosaico5 = cv2.imread('imagenes/mosaico006.jpg', 1)
mosaico6 = cv2.imread('imagenes/mosaico007.jpg', 1)
mosaico7 = cv2.imread('imagenes/mosaico008.jpg', 1)
mosaico8 = cv2.imread('imagenes/mosaico009.jpg', 1)
mosaico9 = cv2.imread('imagenes/mosaico010.jpg', 1)
mosaico10 = cv2.imread('imagenes/mosaico011.jpg', 1)
imgs = [mosaico1, mosaico2, mosaico3, mosaico4, mosaico5, mosaico6, mosaico7, mosaico8, mosaico9, mosaico10]
construye_mosaico(imgs, 'Mosaico de 10 imágenes')

# Bonus 3)
yosemite5 = cv2.imread('imagenes/yosemite5.jpg', 1)
yosemite6 = cv2.imread('imagenes/yosemite6.jpg', 1)
yosemite7 = cv2.imread('imagenes/yosemite7.jpg', 1)
imgs = [yosemite5, yosemite6, yosemite7]
construye_mosaico(imgs, "Mosaico de tres imágenes usando RANSAC propio", False)