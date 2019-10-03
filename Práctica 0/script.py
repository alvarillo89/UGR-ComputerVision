# -*- coding: utf-8 -*-

"""
##############################################################################
# Visión por Computador
# Trabajo 0: Introducción a OpenCV
# Álvaro Fernández García
##############################################################################
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Asignar a esta variable la ruta de la imagen con la que se probarán las
# distintas funciones:
IMG = ""

##############################################################################
# EJERCICIO 2
##############################################################################

# Muestra una imgen en pantalla. La imagen se recibe como una matriz:
def _showImage(img, title='Imagen'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


##############################################################################
# EJERCICIO 1
##############################################################################

# Carga una imagen a color o en escala de grises:
# flagColor == 1 (Color), flagColor == 0 (Grises)
def LeeImagen(filename, flagColor):
    img = cv2.imread(filename, flagColor)
    _showImage(img)
    return img


##############################################################################
# EJERCICIO 3
##############################################################################

# Recibe un array de imágenes y las concatena en una sola, si el array solo
# tiene una imagen, la dibuja.
def PintaMI(vim):
    assert len(vim) > 0
    if len(vim) == 1:
        finalImg = vim
    else:
        finalImg = vim[0]
        vim.pop(0)
        for img in vim:
            finalImg = cv2.hconcat((finalImg, img))
    _showImage(finalImg)


"""
¿Qué pasa si las imágenes no son todas del mismo tipo: (nivel de gris, color,
blanco-negro)?

Los píxeles de las imágenes en escala de grises están formados por un sólo
número entero que determina la intensidad, mientras que por el contrario, en 
las imágenes a color, cada píxel está formado por tres números: azul, verde y 
rojo (BGR). Si intentamos concatenar dos imágenes de distinto tipo, nos 
encontramos con que la tercera dimensión de la matriz imagen no coincide 
(1 para Gris, 3 para BGR). Además tampoco podrían concatenarse dos imágenes 
con distinta altura.
"""


#############################################################################
# EJERCICIO 4
##############################################################################

# Modifica el color de los píxeles indicados en pxList por el de newColor
def ModificaColor(img, pxList, newColor):
    for px in pxList:
        img[px[0], px[1]] = newColor
        
    _showImage(img)


##############################################################################
# EJERCICIO 5
##############################################################################

# Dibuja las imágenes junto con sus titulos (Usando MatplotLib):
def DrawImgAndTitlePLT(images, titles):
    i = 1
    for im, tt in zip(images, titles):
        subpltIndex = int("1" + str(len(images)) + str(i))
        plt.subplot(subpltIndex)
        # Si es a color, pasamos de BGR a RGB:
        if len(im.shape) == 3:  
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im, 'gray')
        plt.title(tt)
        plt.xticks([]), plt.yticks([]) 
        i += 1
    plt.show()
    
    
# Dibuja las imágenes junto con sus titulos (Usando OpenCV):
# Desvetaja: No pueden mostrarse imagenes grises y a color simultáneamente
# y todas las imágenes deben tener la misma altura:
def DrawImgAndTitleCV(images, titles):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    LINE = cv2.LINE_AA
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    assert len(images) > 0
    vim = []
    for i,t in zip(images, titles):
           i = cv2.copyMakeBorder(i, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=BLACK)
           topLay = np.zeros((50, i.shape[1], 3), np.uint8)
           topLay[:] = WHITE
           i = cv2.vconcat((topLay, i))
           cv2.putText(i, t, (30,30), FONT, 1, BLACK, 2, LINE)
           vim.append(i)
    PintaMI(vim)


##############################################################################
# PRUEBA DE LAS FUNCIONES:
##############################################################################

# Ejercicio 1 y 2:
img = LeeImagen(IMG, 1)     # A color
imgGray = LeeImagen(IMG, 0)     # Gris

# Ejercicio 3:
PintaMI([img, img])

# Ejercicio 4:
# Crear la lista de píxeles:
listaPixeles = []
for i in range(0, int(img.shape[0]/2)):
    for j in range(0, int(img.shape[1]/2)):
        listaPixeles.append([i,j])

imgCopy = img.copy()    # Guardar la imagen original
ModificaColor(imgCopy, listaPixeles, (0,0,255))

# Ejercicio 5:
DrawImgAndTitlePLT([img, imgGray], ['A Color', 'Grises'])
DrawImgAndTitleCV([img, img], ['Imagen 1', 'Imagen 2'])