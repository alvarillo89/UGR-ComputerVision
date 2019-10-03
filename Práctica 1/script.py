# -*- coding: utf-8 -*-

"""
##############################################################################
# Visión por Computador
# Trabajo 1: Filtrado y muestreo
# Álvaro Fernández García
##############################################################################
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


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


# Función para dibujar las pirámides:
def _DrawPyr(images, title, color=255):
    oriH = images[0].shape[0]
    out = images[0]
    for i in range(1, len(images)):
        # Hacemos que las imágenes tengan la misma height:
        lay = np.full((oriH - images[i].shape[0], images[i].shape[1]), color, np.uint8)
        tmp = cv2.vconcat((images[i], lay))
        out = cv2.hconcat((out, tmp))
    _showImage(out, title)
    return out


##############################################################################
# EJERCICIO 1
##############################################################################
"""
A)
El cálculo de la convolución de una imagen con una máscara
Gaussiana 2D (Usar GaussianBlur). Mostrar ejemplos con distintos
tamaños de máscara y valores de sigma. Valorar los resultados.
"""
def Gaussiana(img, sigma, tam=None):
    # Si no se da tamaño, se calcula en función de Sigma: 2*int(3*sigma)+1
    if tam == None:
        tam = (2*int(3*sigma)+1, 2*int(3*sigma)+1)
    blur = cv2.GaussianBlur(img, tam, sigma)
    return blur


"""
B)
Usar getDerivKernels para obtener las máscaras 1D que permiten
calcular al convolución 2D con máscaras de derivadas. Representar
e interpretar dichas máscaras 1D para distintos valores de sigma.
"""
# Visualizar los kernels:
def VisualizaKernelDerivado(k, title):
    # Primera derivada:
    kx1, ky1 = cv2.getDerivKernels(dx=1, dy=1, ksize=k)
    # Segunda derivada
    kx2, ky2 = cv2.getDerivKernels(dx=2, dy=2, ksize=k)
    # Derivada en X:
    kxs, ky0 = cv2.getDerivKernels(dx=1, dy=0, ksize=k)
    # Derivada en Y:
    kx0, kys = cv2.getDerivKernels(dx=0, dy=1, ksize=k)

    # Imprimir los kernels:
    print("\n", title)
    np.set_printoptions(suppress=True)
    print("\n1ª derivada:\n", ky1.dot(kx1.transpose()))
    print("\n2ª derivada:\n", ky2.dot(kx2.transpose()))
    print("\nDerivada solo en X\n", ky0.dot(kxs.transpose()))
    print("\nDerivada solo en Y\n", kys.dot(kx0.transpose()))
    np.set_printoptions(suppress=False)


"""
C)
Usar la función Laplacian para el cálculo de la convolución 2D con
una máscara de Laplaciana-de-Gaussiana de tamaño variable.
Mostrar ejemplos de funcionamiento usando dos tipos de bordes y
dos valores de sigma: 1 y 3.
"""
def LaplacianaGaussiana(img, sigma, ksize, border, delta):
    # Añadimos el borde:
    out = cv2.copyMakeBorder(img, 10, 10, 10, 10, border)
    # Suavizamos con la Gaussiana:
    out = cv2.GaussianBlur(out, (ksize, ksize), sigma, borderType=border)
    # Aplicamos la laplaciana:
    out = cv2.Laplacian(src=out, ksize=ksize, borderType=border, ddepth=-1, delta=delta)
    return out


##############################################################################
# EJERCICIO 2:
##############################################################################
"""
A) El cálculo de la convolución 2D con una máscara separable de
tamaño variable. Usar bordes reflejados. Mostrar resultados
"""
def Convolucion2DSeparable(img, kernelx, kernely):
    # Añadir borde reflejado:
    out = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    # Aplicamos la máscara separable:
    out = cv2.sepFilter2D(src=out, kernelX=kernelx, kernelY=kernely, ddepth=-1, borderType=cv2.BORDER_REFLECT)
    return out


"""
B) El cálculo de la convolución 2D con una máscara 2D de 1a
derivada de tamaño variable. Mostrar ejemplos de
funcionamiento usando bordes a cero.
C) El cálculo de la convolución 2D con una máscara 2D de 2a
derivada de tamaño variable.
"""
def Convolucion2DDerivada(img, dx, dy, k):
    # Añadir bordes a 0
    out = cv2.copyMakeBorder(img, 5, 5, 5, 5, 0)
    # Suavizamos para eliminar el ruido: (si sigma < 0 se calcula en función de k)
    out = cv2.GaussianBlur(out, (k,k), -1, borderType=0)
    # Obtenemos los kernels derivados 1D:
    kx, ky = cv2.getDerivKernels(dx=dx, dy=dy, ksize=k, normalize=False)
    # Aplicamos la convolución:
    out = cv2.sepFilter2D(src=out, kernelX=kx, kernelY=ky, ddepth=-1, borderType=0, delta=70)
    return out


"""
D) Una función que genere una representación en pirámide
Gaussiana de 4 niveles de una imagen. Mostrar ejemplos de
funcionamiento usando bordes
"""
# Calcula la pirámide Gaussiana:
def PiramideGaussiana(img, tam, borde):
    # Añadir el borde:
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, borde)

    # Calcular la pirámide
    out = [img]
    for _ in range(tam-1):
        img = cv2.pyrDown(img)
        out.append(img)
    return out


"""
E) Una función que genere una representación en pirámide
Laplaciana de 4 niveles de una imagen. Mostrar ejemplos de
funcionamiento usando bordes.
"""
# Calcula la pirámide Laplaciana:
def PiramideLaplaciana(img, tam, borde):
    out = []
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, borde)
    current = img

    # Calculamos la laplaciana:
    # tam-1 porque el último nivel es el de la gaussiana
    for _ in range(tam-1):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        tmp = cv2.subtract(current, up)
        out.append(tmp)
        current = down

    # Añadir el último nivel de la gaussiana
    out.append(down)
    return out


##############################################################################
# EJERCICIO 3:
##############################################################################
"""
Imágenes Híbridas:
Escribir una función que muestre las tres imágenes ( alta,
baja e híbrida) en una misma ventana. (Recordar que las
imágenes después de una convolución contienen número
flotantes que pueden ser positivos y negativos)
Realizar la composición con al menos 3 de las parejas de
imágenes
"""
def HybridImages(low, high, sigma_low, sigma_high):
    low_frec = Gaussiana(img=low, sigma=sigma_low)
    high_frec = Gaussiana(img=high, sigma=sigma_high)
    high_frec = cv2.subtract(high, high_frec)
    H = cv2.add(low_frec, high_frec) 
    return H, low_frec, high_frec


##############################################################################
# PRUEBA DE LAS FUNCIONES:
##############################################################################

img = cv2.imread("imagenes/fish.bmp", 0)

# Ejercicio 1 A):
# K fijo, sigma variable:
images = []
for sigma in (1,3):
    images.append(Gaussiana(img.copy(), sigma, (19,19)))
PintaMI(images, 'Ejercicio 1A) K fijo, sigma variable')

# sigma fijo, k variable:
images = []
for k in (5, 19):
    images.append(Gaussiana(img.copy(), 3, (k,k)))
PintaMI(images, 'Ejercicio 1A) Sigma fijo, k variable')

# K y sigma variables:
images = []
for sigma in (0.8, 2.5, 5, 9):
    images.append(Gaussiana(img.copy(), sigma))
PintaMI(images, 'Ejercicio 1A) Sigma y K variable')


# Ejercicio 1 B):
# Visualizar los kernels:
VisualizaKernelDerivado(k=3, title="Tamaño del kernel = 3")
input("\nPresiona enter para continuar...")
VisualizaKernelDerivado(k=5, title="Tamaño del kernel = 5")
input("\nPresiona enter para continuar...")


# Ejercicio 1 C):
img = cv2.imread("imagenes/motorcycle.bmp", 0)
images = []
for sigma in (1,3):
    for border in (cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT):
        images.append(LaplacianaGaussiana(img, sigma, 3, border, 70))

_showImage(images[0], 'Ejercicio 1 C) Sigma = 1, bordes replicados')
_showImage(images[1], 'Ejercicio 1 C) Sigma = 1, bordes reflejados')
_showImage(images[2], 'Ejercicio 1 C) Sigma = 3, bordes replicados')
_showImage(images[3], 'Ejercicio 1 C) Sigma = 3, bordes reflejados')


# Ejercicio 2 A):
img = cv2.imread("imagenes/fish.bmp", 0)
images = []
for sigma in (0.8, 2.5, 5, 9):
    kernel = cv2.getGaussianKernel(ksize=2*int(3*sigma)+1, sigma=sigma)
    aux = Convolucion2DSeparable(img, kernelx=kernel, kernely=kernel)
    images.append(aux)

PintaMI(images, "Ejercicio 2 A)")


# Ejercicio 2 B)
img = cv2.imread("imagenes/submarine.bmp", 0)
images = []
for ksize in (3, 5, 7):
    images.append(Convolucion2DDerivada(img=img, dx=1, dy=1, k=ksize))
PintaMI(images, "Ejercicio 2 B) Primera Derivada")


# Ejercicio 2 C)
images = []
for ksize in (3, 5, 7):
    images.append(Convolucion2DDerivada(img=img, dx=2, dy=2, k=ksize))
PintaMI(images, "Ejercicio 2 C) Segunda Derivada")


# Ejercicio 2 D)
img = cv2.imread("imagenes/fish.bmp", 0)
_DrawPyr(PiramideGaussiana(img, 4, cv2.BORDER_REPLICATE), "2 D) Pirámide Gaussiana con bordes replicados")
_DrawPyr(PiramideGaussiana(img, 4, cv2.BORDER_REFLECT), "2 D) Pirámide Gaussiana con bordes reflejados")


# Ejercicio 2 E)
img = cv2.imread("imagenes/plane.bmp", 0)
_DrawPyr(PiramideLaplaciana(img, 4, cv2.BORDER_REPLICATE), "2 D) Pirámide Laplaciana con bordes replicados")
_DrawPyr(PiramideLaplaciana(img, 4, cv2.BORDER_REFLECT), "2 D) Pirámide Laplaciana con bordes reflejados")


# Ejercicio 3)
dog = cv2.imread("imagenes/dog.bmp", 0)
cat = cv2.imread("imagenes/cat.bmp", 0)
H, L, U = HybridImages(cat, dog, sigma_low=10, sigma_high=5)
PintaMI([H, L ,U], 'Ejercicio 3) Perro + Gato')

motor = cv2.imread("imagenes/motorcycle.bmp", 0)
bike = cv2.imread("imagenes/bicycle.bmp", 0)
H, L, U = HybridImages(motor, bike, sigma_low=6, sigma_high=2)
PintaMI([H, L ,U], 'Ejercicio 3) Moto + Bicicleta')

mary = cv2.imread("imagenes/marilyn.bmp", 0)
einstein = cv2.imread("imagenes/einstein.bmp", 0)
H, L, U = HybridImages(mary, einstein, sigma_low=6, sigma_high=3)
PintaMI([H, L ,U], 'Ejercicio 3) Marilyn + Einstein')

bird = cv2.imread("imagenes/bird.bmp", 0)
plane = cv2.imread("imagenes/plane.bmp", 0)
H, L, U = HybridImages(bird, plane, sigma_low=6, sigma_high=2)
PintaMI([H, L ,U], 'Ejercicio 3) Pájaro + Avión')


##############################################################################
# BONUS:
##############################################################################

"""
EJERCICIO 1:
Cálculo del vector máscara Gaussiano
"""
# Definimos la función:
def f(x, sigma):
    return math.exp(-0.5 * ((x**2)/(sigma**2)))

# Función que devuelve la máscara Gaussiana:
def myGaussianKernel(sigma):
    # Calculamos el tamaño:
    k = int(3 * sigma)
    # Creamos la máscara:
    mask = [f(x,sigma) for x in range(-k, k+1)]
    mask = np.array(mask)
    # Para que los valores sumen 1:
    mask = mask / np.sum(mask)
    return mask 


"""
EJERCICIO 2:
Convolución 1D con un vector señal
"""
# Función para añadir los bordes reflejados:
def myAddBorders(img, size):
    return np.concatenate((img[:size][::-1], img, img[len(img)-size:][::-1]))


# Función para eliminar los bordes:
def myRemoveBorders(img, size):
    return img[size:len(img)-size]


# Función auxiliar que calcula la convolución 1D de un solo canal:
def convolve1Dvector(signal, kernel):
    # Verificamos que se cumplan las condiciones:
    assert kernel.shape[0] < signal.shape[0]
    assert len(kernel) % 2 != 0

    # Añadimos los bordes:
    borderLen = kernel.shape[0] // 2
    signal = myAddBorders(signal, borderLen)

    # Creamos la salida:
    salida = np.zeros(signal.shape, signal.dtype)

    # Ahora aplicamos la convolucion:
    for i in range( signal.shape[0] - kernel.shape[0] + 1 ):
        newValue = np.sum( signal[i:i+kernel.shape[0]] * kernel )
        salida[i+borderLen] = newValue

    # Eliminamos los bordes:
    signal = myRemoveBorders(signal, borderLen)
    salida = myRemoveBorders(salida, borderLen)

    return salida


# Se llamará a esta función que si tiene en cuenta los demás canales:
def myConvolution1D(signal, kernel):
    if len(signal.shape) == 3:
        channels = cv2.split(signal)
        channels[0] = convolve1Dvector(channels[0].flatten(), kernel)
        channels[1] = convolve1Dvector(channels[1].flatten(), kernel)
        channels[2] = convolve1Dvector(channels[2].flatten(), kernel)
        out = cv2.merge(channels)
    else:
        out = convolve1Dvector(signal, kernel)

    return out


"""
EJERCICIO 3:
Convolución 2D de una imagen con máscara separable:
"""
def mySeparable2DConvolution(src, kernelX, kernelY):
    # Separamos los canales:
    channels = cv2.split(src)
    convolved_channels = []

    # Para cada canal aplicamos la convolución
    for ch in channels:
        # Creamos una imagen temporal para la convolución:
        tmp1 = np.zeros(ch.shape, ch.dtype)
        
        # Aplicar kernelX por filas:
        for i in range(tmp1.shape[0]):
            tmp1[i] = convolve1Dvector(ch[i], kernelX)

        # Transponemos para hacer la convolución por columas
        tmp1 = tmp1.transpose()

        # Creamos la segunda imagen temporal para la convolución:
        tmp2 = np.zeros(tmp1.shape, ch.dtype)

        # Aplicar kernelY por columnas:    
        for j in range(tmp2.shape[0]):
            tmp2[j] = convolve1Dvector(tmp1[j], kernelY)

        # Deshacemos el cambio
        tmp2 = tmp2.transpose()

        convolved_channels.append(tmp2)

    salida = cv2.merge(convolved_channels)

    return salida


"""
Ejercicio 4: 
Pirámide Gaussiana manual
"""
def myGaussianPyr(img, levels):
    mask = myGaussianKernel(3)
    
    # Calcular la pirámide
    out = [img]
    for _ in range(levels-1):
        # Estas dos líneas son equivalentes a cv2.pyrDown()
        img = mySeparable2DConvolution(img, mask, mask)
        # Nos quedamos con las filas y columnas pares
        img = img[::2, ::2]
        out.append(img)

    return out


"""
Ejercicio 5:
Imágenes Híbridas a color:
"""
def myHybridImagesColor(low, high, sigma_low, sigma_high):
    # Declaramos las variables que necesitaremos:
    mask_low = myGaussianKernel(sigma_low)
    mask_high = myGaussianKernel(sigma_high)
    Hyb = []
    Low = [] 
    Hig = []

    # Separamos los canales:
    low_channels = cv2.split(low)
    high_channels = cv2.split(high)

    # Para cada canal repetimos el siguiente proceso:
    for lch, hch in zip(low_channels, high_channels):
        # Calculamos la img híbrida en ese canal
        # Suavizamos
        low_frec = mySeparable2DConvolution(lch, mask_low, mask_low)
        high_frec = mySeparable2DConvolution(hch, mask_high, mask_high)

        # Hacemos los cálculos en coma flotante para evitar problemas:
        high_frec = hch.astype(np.float64) - high_frec.astype(np.float64)
        hibrida = low_frec.astype(np.float64) + high_frec

        # Volvemos a convertir a uint8
        hibrida = np.clip(hibrida, 0, 255).astype(np.uint8)
        high_frec = np.clip(high_frec, 0, 255).astype(np.uint8)

        # Guardamos la información calculada para cada imagen
        Hyb.append(hibrida)
        Low.append(low_frec)
        Hig.append(high_frec)

    # Juntamos todos los canales:
    Hyb = cv2.merge(Hyb)
    Low = cv2.merge(Low)
    Hig = cv2.merge(Hig)

    return Hyb, Low, Hig


##############################################################################
# Pruebas del bonus:
##############################################################################

print("\nBONUS:")

# Ejercicio 1:
kernel = myGaussianKernel(sigma=3)
print("Ejercicio 1, kernel gaussiano de sigam =3")
print(kernel)
input("\nPulsa enter para continuar...")


# Ejericicio 2:
signal = np.ones(5)
mask = np.array([-1, 0, 1])
result = myConvolution1D(signal=signal, kernel=mask)
print("\nEjercicio 2: Convolución de la señal {} de un canal con el kernel {}".format(signal, mask))
print(result)
input("\nPulsa enter para continuar...")

signal = np.ones((5,1,3))
mask = np.array([-1, 0, 1])
result = myConvolution1D(signal=signal, kernel=mask)
print("\nEjercicio 2: Convolución de la señal {} de tres canales con el kernel {}".format(signal, mask))
print(result)
input("\nPulsa enter para continuar...")


# Ejercicio 3:
img_gray = cv2.imread("imagenes/submarine.bmp", 0)
img_color = cv2.imread("imagenes/submarine.bmp", 1)
# Usaremos el kernel del primer ejercicio para la convolución:
im_gray_conv = mySeparable2DConvolution(img_gray, kernel, kernel)
im_color_conv = mySeparable2DConvolution(img_color, kernel, kernel)
PintaMI([img_gray, im_gray_conv], "Bonus 3) Convolución manual en escala de grises")
PintaMI([img_color, im_color_conv], "Bonus 3) Convolución manual a color")


# Ejercicio 4:
# Probaremos la pirámide Gaussiana con la última imagen híbrida calculada:
_DrawPyr(myGaussianPyr(H, 5), "Bonus 4)")


# Ejercicio 5:
dog = cv2.imread("imagenes/dog.bmp", 1)
cat = cv2.imread("imagenes/cat.bmp", 1)
H, L, U = myHybridImagesColor(cat, dog, sigma_low=10, sigma_high=5)
PintaMI([H, L ,U], 'Bonus 5) Perro + Gato')

motor = cv2.imread("imagenes/motorcycle.bmp", 1)
bike = cv2.imread("imagenes/bicycle.bmp", 1)
H, L, U = myHybridImagesColor(motor, bike, sigma_low=6, sigma_high=2)
PintaMI([H, L ,U], 'Bonus 5) Moto + Bicicleta')

mary = cv2.imread("imagenes/marilyn.bmp", 1)
einstein = cv2.imread("imagenes/einstein.bmp", 1)
H, L, U = myHybridImagesColor(mary, einstein, sigma_low=6, sigma_high=3)
PintaMI([H, L ,U], 'Bonus 5) Marilyn + Einstein')

bird = cv2.imread("imagenes/bird.bmp", 1)
plane = cv2.imread("imagenes/plane.bmp", 1)
H, L, U = myHybridImagesColor(bird, plane, sigma_low=6, sigma_high=2)
PintaMI([H, L ,U], 'Bonus 5) Pájaro + Avión')

sub = cv2.imread("imagenes/submarine.bmp", 1)
fish = cv2.imread("imagenes/fish.bmp", 1)
H, L, U = myHybridImagesColor(sub, fish, sigma_low=6, sigma_high=2)
PintaMI([H, L ,U], 'Bonus 5) Submarino + Pez')