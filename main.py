import numpy as np
import cv2
import matplotlib.pyplot as plt
imagenes = {}
for i in range(6, 56):
    if i <10:
        p = f"0{i}"
    else:
        p = i
    imagen = cv2.imread(f"images/ISIC_00243{p}.jpg")
    segmentation = cv2.imread(f"images/ISIC_00243{p}_segmentation.png")
    imagenes[f"imagen {i}"] = [imagen, segmentation]
def filtro_gauss():
    

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imagenes["imagen 19"][0])
plt.title("Imagen Original")
plt.subplot(1, 2, 2)
plt.imshow(imagenes["imagen 19"][1])
plt.title("Imagen Segmentada ideal")
plt.show()