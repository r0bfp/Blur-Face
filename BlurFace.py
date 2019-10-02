import cv2
import sys


#argumentos de entrada para imagem e classificadores
imagePath = sys.argv[1]
cascPath = sys.argv[2]

#criacao do classificador
faceCascades = cv2.CascadeClassifier(cascPath)

#leitura da imagem
image = cv2.imread(imagePath)
#converte imagem em cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#deteca a face
faces = faceCascades.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

#desenha um retangulo na face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#mostra a imagem
cv2.imshow("Faces found", image)
#espera uma tecla para fechar o programa
cv2.waitKey(0)