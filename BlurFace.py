# -*- coding: utf-8 -*-

import cv2
import sys
import numpy


#argumento para entrada de arquivos de video
video = sys.argv[1]
#argumento para entrada de classificadores
cascPath = sys.argv[2]

#carrega o arquivo de classificacao treinado
faceCascades = cv2.CascadeClassifier(cascPath)

#capturando o video
cap = cv2.VideoCapture(video)

while(1):
    #lendo a imagem
    ret, frame = cap.read()
    #passando a imagem para cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #localizando faces
    faces = faceCascades.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #desenhando circulos na face encontrada
    for (x,y,w,h) in faces:
        #encontrando ponto central do retangulo
        center = (x+w//2, y+h//2)
        #encontrando o raio do circulo
        radius = (w+h)//4

        #desenhando circulo 
        cv2.circle(frame, center, radius, (255, 0, 255), -1)

        #exibindo o frame processado
        cv2.imshow('frame', frame)

    #aguardando tecla para finalizar o programa
    if cv2.waitKey(1) == ord('q'):
        break

#destroi todas janelas abertas 
cv2.destroyAllWindows()
