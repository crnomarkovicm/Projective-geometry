import cv2
import numpy as np
from numpy import linalg as la
from PIL import Image
koord = []
def afina(tacka):
    return [tacka[0]/tacka[2], tacka[1]/tacka[2]]

#Funkcija koja nam vraca homogene koordinate tacke zadate njenim afinim koordinatama
def homogena(tacka):
    return [tacka[0], tacka[1], 1]

def primeni(tacke, matrica):
    t = []
    n = len(tacke)
    for i in range(n):
        t.append(np.matmul(matrica, tacke[i]))
    return t

#Funkcija koja vraca matricu normalizacije tacaka i normalizovane tacke
def normalizacija(n,tacke):
    afine_tacke = []

    #Odredjujemo teziste tacaka
    cx = 0
    cy = 0
    
    for i in range(n):
        afine_tacke.append(afina(tacke[i]))
        cx = cx + afine_tacke[i][0]
        cy = cy + afine_tacke[i][1]
    cx = cx/n
    cy = cy/n

    #Pravimo matricu koja translira teziste u koordinatni pocetak
    G = [[1,0,-cx], [0,1,-cy], [0,0,1]]

    #Svaku tacku transliramo za vektor CO
    for i in range(n):
        afine_tacke[i] = np.matmul(G, homogena(afine_tacke[i]))
   
    #Racunamo prosecno rastojanje tacaka od koordinatnog pocetka
    dist = 0
    for i in range(n):
        dist = dist + np.sqrt(np.square(afine_tacke[i][0]) + np.square(afine_tacke[i][1]))

    dist = dist/n

    #Pravimo matricu homotetije kako bismo skalirali tacke da prosecna udaljenost bude koren iz dva
    S = [[np.sqrt(2)/dist,0, 0], [0,np.sqrt(2)/dist,0], [0,0,1]]

    #Skaliramo tacke
    for i in range(n):
        afine_tacke[i] = np.matmul(S, homogena(afine_tacke[i]))

    #Matricu normalizacije dobijamo kao proizvod matrice homotetije i matrice translacije
    T = np.matmul(S,G)

    return T

def dlt_algoritam(n, originali, slike):
    matrica = []

    #Za sve korespodencije originala i slike odredjujemo 2x9 matricu a potom ih spajamo u jednu 2n*9
    for i in range(n):
        if i > 0:
            m = matrica_korespodencije(originali[i], slike[i])
            matrica = np.concatenate((matrica, m), axis=0)
        else:
            matrica = matrica_korespodencije(originali[i], slike[i])
    
    #Radimo SVD dekompoziciju matrice 
    U, D, Vt = la.svd(matrica, full_matrices=True)
    
    #Matrica P ce biti poslednja kolona matrice V sto je ustvari poslednja vrsta matrica Vt
    P = Vt[-1]
    P = P.reshape(3,3)

    return P

#Funkcija koja racuna matricu za jednu korespodenciju
def matrica_korespodencije(o, s):
    m = np.matrix([[0, 0, 0, -s[2]*o[0], -s[2]*o[1], -s[2]*o[2], s[1]*o[0], s[1]*o[1], s[1]*o[2]],
     [s[2]*o[0], s[2]*o[1], s[2]*o[2], 0, 0, 0, -s[0]*o[0], -s[0]*o[1], -s[0]*o[2]]])

    return m
#Normalizovani DLT algoritam
def dltnorm_algoritam(n,originali, slike):

    #Vrsimo normalizaciju originalnih tacaka
    originali_t = normalizacija(n,originali)

    #Vrsimo normalizaciju slika tacaka
    slike_t = normalizacija(n,slike)

    originali_n = primeni(originali, originali_t)
  
    slike_n = primeni(slike, slike_t)

    #print("Matrica za normalizaciju originala")
    #print(originali_t)

    #print("Matrica za normalizaciju slika")
    #print(slike_t)

    #Dobijamo matricu P' tako sto primenjujemo obican DLT algoritam na normalizovane tacke
    matrica_p = dlt_algoritam(n, originali_n, slike_n)

    #Matricu projektivnog preslikavanja P dobijamo kao proizvod inverza matrice normalizacije slika, P' i matrice normalizacije originala
    matrica = np.matmul(matrica_p, originali_t)
    matrica = np.matmul(la.inv(slike_t), matrica)

    return np.round(matrica,decimals=10)
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        koord.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ', ' + str(y), (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)     

img = cv2.imread("ppgr11.jpg", 1)
img = cv2.resize(img,dsize=(800,600))
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

while True:
    cv2.imshow("image",img)
    if cv2.waitKey(1) & len(koord) == 6:
        break

img = cv2.imread("ppgr12.jpg", 1)
img = cv2.resize(img,dsize=(800,600))
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

while True:
    cv2.imshow("image",img)
    if cv2.waitKey(1) & len(koord) == 12:
        break
cv2.destroyAllWindows()

originali = []
for i in range(6):
    originali.append(homogena(koord[i]))

slike = []
for i in range(6):
    koord[i+6][0] +=800
    slike.append(homogena(koord[i+6]))

matrica = dltnorm_algoritam(6, originali, slike)
matrica = np.round(matrica,decimals=10)

img = cv2.imread("ppgr11.jpg", 1)
img = cv2.resize(img,dsize=(800,600))

img2 = cv2.imread("ppgr12.jpg", 1)
img2 = cv2.resize(img2,dsize=(800,600))
   
M = np.float32(matrica)
sl = cv2.warpPerspective(img,M,(1600,600))

j = [[1,0,0],[0,1,0],[0,0,1]]
j = np.float32(j)

sl = cv2.warpPerspective(sl,j,(800,600))

panorama = cv2.hconcat([sl,img2])
cv2.imshow('Panorama',panorama)
cv2.waitKey(0)       


#Panorama sa ugradjenom funkcijom:
images = [img,img2]
stitcher = cv2.Stitcher.create()
res = stitcher.stitch(images)
#cv2.imshow('Panorama',res[1])
#cv2.waitKey(0)
