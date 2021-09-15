import cv2
import numpy as np
from numpy import linalg as la

#Niz u koji smestamo koordinate koje korisnik oznaci misem
koord = []

#Funkcija koja pretvara afine koordinate tacke u homogene
def homogena(tacka):
    return [tacka[0], tacka[1], 1]

#Dlt algoritam koji koristimo kako bimso odredili matricu projektivnog preslikavanja izmedju oznacenog cevtorougla i pravougaonika
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

#Funkcija koja odredjuje sta se desava na klik misa
def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        koord.append((x,y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ', ' + str(y), (x, y), font, 1, (130, 20, 21), 2)
        cv2.imshow('image', img)

#Funkcija koja sluzi za sortiranje tacaka kako ne bi bio bitan redosled temena koja korisnik bira (npr 1,3,4,2 umesto 1,2,3,4)
def sortiraj_originale(originali):
    sortirane = []
    sort = []
    for i in range(4):
        sortirane.append(originali[i])
        sort.append(originali[i])
        
    sortirane.sort()
 
    if(sortirane[1][1] < sortirane[0][1]):
        sort[0] = sortirane[1]
        sort[1] = sortirane[0]
    else:
        sort[1] = sortirane[1]
        sort[0] = sortirane[0]
    if(sortirane[2][1] < sortirane[3][1]):
        sort[2] = sortirane[3]
        sort[3] = sortirane[2]
    else:
        sort[3] = sortirane[3]
        sort[2] = sortirane[2]
    return sort

#Funkcija koja racuna pravougaonik u koji slikamo nas cetvorougao.
#Za duzinu i sirinu pravougaonika uzimala sam srednje vrednosti naspramnih stranica cetvorougla
def nadji_slike(originali):
    slike = []
    originali_n = sortiraj_originale(originali)
    duzina = ((originali_n[2][0] - originali_n[1][0])+(originali_n[3][0] - originali_n[0][0]))/2.0
    sirina = ((originali_n[1][1] - originali_n[0][1])+(originali_n[2][1] - originali_n[3][1]))/2.0  
    slike.append((originali_n[0][0], originali_n[0][1], originali_n[0][2]))
    slike.append((originali_n[0][0], originali_n[0][1] + sirina, originali_n[0][2]))
    slike.append((originali_n[0][0] + duzina, originali_n[0][1] + sirina, originali_n[0][2]))     
    slike.append((originali_n[0][0] + duzina, originali_n[0][1], originali_n[0][2])) 

    return slike

#Ucitavamo sliku i prilagodjavamo njenu velicinu prozoru kako bi bilo lakse koristiti program
img = cv2.imread("1.jpg", 1)
img = cv2.resize(img,(800,600))
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

#Podesavamo da se ucitavanje prekine nakon sto su izabrane koordinate cetvorougla
while True:
    if cv2.waitKey(1) & len(koord) == 4:
        break

#Uzimamo homogene koordinate ucitanih
originali = []
for i in range(4):
    originali.append(homogena(koord[i]))

#Sortiramo originale
originali_n = sortiraj_originale(originali)

#Nalazimo pravougaonik
slike = nadji_slike(originali)

#Trazimo preslikavanje koje slika nas cetvorougao u pravougaonik
matrica = dlt_algoritam(4, originali_n, slike)
matrica = np.round(matrica,decimals=10)


img = cv2.imread("1.jpg", 1)
img = cv2.resize(img,(800,600))

#Primenjujemo preslikavanje na pocetnu sliku kako bismo otklonili projektivnu distorziju
M = np.float32(matrica)
dst = cv2.warpPerspective(img,M,(800,600))

#Finalna slika
cv2.imshow('img',dst)
cv2.waitKey(0)
