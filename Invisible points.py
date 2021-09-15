import numpy as np

#Klasa koja predstavlja tacku zadatu njenim afinim koordinatama
class Tacka:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
#Funkcija koja vraca vektor predstavnik tacke t
def homogena(t):
    return [t.x, t.y, 1]

#Funkcija koja vraca afine koordinate tacke (za izlaz iz programa)
def dehomog(T):
    return Tacka(round(T[0]/T[2]), round(T[1]/T[2]))

#Funkcija koja odredjuje 8., nevidljivu tacku, na osnovu vidljivih 7
def nevidljiva_tacka(A, B, C, D, A1, B1, C1):
    A = homogena(A)
    B = homogena(B)
    C = homogena(C)
    D = homogena(D)
    A1 = homogena(A1)
    B1 = homogena(B1)
    C1 = homogena(C1)

#Trazimo presek ivica AB i A1B1
    AB = np.cross(A,B)
    A1B1 = np.cross(A1,B1)
    P = np.cross(AB, A1B1)

#Trazimo presek ivica BC i AD
    BC = np.cross(B,C)
    AD = np.cross(A,D)
    Q = np.cross(BC, AD)

#Trazenu tacku T mozemo dobiti, npr, kao presek PC1 i A1Q
    PC1 = np.cross(P,C1)
    A1Q = np.cross(A1,Q)
    T = np.cross(PC1, A1Q)

    return dehomog(T)
     
#Koordinate temena sa prilozene slike
A = Tacka(49, 589)
B = Tacka(497, 573)
C = Tacka(597, 66)
D = Tacka(217, 134)
A1 = Tacka(169, 711)
B1 = Tacka(583, 713)
C1 = Tacka(660, 228)

#Dobijamo nevidljivo teme
t = nevidljiva_tacka(A, B, C, D, A1, B1, C1)
print("Nevidljivo teme sa prilozene slike: (", t.x, ",", t.y , ")")

#Testiranje funkcije za primer sa sajta
C1 = Tacka(595, 301)
B1 = Tacka(292, 517)
A1 = Tacka(157, 379)
C = Tacka(665, 116)
B = Tacka(304, 295)
A = Tacka(135, 163)
D = Tacka(509, 43)

#Resenje je priblizno onom koje je dobijeno u Mathematici zbog razlicitih preciznosti
t = nevidljiva_tacka(A, B, C, D, A1, B1, C1)
print("Nevidljivo teme na primeru sa sajta: (", t.x, ",", t.y , ")")

