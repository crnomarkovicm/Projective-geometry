#Ukljucujemo potrebne biblioteke
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation
from numpy import linalg as LA
import numpy as np
import math

#Funkcija kojom na osnovu tri ulazna parametra racunamo matricu A
def Euler2A(fi, teta, psi):

    Rz = np.array([[math.cos(psi), -math.sin(psi), 0],
                   [math.sin(psi), math.cos(psi), 0],
                   [0, 0, 1]])

    Ry = np.array([[math.cos(teta), 0, math.sin(teta)],
                   [0, 1, 0],
                   [-math.sin(teta), 0, math.cos(teta)]])

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(fi), -math.sin(fi)],
                   [0, math.sin(fi), math.cos(fi)]])

    return Rz.dot(Ry).dot(Rx)

#Funkcija koja predstavlja matricu preko ose i ugla
def AxisAngle(A):
    
    if round(LA.det(A)) != 1:
        print("Determinanta je razlicita od 1")
        return

    if np.any(np.round(A.dot(A.T),6) != np.eye(3)):
        print("Matrica A nije ortogonalna")
        return
    
    lambdas, vector = np.linalg.eig(A, )
    for i in range(len(lambdas)):
        if np.round(lambdas[i], 6) == 1.0:
            p = np.real(vector[:, i])

    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    if p1 == 0 and p2 == 0 and p3 == 0:
        print("Ne smeju sve tri koordinate da budu 0")
        return
    elif p1 != 0:
        u = np.cross(p, np.array([-p1, p2, p3]))
    elif p2 != 0:
        u = np.cross(p, np.array([p1, -p2, p3]))
    else:
        u = np.cross(p, np.array([p1, p2, -p3]))
    u = u/math.sqrt(u[0]**2+u[1]**2+u[2]**2)

    up = A.dot(u)

    fi = np.round(math.acos(np.sum(u*up)), 5)
    if np.round(np.sum(np.cross(u, up)*p), 5) < 0:
        p = (-1)*p

    return [p, fi]

#Funkcija koja vraca jedinicni kvaternion dobijen preko ose i ugla
def AxisAngle2Q(p, fi):
    w = math.cos(fi/2.0)
    norm = np.linalg.norm(p)
    if norm != 0:
        p = p/norm
    [x, y, z] = math.sin(fi/2.0) * p
    return [x, y, z, w]

#Linearna interpolacija, pomocna funkcija za Slerp
def lerp(q1, q2, tm, t):
    return (1-(t/tm))*q1 + (t/tm)*q2

#Sferna linearna interpolacija
def slerp(q1, q2, tm, t):
    cos0 = np.dot(q1, q2)
    if cos0 < 0:
        q1 = -1 * q1
        cos0 = -cos0
    if cos0 > 0.95:
        return lerp(q1, q2, tm, t)
    angle = math.acos(cos0)
    q = (math.sin(angle*(1-t/tm)/math.sin(angle)))*q1 + (math.sin(angle*(t/tm)/math.sin(angle)))*q2
    return q

#Vraca konjugovani kvaternion zadatog kvaterniona
def q_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

#Funkcija koja mnozi dva kvaterniona
def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return x, y, z, w

#Funkcija kojom vrsimo rotaciju v kvaternionom tako sto radimo q*v*q_conj. V moramo predstaviti kao kvaternion pa zato dodajemo 0 na kraj (w=0)
def transform(v, q):
    v1 = (v[0], v[1], v[2], 0.0)
    return q_mult(q_mult(q, v1), q_conjugate(q))[:-1]
    
#Zadajemo broj frejmova (koraka)
tm = 100
#Pozicija centra prvog objekta
pos1 = np.array([4, 0, 3])

#Pozicija centra drugog objekta
pos2 = np.array([9, 1, 7])

#Orijentacije, tj Ojlerovi uglovi prvog objekta
e_angle1 = np.array([math.radians(-30), math.radians(45), math.radians(60)])

#Orijentacije, tj Ojlerovi uglovi prvog objekta
e_angle2 = np.array([math.radians(20), math.radians(-30), math.radians(90)])

#Na osnovu Ojlerovih uglova prvog objekta dobijamo matricu A, zatim je funkcijom AxisAngle(A) predtsvljamo pomocu ose i ugla a onda na osnovu njih dobijamo jedinicni kvaternion
A = Euler2A(e_angle1[0], e_angle1[1], e_angle1[2])
N = AxisAngle(A)
p = N[0] #osa
fi = N[1] #ugao
q1 = AxisAngle2Q(p,fi)

#Na osnovu Ojlerovih uglova drugog objekta dobijamo matricu A, zatim je funkcijom AxisAngle(A) predtsvljamo pomocu ose i ugla a onda na osnovu njih dobijamo jedinicni kvaternion
A = Euler2A(e_angle2[0], e_angle2[1], e_angle2[2])
N = AxisAngle(A)
p = N[0] #osa
fi = N[1] #ugao
q2 = AxisAngle2Q(p,fi)

#Postavljamo koordinatne ose i labele, kao i odgovarajuce parametre 
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim((0, 10))
ax.set_ylim((0, 10))
ax.set_zlim((0, 10))

ax.view_init(10, 0)

#Postavljamo boje na crvenu, zelenu i plavu
colors = ['r', 'g', 'b']

lines = np.array(sum([ax.plot([], [], [], c=c) for c in colors], []))

#Ovo su pocetne i krajnje tacke duzi od kojih krecemo
startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
endpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#Iscrtavanje pocetne pozicije
#U petlji transformisemo pocetnu i krajnju tacku svake od tri duzi, a zatim na dobijenu transformaciju dodajemo poziciju centra jer se inicijalno sve nalazi u koordinatnom pocetku
for i in range(3):
	start = transform(startpoints[i], q1)
	end = transform(endpoints[i], q1)
	start += pos1
	end += pos1
    #Iscrtavamo duz na grafiku
	ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])


startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
endpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

#Iscrtavanje krajnje pozicije
#U petlji transformisemo pocetnu i krajnju tacku svake od tri duzi, a zatim na dobijenu transformaciju dodajemo poziciju centra jer se inicijalno sve nalazi u koordinatnom pocetku
for i in range(3):
	start = transform(startpoints[i], q2)
	end = transform(endpoints[i], q2)
	start += pos2
	end += pos2
	ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])

#Init funkcija za animaciju
def init():
	for line in lines:
		line.set_data(np.array([]), np.array([]))
		line.set_3d_properties(np.array([]))

	return lines

#Funkcija u kojoj za svaki od tm frameova pozivamo slerp algoritam, ona nam vraca kvaternion a onda nase duzi rotiramo pomocu njega
def animate(i):
	q = slerp(np.array(q1), np.array(q2), tm,i)
    #Racunamo korak koji treba dodati tackama vektora kako bi se konstantno od pocetnih translirao ka krajnjim tackama a ne samo rotirao u koordinatnom pocetku
	k = i *(pos2-pos1)/tm
	for line, start, end in zip(lines, startpoints, endpoints):
		start = transform(np.array(start), np.array(q))
		end = transform(np.array(end), np.array(q))
		start += pos1 + k
		end += pos1 + k

		line.set_data(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
		line.set_3d_properties(np.array([start[2], end[2]]))

	fig.canvas.draw()
	return lines

anim = animation.FuncAnimation(fig,frames=tm, func=animate, init_func=init, interval=5, blit=True)
anim.save('animation.gif')
