import pickle
import numpy as np
import matplotlib as plt
from spylizard import *

mymesh = mesh()
mymesh.load("h_magnet.msh")

mur_frame = 1000
mur_magnet = 1
b_r_magnet = 1.5
formulationType = 1

pi = getpi()
mu0 = 4*pi*1e-7
magnet = 0
frame = 1
air = 2
inf = 3
wholedomain = selectunion([magnet, air, frame, inf])
mu = parameter()
mu.setvalue(wholedomain, mu0)
mu.setvalue(frame, mu0*mur_frame)
mu.setvalue(magnet, mu0*mur_magnet)
br = parameter(3,1)
br.setvalue(wholedomain, array3x1(0, 0, 0))
br.setvalue(magnet, array3x1(0, 0, b_r_magnet))

magnetostatics = formulation()
if formulationType == 0:
    phi = field("h1")
    phi.setorder(wholedomain, 2)
    phi.setconstraint(inf)
    magnetostatics += integral(wholedomain, - grad(dof(phi)) * mu * grad(tf(phi)))
    magnetostatics += integral(magnet, br * grad(tf(phi)))
else:
    spantree = spanningtree([wholedomain])
    A = field("hcurl", spantree)
    A.setorder(wholedomain, 1)
    A.setconstraint(inf)
    A.setgauge(wholedomain)
    magnetostatics += integral(wholedomain,  1/mu * curl(dof(A)) * curl(tf(A)))
    magnetostatics += integral(magnet, - br/mu * curl(tf(A)))

magnetostatics.generate()
matA = magnetostatics.A(True)
print("A has " + str(matA.countrows()) + " rows")
print("ainds has " + str(matA.getainds().countrows()) + " rows")
print("dinds has " + str(matA.getdinds().countrows()) + " rows")
sol = solve(magnetostatics.A(), magnetostatics.b())

if formulationType == 0:
    phi.setdata(wholedomain, sol)
    h = (-grad(phi))
    b = mu*h + br
else:
    A.setdata(wholedomain, sol)
    b = curl(A)
    h = b/mu - br/mu0
B_outside = norm(b).interpolate(wholedomain, [0.04,0.005,0.01])[0]
print("|B| in frame is " + str(B_outside) + " T")
h.write(wholedomain, "h.pos", 1)
b.write(wholedomain, "b.pos", 1)
norm(b).write(wholedomain, "b_norm.pos", 1)




