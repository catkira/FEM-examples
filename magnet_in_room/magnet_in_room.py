import pickle
import numpy as np
import matplotlib as plt
from spylizard import *

mymesh = mesh()
mymesh.load("magnet_in_room.msh")

mur_wall = 1000
b_r_magnet = 1.5
formulationType = 1
Az_only = True

pi = getpi()
mu0 = 4*pi*1e-7
magnet = 1
insideAir = 2
wall = 3
outsideAir = 4
inf = 5
wholedomain = selectunion([magnet, insideAir, wall, outsideAir, inf])
mu = parameter()
mu.setvalue(wholedomain, mu0)
mu.setvalue(wall, mu0*mur_wall)

magnetostatics = formulation()
if formulationType == 0:
    phi = field("h1")
    phi.setorder(wholedomain, 2)
    phi.setconstraint(inf)
    br = parameter(2,1)
    br.setvalue(wholedomain, array2x1(0, 0))
    br.setvalue(magnet, array2x1(-b_r_magnet, 0))
    magnetostatics += integral(wholedomain, - grad(dof(phi)) * mu * grad(tf(phi)))
    magnetostatics += integral(magnet, br * grad(tf(phi)))
else:
    if Az_only:
        Az = field("h1")
        Az.setorder(wholedomain, 1)
        Az.setconstraint(inf)
        A = array3x1(0,0,Az)
    else:
        spantree = spanningtree([wholedomain])
        A = field("hcurl", spantree)
        A.setorder(wholedomain, 1)
        A.setconstraint(inf)
        A.setgauge(wholedomain)
    br = parameter(3,1)
    br.setvalue(wholedomain, array3x1(0, 0, 0))
    br.setvalue(magnet, array3x1(-b_r_magnet, 0, 0))
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
    if Az_only:
        Az.setdata(wholedomain, sol)
        A = array3x1(0,0,Az)
    else:
        A.setdata(wholedomain, sol)
    b = curl(A)
    h = b/mu - br/mu0
B_outside = norm(b).interpolate(wholedomain, [2.1,0,0])[0]
print("|B| outside of wall is " + str(B_outside) + " T")
h.write(wholedomain, "magnet_in_room_h.pos", 1)
b.write(wholedomain, "magnet_in_room_b.pos", 1)
norm(b).write(wholedomain, "magnet_in_room_b_norm.pos", 1)




