import pickle
import numpy as np
import matplotlib as plt
from spylizard import *

mymesh = mesh()
mymesh.load("h_magnet_octant.msh")


mur_frame = 1000
mur_magnet = 1
b_r_magnet = 1.5
formulationType = 1

pi = getpi()
mu0 = 4*pi*1e-7
magnet = 1
frame = 2
air = 3
inf = 4
innerXZBoundary = 5
innerYZBoundary = 6
innerXYBoundary = 7
magnetXYBoundary = 8
wholedomain = selectunion([magnet, air, frame, inf, innerXZBoundary, innerXYBoundary, innerYZBoundary, magnetXYBoundary])
wholeXYBoundary = selectunion([innerXYBoundary, magnetXYBoundary])
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
    phi.setconstraint(wholeXYBoundary) # perfect magnetic conductor
    # natural neumann boundary condition on innerXZBoundary and innerYZBoundary is magnetic isolator
    magnetostatics += integral(wholedomain, - grad(dof(phi)) * mu * grad(tf(phi)))
    magnetostatics += integral(magnet, br * grad(tf(phi)))
else:
    spantree = spanningtree([wholedomain])
    spantree.write("octant_spantree.pos")
    A = field("hcurl", spantree)
    A.setorder(wholedomain, 0)   # TODO: make higher orders work without gauging
    A.setconstraint(innerXZBoundary) # magnetic isolator
    A.setconstraint(innerYZBoundary) # magnetic isolator
    A.setgauge(wholedomain) # gauging with tree-cotree destroys the PMC BC
    # natural neumann boundary condition on wholeXYBoundary is perfect magnetic conductor
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
    phi.printvalues(False)
    h = (-grad(phi))
    b = mu*h + br
else:
    A.setdata(wholedomain, sol)
    b = curl(A)
    h = b/mu - br/mu0
B_outside = norm(b).interpolate(wholedomain, [0.04,0.005,0.01])[0]
print("|B| in frame is " + str(B_outside) + " T")
h.write(wholedomain, "octant_h.pos", 1)
b.write(wholedomain, "octant_b.pos", 1)
norm(b).write(wholedomain, "octant_b_norm.pos", 1)




