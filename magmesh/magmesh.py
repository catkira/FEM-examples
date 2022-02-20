import pickle
import numpy as np
import matplotlib as plt
from spylizard import *

mymesh = mesh()
mymesh.load("magmesh.msh")

mur_frame = 1000
mur_magnet = 1
b_r_magnet = 1.5
formulationType = 1

pi = getpi()
mu0 = 4*pi*1e-7
conductor = 1
shield = 2
air = 3
inf = 4
wholedomain = selectunion([conductor, shield, air])
mu = parameter()
mu.setvalue(wholedomain, mu0)
mu.setvalue(shield, mu0*mur_frame)

magnetostatics = formulation()
spantree = spanningtree([wholedomain])
A = field("hcurl", spantree)
A.setorder(wholedomain, 1)
A.setconstraint(inf)
A.setgauge(wholedomain)
magnetostatics += integral(wholedomain,  1/mu * curl(dof(A)) * curl(tf(A)))
magnetostatics += integral(conductor, - array3x1(0,0,1)*tf(A))

magnetostatics.generate()
matA = magnetostatics.A(True)
print("A has " + str(matA.countrows()) + " rows")
print("ainds has " + str(matA.getainds().countrows()) + " rows")
print("dinds has " + str(matA.getdinds().countrows()) + " rows")
sol = solve(magnetostatics.A(), magnetostatics.b())

A.setdata(wholedomain, sol)
b = curl(A)
b.write(wholedomain, "b.pos", 1)
norm(b).write(wholedomain, "b_norm.pos", 1)




