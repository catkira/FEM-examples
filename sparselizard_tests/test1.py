import pickle
import numpy as np
import matplotlib as plt
from spylizard import *

mymesh = mesh("disk.msh")
v = field("h1")
vol = 1
sur = 2
top = 3
x = field("x")
y = field("y")
z = field("z")
expr = expression(x+y, 2*x, 0)
#expr = expression (2,3,[12,v,v*(1-v),3,14-v,0])
#expr = expression(3,3,[1,2,3,4,5,6])
expr = expression(3,3,[1,2,3,4,5,6])
#expr.print()
maxdata = (2*x).max(vol, 1)
xyzcoord = [0.5, 0.6, 0.05, 0.1, 0.1, 0.1]
interpolated = []
isfound = []
array3x1(x,y,z).interpolate(vol, xyzcoord, interpolated, isfound)
print(interpolated)
print(isfound)
expr = expression(10)
integralvalue = expr.integrate(vol, 4)
print(v.countcomponents())
outvals=v.getnodalvalues(indexmat(6,1,[1,2,3,4,5,6]))
outvals.print()
normal(vol).write(top, "normal.pos", 1)
tangent().write(top, "tangent.pos", 1)
expr = expression(x+y, 2*x, 0)
expr = expression(1,1, x+y)
(x+y).write(top, "xplusy.pos",1)
y = abs(x)
v1 = array3x1(1,2,3)
v2 = array3x1(1,0,0)
(v1*v2).print()
crossproduct(v1,v2).print()
