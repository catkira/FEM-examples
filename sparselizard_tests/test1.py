import pickle
import numpy as np
import matplotlib as plt
from spylizard import *
from spylizardstubs import *

mymesh = mesh("disk.msh")
vol = 1
sur = 2
top = 3
x = field("x")
y = field("y")
expr = expression(x+y, 2*x, 0)
expr.print()

