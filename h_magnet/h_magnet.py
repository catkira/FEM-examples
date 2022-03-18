import pickle
import numpy as np
import matplotlib.pyplot as plt
from spylizard import *
import time

from scipy.sparse import *
import sys
import pkg_resources
if 'petsc4py' in pkg_resources.working_set.by_key:
    hasPetsc = True
    import petsc4py
    petsc4py.init(sys.argv)        
    from petsc4py import PETSc
else:
    print("Warning: no petsc4py found, solving will be very slow!")
np.set_printoptions(linewidth=400)    

def is_symmetric(m):

    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]
    check = np.allclose(vl, vu, atol=0.0000001)
    return check


def solve2(A, b, method='np'):
    start = time.time()
    if method == 'sparse':
        from scipy.sparse.linalg import inv    
        A = csc_matrix(A)
        u = inv(A) @ b
    elif method == 'petsc':
        if not hasPetsc:
            print("petsc is not available on this system")
            sys.exit()            
        opts = PETSc.Options()
        #opts.setValue("st_pc_factor_shift_type", "NONZERO")    
        n = len(b)   
        Ap = PETSc.Mat().createAIJWithArrays(size=(n, n),  csr=(A.indptr, A.indices, A.data), comm=PETSc.COMM_WORLD)        
        Ap.setUp()
        Ap.assemblyBegin()
        Ap.assemblyEnd()
        #if Ap.isSymmetric() == False:
        #    print("Warning: A is not symmetric!")        
        bp = PETSc.Vec().create() 
        bp.setSizes(n)
        bp.setFromOptions()
        bp.setUp()
        bp.setValues(range(n),b)    
        bp.assemblyBegin()
        bp.assemblyEnd()    
        up = PETSc.Vec().create()
        up.setSizes(n)
        up.setFromOptions()         
        up.setUp()
        up.assemblyBegin()
        up.assemblyEnd()         
        ksp = PETSc.KSP().create()
        ksp.getPC().setType(PETSc.PC.Type.LU)
        ksp.getPC().setFactorSolverType('mumps')
        ksp.setOperators(Ap)        
        ksp.setFromOptions()
        #ksp.getPC().setFactorSolverType('petsc')
        ksp.setUp()
        #ksp.getInitialGuessNonzero()        
        #ksp.setType(PETSc.KSP.Type.PREONLY)
        #ksp.setType(PETSc.KSP.Type.CG)  # conjugate gradient
        #ksp.setType(PETSc.KSP.Type.GMRES)
        #ksp.getPC().setFactorSolverType("superlu_dist")
        #ksp.getPC().setType('cholesky') # cholesky
        #ksp.getPC().setType('icc') # incomplete cholesky
        print(f'Solving {n} dofs with: {ksp.getType():s}')
        ksp.solve(bp, up)
        print(f"Converged in {ksp.getIterationNumber():d} iterations.")
        u = np.array(up)
    elif method == 'np':
        #u = np.linalg.inv(A.toarray()) @ b
        from scipy.sparse.linalg import spsolve
        #u = spsolve(A,b)
        #from scipy.sparse.linalg import cg
        #u = cg(A,b)
        from scipy.sparse.linalg import splu
        B = splu(csc_matrix(A))
        B.solve(b)
    else:
        print("unknown method")
        sys.exit()
    numDofs = len(u)
    #assert numDofs == countAllFreeDofs()
    #putSolutionIntoFields(u)
    stop = time.time()
    print(f"solved {numDofs} dofs in {stop - start:.2f} s")    

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
    spantree = spanningtree([inf])
    spantree.write("h_magnet_spanntree.pos")
    print(f'edges in tree {spantree.countedgesintree()}')
    A = field("hcurl", spantree)
    A.setorder(wholedomain, 0)
    A.setconstraint(inf)
    A.setgauge(wholedomain)
    magnetostatics += integral(wholedomain,  1/mu * curl(dof(A)) * curl(tf(A)))
    magnetostatics += integral(magnet, - br/mu * curl(tf(A)))

magnetostatics.generate()
matA = magnetostatics.A(True)
vecb = magnetostatics.b(True)
vecb.write("rhs.txt")
print("A has " + str(matA.countrows()) + " rows")
print("ainds has " + str(matA.getainds().countrows()) + " rows")
print("dinds has " + str(matA.getdinds().countrows()) + " rows")

if False:
    fig, axs = plt.subplots(2)
    fig.suptitle('SL')
    indexptr = matA.getarows().getvalues()
    nnzs = np.empty(len(indexptr)-1)
    for i in range(len(indexptr)-1):
        nnzs[i] = indexptr[i+1] - indexptr[i]
    axs[0].hist(nnzs, bins=range(30))

    data = matA.getavals().getvalues()
    axs[1].hist(data, bins=(np.arange(31)-15)/15*np.max(data))
    plt.show()

if True:
    data = matA.getavals().getvalues()
    indptr = matA.getarows().getvalues()
    indices = matA.getacols().getvalues()
    A2 = csr_matrix((data, indices, indptr), shape=[matA.getainds().countrows(), matA.getainds().countrows()])
    rhs = matA.eliminate(vecb).getallvalues().getvalues()
    np.max(rhs)
    result = solve(indptr, indices, data, rhs)

sol = solve(magnetostatics.A(), magnetostatics.b(), soltype="lu")

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
b.write(wholedomain, "b.vtk", 1)
norm(b).write(wholedomain, "b_norm.pos", 1)





