import numpy as np
from firedrake import *
from irksome import Dt, TimeStepper, RadauIIA, GaussLegendre

Print = PETSc.Sys.Print
options = PETSc.Options()

nx = options.getInt("nx", 50)
Lx = nx
mesh = IntervalMesh(nx, Lx)
x, = SpatialCoordinate(mesh)

degree = options.getInt("fs_degree", 0)
variant = options.getString("fs_variant", "integral")
V = FunctionSpace(mesh, "DG", degree, variant=variant)

q = Function(V)
v = TestFunction(V)

u = as_vector([Constant(1.0)])

n = FacetNormal(mesh)
un = Constant(0.5)*(dot(u, n) + abs(dot(u, n)))

flux = (un*q)('+') - (un*q)('-')

mu = Constant(options.getReal("mu", 0.0))

F = (
    inner(Dt(q), v)*dx
    + q*div(v*u)*dx
    + (v('+')*flux - v('-')*flux)*dS
    + mu*q*v*dx
)

q.interpolate(sin(2*pi*x/Lx))

t = Constant(0)
dt = Constant(options.getReal("dt", 1.0))

num_stages = options.getInt("irk_num_stages", 2)
tableau = {
    'radau': RadauIIA,
    'gl': GaussLegendre,
}[options.getString("irk_type", "radau")](num_stages)

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}

monolithic_parameters = {
    'snes_type': 'ksponly',
    **lu_params,
}

eksm_parameters = {
    'snes_type': 'ksponly',
    'ksp_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'eksm.IRKKroneckerPC',
    'irkkron': {
        "ksp_view": ":irkkronview.log",
        "ksp_converged_rate": None,
        "ksp_monitor": None,
        "ksp_rtol": 1e-3,
        "ksp_max_it": 100,
        "ksp_type": "gmres",
        "ksp_gmres_restart": 100,
        # "pc_type": "none",
        "ksp_type": "python",
        "ksp_python_type": "eksm.SylvesterEKSP",
        "sylvester": {**lu_params},
    },
}

parameters = eksm_parameters

from irksome.tools import IA, AI
stepper = TimeStepper(
    F, tableau, t, dt, q,
    # splitting=AI,
    solver_parameters=parameters,
    options_prefix="")

stepper.advance()
