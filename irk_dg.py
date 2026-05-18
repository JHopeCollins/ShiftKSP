import numpy as np
from firedrake import *
from irksome import Dt, TimeStepper, RadauIIA, GaussLegendre

Print = PETSc.Sys.Print
options = PETSc.Options()

nx = options.getInt("nx", 100)
mesh = UnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

equation_type = options.getString("equation", "advection")

cfl = options.getReal("cfl", 0.6)
h = 1/nx

if equation_type == "advection":
    degree = options.getInt("fs_degree", 0)
    variant = options.getString("fs_variant", "integral")
    V = FunctionSpace(mesh, "DG", degree, variant=variant)
    
    q = Function(V)
    v = TestFunction(V)
    
    ubar = options.getReal("ubar", 1.0)
    u = as_vector([Constant(ubar)])
    
    n = FacetNormal(mesh)
    un = Constant(0.5)*(dot(u, n) + abs(dot(u, n)))
    
    flux = (un*q)('+') - (un*q)('-')
    
    F = (
        inner(Dt(q), v)*dx
        + q*div(v*u)*dx
        + (v('+')*flux - v('-')*flux)*dS
    )
    
    dt = cfl*h/ubar

    K_ksp_type = 'gmres'

elif equation_type == "diffusion":
    degree = options.getInt("fs_degree", 1)
    V = FunctionSpace(mesh, "CG", degree)

    q = Function(V)
    v = TestFunction(V)

    nubar = options.getReal("nu", 1.0)
    nu = Constant(nubar)

    F = (
        inner(Dt(q), v)*dx
        + inner(nu*grad(q), grad(v))*dx
    )

    dt = cfl*h*h/nubar

    K_ksp_type = 'cg'

else:
    raise ValueError("Unrecognised {equation_type = }")

dt = Constant(dt)
t = Constant(0)
    
q.interpolate(sin(2*pi*x))

num_stages = options.getInt("irk_num_stages", 2)
tableau = {
    'radau': RadauIIA,
    'gl': GaussLegendre,
}[options.getString("irk_type", "radau")](num_stages)

monolithic_parameters = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'pc_factor_shift_type': 'nonzero',
}

ksp_its = options.getInt("ksp_its", 100)

eksm_parameters = {
    'snes_type': 'ksponly',
    "ksp_view": ":ksp_view.log",
    'ksp_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'eksm.IRKKroneckerPC',
    'pc_irkkron_shift_type': 'eigmin',
    'irkkron': {
        "ksp_converged_rate": None,
        "ksp_monitor": None,
        "ksp_rtol": 1e-5,
        "ksp_max_it": ksp_its,

        # # just solve the kronecker matrix
        # "ksp_type": "gmres",
        # "ksp_gmres_restart": ksp_its,
        # "pc_type": "none",

        # solve the sylvester equation
        "ksp_type": "python",
        "ksp_python_type": "eksm.SylvesterEKSP",
        "ksp_sylvester_adaptive_tol": "yes",
        "sylvester_A": {
            'ksp_rtol': 1e-5,
            'ksp_type': K_ksp_type,
            'pc_type': 'jacobi',
        },
        "sylvester_M": {
            'ksp_type': 'preonly',
            'pc_type': 'cholesky',
        },
    },
}

parameters = eksm_parameters

from irksome.tools import IA, AI
stepper = TimeStepper(
    F, tableau, t, dt, q,
    splitting=AI,
    solver_parameters=parameters,
    options_prefix="")

stepper.advance()
