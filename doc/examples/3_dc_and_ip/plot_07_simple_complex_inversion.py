#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Complex-valued electrical modeling
----------------------------------

In this example an electrical complex-valued forward modeling is conducted. The
use of complex resistivities implies an out-of-phase polarization response of
the subsurface, commonly being measured in the frequency domain as complex
resistivity (CR), or, if multiple frequencies are measured, as the spectral
induced polarization (SIP). Please note that the time-domain induced
polarization (TDIP) is a compound signature of a wide range of frequencies.

It is common to parameterize the complex resistivities using magnitude (in
:math:`\Omega m`) and phase :math:`\phi` (in mrad), although the PyGimli
forward operator takes real and imaginary parts.
"""
# sphinx_gallery_thumbnail_number = 5
import numpy as np
import matplotlib.pylab as plt

# import matplotlib.pylab as plt
from pygimli.physics.ert.ert import ERTModelling

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

###############################################################################
# * generate mesh
# * read in data
# * determine starting model
# * determine response and Jacobian for starting model
# * compute model update
# * m1 = m0 + delta m
# if required, repeat
###############################################################################


def plot_fwd_model(axes):
    # Mesh generation
    world = mt.createWorld(
        start=[-55, 0], end=[105, -80], worldMarker=True)

    polarizable_anomaly = mt.createCircle(
        pos=[40, -7], radius=5, marker=2
    )

    plc = mt.mergePLC((world, polarizable_anomaly))

    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        plc.createNode(s + [0.0, -0.2])

    mesh_coarse = mt.createMesh(plc, quality=33)
    # additional refinements
    mesh = mesh_coarse.createH2()

    # Prepare the model parameterization
    # We have two markers here: 1: background 2: circle anomaly
    # Parameters must be specified as a complex number, here converted by the
    # utility function :func:`pygimli.utils.complex.toComplex`.
    rhomap = [
        [1, pg.utils.complex.toComplex(100, 0 / 1000)],
        # Magnitude: 50 ohm m, Phase: -50 mrad
        [2, pg.utils.complex.toComplex(50, -50 / 1000)],
    ]

    # For visualization, map the rhomap into the actual mesh, resulting in a
    # rho vector with a complex resistivity associated with each mesh cell.
    rho = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)
    pg.show(
        mesh,
        data=np.log(np.abs(rho)),
        ax=axes[0],
        label=r"$log_{10}(|\rho|~[\Omega m])$"
    )
    pg.show(mesh, data=np.abs(rho), ax=axes[1], label=r"$|\rho|~[\Omega m]$")
    pg.show(
        mesh, data=np.arctan2(np.imag(rho), np.real(rho)) * 1000,
        ax=axes[2],
        label=r"$\phi$ [mrad]",
        cMap='jet_r'
    )
    fig.tight_layout()
    fig.show()


# Create a measurement scheme for 51 electrodes, spacing 1
scheme = ert.createERTData(
    elecs=np.linspace(start=0, stop=50, num=51),
    schemeName='dd'
)
m = scheme['m']
n = scheme['n']
scheme['m'] = n
scheme['n'] = m
scheme.set('k', [1 for x in range(scheme.size())])

###############################################################################
# Mesh generation
# world = mt.createRectangle([-10, 0], [60, -20])
world = mt.createWorld(
    start=[-15, 0], end=[65, -30], worldMarker=False, marker=2)

# local refinement of mesh near electrodes
for s in scheme.sensors():
    world.createNode(s + [0.0, -0.4])

mesh_coarse = mt.createMesh(world, quality=33)
mesh = mesh_coarse.createH2()
for nr, c in enumerate(mesh.cells()):
    c.setMarker(nr)

# import IPython
# IPython.embed()
# additional refinements
# mesh = mesh_coarse.createH2()

# pg.show(plc, marker=True)
# pg.show(plc, markers=True)
# pg.show(mesh)
###############################################################################
# Start model
start_model = np.ones(mesh.cellCount()) * pg.utils.complex.toComplex(
    80, -0.01 / 1000)
print('Start model', start_model)

###############################################################################
fop = ERTModelling(
    sr=False,
    verbose=True,
)
fop.setComplex(True)
fop.setData(scheme)
fop.setMesh(mesh, ignoreRegionManager=True)
fop.mesh()

###############################################################################
# Response for Starting model
start_re_im = pg.utils.squeezeComplex(start_model)

f_0 = np.array(fop.response(start_re_im))

J_block = fop.createJacobian(start_re_im)
J_re = np.array(J_block.matrices()[0])
J_im = np.array(J_block.matrices()[1])
J0 = J_re + 1j * J_im

###############################################################################
# determine constraints (regularization matrix)
rm = fop.regionManager()
rm.setVerbose(True)
rm.setConstraintType(2)

Wm = pg.matrix.SparseMapMatrix()
rm.fillConstraints(Wm)
Wm = pg.utils.sparseMatrix2coo(Wm)  # .toarray()
###############################################################################
# read-in data and determine error parameters
data_rre_rim = np.loadtxt('data_rre_rim.dat')
N = int(data_rre_rim.size / 2)
d_rcomplex = data_rre_rim[:N] + 1j * data_rre_rim[N:]

dpha = np.arctan2(d_rcomplex.imag, d_rcomplex.real) * 1000

fig, ax = plt.subplots()
ert.showERTData(scheme, vals=dpha, ax=ax)
fig.savefig('dpha_ps.jpg', dpi=300)

# real part: log-magnitude
# imaginary part: phase [rad]
d_rlog = np.log(d_rcomplex)

# import IPython
# IPython.embed()
# exit()

# add some noise
np.random.seed(42)

noise_magnitude = np.random.normal(
    loc=0,
    scale=np.exp(d_rlog.real) * 0.04
)

# absolute phase error
noise_phase = np.random.normal(
    loc=0,
    scale=np.ones(N) * (0.5 / 1000)
)

d_rlog = np.log(np.exp(d_rlog.real) + noise_magnitude) + 1j * (
    d_rlog.imag + noise_phase)

# determine crude error estimations
rmag_linear = np.exp(d_rlog.real)
err_mag_linear = rmag_linear * 0.01 + np.min(rmag_linear)
err_mag_log = np.abs(1 / rmag_linear * err_mag_linear)
err_mag_log = np.ones_like(rmag_linear) * 0.02

Wd = np.diag(1.0 / err_mag_log)
WdTwd = Wd.conj().dot(Wd)
# import IPython
# IPython.embed()

###############################################################################
# naive inversion in log-log


def plot_inv_pars(filename, d, response, Wd):
    fig, axes = plt.subplots(1, 2, figsize=(20 / 2.54, 10 / 2.54))

    psi = Wd.dot(d - response)

    ert.showERTData(
        scheme, vals=psi.real, ax=axes[0],
        label=r"$(d' - f') / \epsilon$"
    )
    ert.showERTData(
        scheme, vals=psi.imag, ax=axes[1],
        label=r"$(d'' - f'') / \epsilon$"
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300)


m_old = np.log(start_model)
d = np.log(pg.utils.toComplex(data_rre_rim))
response = np.log(pg.utils.toComplex(f_0))
J = J0 / np.exp(response[:, np.newaxis]) * np.exp(m_old)[np.newaxis, :]
# lam = 20
lam = 100

plot_inv_pars('stats_it0.jpg', d, response, Wd)


for i in range(4):
    print('-' * 80)
    print('Iteration {}'.format(i + 1))

    term1 = J.conj().T.dot(WdTwd).dot(J) + lam * Wm.T.dot(Wm)
    term1_inverse = np.linalg.inv(term1)
    term2 = J.conj().T.dot(WdTwd).dot(d - response) - lam * Wm.T.dot(Wm).dot(
        m_old)
    model_update = term1_inverse.dot(term2)

    print('Model Update')
    print(model_update)

    m1 = np.array(m_old + 0.15 * model_update).squeeze()

    fig, axes = plt.subplots(2, 3, figsize=(26 / 2.54, 15 / 2.54))
    plot_fwd_model(axes[0, :])
    axes[0, 0].set_title('This row: Forward model')

    pg.show(
        mesh, data=m1.real, ax=axes[1, 0],
        cMin=np.log(50),
        cMax=np.log(100),
        label=r"$log_{10}(|\rho|~[\Omega m])$"
    )
    pg.show(
        mesh, data=np.exp(m1.real), ax=axes[1, 1],
        cMin=50, cMax=100,
        label=r"$|\rho|~[\Omega m]$"
    )
    pg.show(
        mesh, data=m1.imag * 1000, ax=axes[1, 2], cMap='jet_r',
        label=r"$\phi$ [mrad]",
        cMin=-50, cMax=0,
    )

    axes[1, 0].set_title('This row: Complex inversion')

    for ax in axes.flat:
        ax.set_xlim(-10, 60)
        ax.set_ylim(-20, 0)
        for s in scheme.sensors():
            ax.scatter(s[0], s[1], color='k', s=5)

    fig.tight_layout()
    fig.savefig('test_inv_it_{}.jpg'.format(i + 1), dpi=300)

    m_old = m1
    # Response for Starting model
    m_re_im = pg.utils.squeezeComplex(np.exp(m_old))
    response_re_im = np.array(fop.response(m_re_im))
    response = np.log(pg.utils.toComplex(response_re_im))

    plot_inv_pars('stats_it{}.jpg'.format(i + 1), d, response, Wd)

    J_block = fop.createJacobian(m_re_im)
    J_re = np.array(J_block.matrices()[0])
    J_im = np.array(J_block.matrices()[1])
    J = J_re + 1j * J_im
    J = J / np.exp(response[:, np.newaxis]) * np.exp(m_old)[np.newaxis, :]
