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

# Create a measurement scheme for 51 electrodes, spacing 1
scheme = ert.createERTData(
    elecs=np.linspace(start=0, stop=50, num=51),
    schemeName='dd'
)
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
    80, -0.1 / 1000)
print('Start model', start_model)

###############################################################################
# read-in data
data_rre_rim = np.loadtxt('data_rre_rim.dat')
np.random.seed(42)
noise = np.random.normal(
    loc=0,
    scale=np.abs(data_rre_rim * 0.03)
)
data_rre_rim += noise

# estimate errors
error_estimates = data_rre_rim * 0.01


N = int(error_estimates.size / 2)
Wd = np.diag(error_estimates[0:N] + 1j * error_estimates[N:])
WdTwd = Wd.conj().dot(Wd)
# import IPython
# IPython.embed()

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
# naive inversion

m_old = start_model
J = J0
d = pg.utils.toComplex(data_rre_rim)
response = pg.utils.toComplex(f_0)
lam = 10

term1 = J.conj().T.dot(WdTwd).dot(J) + lam * Wm.T.dot(Wm)
term1_inverse = np.linalg.inv(term1)
term2 = J.conj().T.dot(WdTwd).dot(d - response) - lam * Wm.T.dot(Wm).dot(m_old)
model_update = term1_inverse.dot(term2)
print('Model Update')
print(model_update)

m1 = np.array(m_old + model_update).squeeze()
import matplotlib.pylab as plt
fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 12 / 2.54))
pg.show(mesh, data=m1.real, ax=axes[0])
pg.show(mesh, data=m1.imag, ax=axes[1])
fig.tight_layout()
fig.savefig('test_inv.jpg', dpi=300)

###############################################################################

        # debug on

# Prepare the model parameterization
# We have two markers here: 1: background 2: circle anomaly
# Parameters must be specified as a complex number, here converted by the
# utility function :func:`pygimli.utils.complex.toComplex`.
#rhomap = [
#    [1, pg.utils.complex.toComplex(100, 0 / 1000)],
#    # Magnitude: 100 ohm m, Phase: -50 mrad
#    [2, pg.utils.complex.toComplex(100, -50 / 1000)],
#]

## For visualization, map the rhomap into the actual mesh, resulting in a rho
## vector with a complex resistivity associated with each mesh cell.
#rho = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)
#fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, 16 / 2.54))
#pg.show(mesh, data=np.real(rho), ax=axes[0, 0], label=r"$\rho'~[\Omega m]$")
#pg.show(mesh, data=np.imag(rho), ax=axes[0, 1], label=r"$\rho''~[\Omega m]$")
#pg.show(mesh, data=np.abs(rho), ax=axes[1, 0], label=r"$|\rho|~[\Omega m]$")
#pg.show(
#    mesh, data=np.arctan2(np.imag(rho), np.real(rho)),
#    ax=axes[1, 1], label=r"$\phi$ [mrad]"
#)
#fig.tight_layout()
## fig.show()

################################################################################
## Do the actual forward modeling
#data = ert.simulate(
#    mesh,
#    res=rhomap,
#    scheme=scheme,
#    # noiseAbs=0.0,
#    # noiseLevel=0.0,
#)

################################################################################
## Visualize the modeled data
## Convert magnitude and phase into a complex apparent resistivity
#rho_a_complex = data['rhoa'].array() * np.exp(1j * data['phia'].array())

## Please note the apparent negative phases!
#fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, 16 / 2.54))
#ert.showERTData(data, vals=data['rhoa'], ax=axes[0, 0])
## phia is stored in radians
#ert.showERTData(
#    data, vals=data['phia'] * 1000, label=r'$\phi$ [mrad]', ax=axes[0, 1])

#ert.showERTData(
#    data, vals=np.real(rho_a_complex), ax=axes[1, 0],
#    label=r"$\rho_a'~[\Omega m]$"
#)
#ert.showERTData(
#    data, vals=np.imag(rho_a_complex), ax=axes[1, 1],
#    label=r"$\rho_a''~[\Omega m]$"
#)
#fig.tight_layout()
## fig.show()

################################################################################

