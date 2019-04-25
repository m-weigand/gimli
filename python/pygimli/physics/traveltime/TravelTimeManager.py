#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class for managing first arrival travel time inversions"""
import numpy as np

from matplotlib.collections import LineCollection


import pygimli as pg
from pygimli.frameworks import MeshModelling
from pygimli.manager import MeshMethodManager

from . raplot import drawTravelTimeData, drawVA, showVA


class TravelTimeDijkstraModelling(MeshModelling):
    def __init__(self, **kwargs):
        self.dijkstra = pg.TravelTimeDijkstraModelling()

        super(TravelTimeDijkstraModelling, self).__init__(**kwargs)

        self.jacobian = self.dijkstra.jacobian
        self.createJacobian = self.dijkstra.createJacobian

        self.setJacobian(self.dijkstra.jacobian())

    def regionManagerRef(self):
        # necessary because core dijkstra use its own RM
        return self.dijkstra.regionManagerRef()

    def setMeshPost(self, mesh):
        """
        """
        pg._r("+"*100)
        self.dijkstra.setMesh(mesh)

    def setDataPost(self, data):
        """
        """
        pg._r("*"*100)
        self.dijkstra.setData(data)

    # def createStartModel(self, t):
    #     pg._y('startmodel', t)
    #     s = self.dijkstra.createDefaultStartModel()
    #     pg._r(s)
    #     return s

    def response(self, par):
        return self.dijkstra.response(par)

    def drawModel(self, ax, model, **kwargs):
        kwargs['label'] = pg.unit('vel')
        super(TravelTimeDijkstraModelling, self).drawModel(ax=ax,
                                                           model=model,
                                                           **kwargs)
        return ax

    def drawData(self, ax, data, err=None, **kwargs):
        kwargs['label'] = pg.unit('as')
        return showVA(self.data, vals=data, usePos=False,
                      ax=ax, **kwargs)


class TravelTimeManager(MeshMethodManager):
    """Manager for refraction seismics (traveltime tomography)

    TODO Document main members and use default MethodManager interface
    e.g., self.inv, self.fop, self.paraDomain, self.mesh, self.data
    """

    def __init__(self, **kwargs):
        """Init function with optional data load"""
        self._useFMM = False

        super(TravelTimeManager, self).__init__(**kwargs)

        self._dataToken = 't'
        self.inv.dataTrans = pg.RTransLog()

    def createForwardOperator(self, **kwargs):
        """Create default forward operator for Traveltime modelling.

        Your want your Manager use a special forward operator you can add them
        here on default Dijkstra is used.
        """
        fop = TravelTimeDijkstraModelling(**kwargs)

        return fop

    def simulate(self, mesh, slowness, scheme, **kwargs):
        """
        Simulate an Traveltime measurement.

        Perform the forward task for a given mesh,
        a slowness distribution (per cell) and return data
        (Traveltime) for a measurement scheme.
        This is a static method since it does not interfere with the Managers
        inversion approaches.

        Parameters
        ----------
        mesh : :gimliapi:`GIMLI::Mesh`
            Mesh to calculate for.

        slowness : array(mesh.cellCount()) | array(N, mesh.cellCount())
            slowness distribution for the given mesh cells can be:

            . a single array of len mesh.cellCount()
            . a matrix of N slowness distributions of len mesh.cellCount()
            . a res map as [[marker0, res0], [marker1, res1], ...]

        scheme : :gimliapi:`GIMLI::DataContainer`
            data measurement scheme

        Other parameters
        ----------------
        noisify : add normal distributed noise based on scheme('err')
            IMPLEMENTME

        Returns
        -------
        t : array(N, data.size()) | DataContainer
            The resulting simulated travel time values.
            Either one column array or matrix in case of slowness matrix.
            A DataContainer is return if noisify set to True.

        """

        fop = self.createForwardOperator()

        fop.setData(scheme)
        fop.setMesh(mesh, ignoreRegionManager=True)

        if len(slowness) == mesh.cellCount():
            if max(slowness) > 1.:
                print('Warning: slowness values larger than 1 (' +
                      str(max(slowness)) + ').. assuming that are velocity '
                      'values .. building reciprocity')
                t = fop.response(1./slowness)
            else:
                t = fop.response(slowness)
        else:
            print(mesh)
            print("slowness: ", slowness)
            raise BaseException("Simulate called with wrong slowness array.")

        ret = pg.DataContainer(scheme)
        ret.set('t', t)

        noiseLevel = kwargs.pop('noiseLevel', 0)

        if noiseLevel > 0:
            if not ret.allNonZero('err'):
                ret.set('t', t)
                ret.set('err', pg.physics.Refraction.estimateError(
                        ret, absoluteError=kwargs.pop('noiseAbs', 1e-4),
                        relativeError=noiseLevel))

            if self.verbose:
                print("Data error estimates (min:max) ",
                      min(ret('err')), ":", max(ret('err')))

            t += pg.randn(ret.size()) * ret('err')
            ret.set('t', t)

        if kwargs.pop('returnArray', False):
            return t

        return ret

    def invert(self, data=None, **kwargs):
        """Invert data.

        Parameters
        ----------
        data : pg.DataContainer()
            Data container with at least SensorIndieces 's g' and
            data values 't' (traveltime in ms) and 'err' (absolute error in ms)
        """

        if isinstance(data, pg.DataContainer):
            self.fop.data = data

        if 'mesh' in kwargs:
            self.fop.setMesh(kwargs.pop('mesh'))

        dataVals = self._ensureData(data)
        errVals = self._ensureError(data)

        # startModel = kwargs.pop('startModel', pg.median(dataVals))
        self.fop.setRegionProperties('*', startModel=1/500)

        return self.fw.run(dataVals, errVals, **kwargs)



        self.fop.setData(data)
        mesh = kwargs.pop('mesh', None)
        secNodes = kwargs.pop('secNodes', 3)
        mesh = mesh.createMeshWithSecondaryNodes(secNodes)
        self.setMesh(mesh)

        # mesh = self.fop.regionManager().mesh().createMeshWithSecondaryNodes(secNodes)
        # self.fop.setMeshPost(mesh)

        # startModel = kwargs.pop('startModel', pg.median(dataVals))
        # self.fop.setRegionProperties('*', startModel=startModel)

        slowness = super(MeshMethodManager, self).invert(data=data,
                                                         **kwargs)
        self.fw.model = 1./slowness
        return self.model


    def showRayPaths(self, model=None, complete=False, ax=None, **kwargs):
        """Show model with ray paths for `model` or last model for
        which the last Jacobian was calculated.

        Parameters
        ----------
        model : array
            Velocity model for which to calculate and visualize ray paths (the
            default is model for last Jacobian calculation in self.velocity).
        ax : matplotlib.axes object
            To draw the model and the path into.
        complete : bool [False]
            Draw all shot-receiver combination instead of the used in
            self.data.
        **kwargs : type
            Additional arguments passed to LineCollection (alpha, linewidths,
            color, linestyles).

        Returns
        -------
        ax : matplotlib.axes object
        cb : matplotlib.colorbar object (only if model is provided)

        Examples
        --------
        >>> # No reason to import matplotlib
        >>> import pygimli as pg
        >>> from pygimli.physics import Refraction
        >>> from pygimli.physics.traveltime import createRAData
        >>>
        >>> x, y = 8, 6
        >>> mesh = pg.createGrid(x, y)
        >>> data = createRAData([(0,0)] + [(x, i) for i in range(y)], shotdistance=y+1)
        >>> data.set("t", pg.RVector(data.size(), 1.0))
        >>> rst = Refraction()
        >>> rst.setDataContainer(data)
        Data: Sensors: 7 data: 6
        >>> rst.setMesh(mesh, 5)
        >>> ax, cb = rst.showRayPaths()
        """
        cbar = None

        if model is None:
            model = self.model

        if model is None:
            pg.info("No previous inversion result found and no model given.",
                    "Using homogeneous slowness model.")
            vel = pg.Vector(self.fop.parameterCount(), 1.0)
            self.fop.createJacobian(1./vel)
        else:
            if self.model is not None:
                if not np.allclose(model, self.model):
                    self.fop.createJacobian(1/model)

        ax, cbar = self.showModel(ax=ax, model=model)

        _ = kwargs.setdefault("color", "w")
        _ = kwargs.setdefault("alpha", 0.5)
        _ = kwargs.setdefault("linewidths", 0.8)

        if complete:
            # Due to different numbering scheme of way matrix
            _, shots = np.unique(self.fop.data("s"), return_inverse=True)
            _, receivers = np.unique(self.fop.data("g"), return_inverse=True)

            # Collecting way segments for all shot/receiver combinations
            segs = []
            for s, g in zip(shots, receivers):
                print(s,g)
                wi = self.fop.dijkstra.way(s, g)
                points = self.fop.dijkstra.mesh().positions(withSecNodes=True)[wi]
                segs.append(np.column_stack((pg.x(points), pg.y(points))))
        else:
            for s, g in zip(self.fop.data("s"), self.fop.data("g")):
                print(s,g)
                wi = self.fop.dijkstra.way(s-1, g-1)
                points = self.fop.dijkstra.mesh().positions(withSecNodes=True)[wi]
                segs.append(np.column_stack((pg.x(points), pg.y(points))))

        line_segments = LineCollection(segs, **kwargs)
        ax.add_collection(line_segments)
        return ax, cbar


if __name__ == '__main__':
    pg.wait()
