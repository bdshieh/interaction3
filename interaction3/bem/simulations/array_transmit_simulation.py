
import numpy as np
import scipy.sparse.linalg as spsl
from timeit import default_timer as timer
import attr

try:
    import resource
    _RESOURCE_IMPORTED = True
except ImportError:
    _RESOURCE_IMPORTED = False

from .. core . fma_trees import FmaQuadTree


@attr.s
class ArrayTransmitSimulation:

    # M, B, K, Kss, Zr1, G, G1, P1inv

    frequency = attr.ib()
    Gmech = attr.ib(repr=False)
    P = attr.ib(repr=False)
    nodes = attr.ib(repr=False)
    node_area = attr.ib(repr=False)
    sound_speed = attr.ib(repr=False)

    use_preconditioner = attr.ib(default=True, repr=False)
    use_pressure_load = attr.ib(default=False, repr=False)
    Ginv = attr.ib(default=None, repr=False)
    tolerance = attr.ib(default=1.0, repr=False)
    max_iterations = attr.ib(default=100, repr=False)

    wavenumber = attr.ib(init=False)
    _tree = attr.ib(init=False)
    _linear_operator = attr.ib(init=False)
    result = attr.ib(init=False, default=attr.Factory(dict))

    @wavenumber.default
    def _wavenumber_default(self):
        return 2 * np.pi * self.frequency / self.sound_speed

    @_tree.default
    def _tree_default(self):

        f = self.frequency
        nodes = self.nodes
        s_n = self.node_area
        result = self.result

        t0 = timer()
        tree = FmaQuadTree(nodes=nodes, frequency=f, node_area=s_n)
        setup_time = timer() - t0

        result['setup_time'] = setup_time

        return tree

    @_linear_operator.default
    def _linear_operator_default(self):

        f = self.frequency
        s_n = self.node_area
        use_preconditioner = self.use_preconditioner
        tree = self._tree
        Gmech = self.Gmech
        Ginv = self.Ginv

        nnodes = len(self.nodes)

        # define matrix-vector product
        def matvec(x):

            if use_preconditioner:

                y = Ginv.dot(x)
                q = 1j * 2 * np.pi * f * y * s_n
                return Gmech.dot(y) + tree.apply(2 * np.squeeze(q))

            else:

                q = 1j * 2 * np.pi * f * x * s_n
                return Gmech.dot(x) + tree.apply(2 * np.squeeze(q))

        # define LinearOperator
        return spsl.LinearOperator(shape=(nnodes, nnodes), matvec=matvec, dtype=np.complex128)

    class Counter:

        def __init__(self):
            self.count = 0

        def increment(self, *args):
            self.count += 1

    def solve(self):
        '''
        Solve.
        '''
        linear_operator = self.linear_operator
        result = self.result
        Ginv = self.Ginv
        P = self.P
        maxiter = self.max_iterations
        tol = self.tolerance
        use_preconditioner = self.use_preconditioner

        niter = self.Counter()
        t0 = timer()
        y, _ = spsl.lgmres(linear_operator, P, x0=None, M=None, tol=tol, maxiter=maxiter, callback=niter.increment)

        if use_preconditioner:
            x = Ginv.dot(y)
        else:
            x = y
        solve_time = timer() - t0

        # determine max RAM usage
        if _RESOURCE_IMPORTED:

            ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
            result['ram_usage']

        result['x'] = x
        result['niter'] = niter.count
        result['solve_time'] = solve_time


