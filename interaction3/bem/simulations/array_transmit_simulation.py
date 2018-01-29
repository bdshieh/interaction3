
import numpy as np
# import resource
from scipy.sparse import linalg as spsl
from timeit import default_timer as timer

from . core . trees import QuadTree
# from . base_simulation import BaseSimulation


class ArrayTransmitSimulation:

    # M, B, K, Kss, Zr1, G, G1, P1inv

    def __init__(self, Gmech, P, nodes, node_area, frequency, sound_speed, use_preconditioner=True,
                 use_pressure_load=False, Ginv=None, tol=1.0, maxiter=100, **kwargs):

        s_n = node_area
        f = frequency
        c = sound_speed
        nnodes = len(nodes)

        k = 2 *np.pi * f

        # create fmm quadtree
        quadtree = QuadTree(nodes, wavenumber=k, node_area=s_n)

        # setup quadtree
        t0 = timer()
        quadtree.setup()
        setup_time = timer() - t0

        # define matrix-vector product
        def matvec(x):

            if use_preconditioner:

                y = Ginv.dot(x)
                q = 1j * 2 * np.pi * f * y * s_n
                p1 = quadtree.apply(2 * np.squeeze(q))
                p2 = Gmech.dot(y) + p1

            else:

                q = 1j * 2 * np.pi * f * x * s_n
                p1 = quadtree.apply(2 * np.squeeze(q))
                p2 = Gmech.dot(x) + p1

            return p2

        # define LinearOperator
        linear_operator = spsl.LinearOperator(shape=(nnodes, nnodes), matvec=matvec, dtype=np.complex128)

        options = dict()
        options['use_preconditioner'] = use_preconditioner
        options['tol'] = tol
        options['maxiter'] = maxiter

        bem = dict()
        bem['P'] = P
        if use_preconditioner:
            bem['Ginv'] = Ginv

        result = dict()
        result['setup_time'] = setup_time

        self._quadtree = quadtree
        self._linear_operator = linear_operator
        self._options = options
        self._bem = bem
        self.result = result

    def solve(self):
        '''
        Solve.
        '''
        bem = self._bem
        options = self._options
        result = self.result

        # solve
        linear_operator = self.linear_operator
        Ginv = bem['Ginv']
        P = bem['P']
        maxiter = options['maxiter']
        tol = options['tolerance']
        use_preconditioner = options['use_preconditioner']

        class Counter():

            def __init__(self):
                self.count = 0

            def increment(self, *args):
                self.count += 1

        niter = Counter()
        t0 = timer()
        y, _ = spsl.lgmres(linear_operator, P, x0=None, M=None, tol=tol, maxiter=maxiter, callback=niter.increment)

        if use_preconditioner:
            x = Ginv.dot(y)
        else:
            x = y
        solve_time = timer() - t0

        # calculate membrane and channel averages
        # membranes = transducer.membranes
        # channels = transducer.channels
        #
        # x_mem = np.zeros(len(membranes), dtype=np.complex128)
        # for mem_no in range(len(membranes)):
        #     x_mem[mem_no] = np.mean(x[membranes[mem_no]['nodes_idx']])
        #
        # x_ch = np.zeros(len(channels), dtype=np.complex128)
        # for ch_no in range(len(channels)):
        #     x_ch[ch_no] = np.mean(x_mem[channels[ch_no]['membrane_no']])

        # determine max RAM usage
        # ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.

        result['x'] = x
        # result['x_mem'] = x_mem
        # result['x_ch'] = x_ch
        result['niter'] = niter.count
        result['solve_time'] = solve_time
        # result['ram_usage'] = ram_usage
        result['tolerance'] = tol


