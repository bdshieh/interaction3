
import numpy as np
import scipy.sparse as sps
from timeit import default_timer as timer
import attr

try:
    import resource
    _RESOURCE_IMPORTED = True
except ImportError:
    _RESOURCE_IMPORTED = False

from interaction3 import abstract

from .. core . fma_trees import FmaQuadTree
from .. core import bem_functions as bem
from . import _subconnectors as sub


@attr.s
class ArrayTransmitSimulation(object):

    # M, B, K, Kss, Zr1, G, G1, P1inv

    frequency = attr.ib()
    bbox = attr.ib()
    Gmech = attr.ib(repr=False)
    P = attr.ib(repr=False)
    nodes = attr.ib(repr=False)
    node_area = attr.ib(repr=False)
    sound_speed = attr.ib(repr=False)

    use_preconditioner = attr.ib(default=True, repr=False)
    use_pressure_load = attr.ib(default=False, repr=False)
    Ginv = attr.ib(default=None, repr=False)
    max_level = attr.ib(default=6)
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
        return sps.linalg.LinearOperator(shape=(nnodes, nnodes), matvec=matvec, dtype=np.complex128)

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
        y, _ = sps.linalg.lgmres(linear_operator, P, x0=None, M=None, tol=tol, maxiter=maxiter,
                                 callback=niter.increment)

        if use_preconditioner:
            x = Ginv.dot(y)
        else:
            x = y
        solve_time = timer() - t0

        # determine max RAM usage
        if _RESOURCE_IMPORTED:
            result['ram_usage'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000

        result['x'] = x
        result['niter'] = niter.count
        result['solve_time'] = solve_time


def connector(*args):

    if len(args) != 2:
        raise TypeError

    for arg in args:

        if isinstance(arg, abstract.Array):
            array = arg
        if isinstance(arg, abstract.Simulation):
            simulation = arg

    f = simulation['frequency']
    use_preconditioner = simulation['use_preconditioner']
    use_pressure_load = simulation['use_pressure_load']

    is_cmut = isinstance(array['channels']['membranes'][0], (abstract.SquareCmutMembrane,
                                                             abstract.CircularCmutMembrane))
    is_pmut = isinstance(array['channels']['membranes'][0], (abstract.SquarePmutMembrane,
                                                             abstract.CircularPmutMembrane))

    omega = 2 * np.pi * f

    # general
    M_list = list()
    B_list = list()
    K_list = list()
    P_list = list()
    nodes_list = list()
    e_mask_list = list()

    if use_preconditioner:

        G1_list = list()
        Zr1_list = list()
        G1inv_list = list()

    # CMUTs only
    if is_cmut:

        Kss_list = list()
        t_ratios_list = list()
        u0_list = list()

    # PMUTs only
    if is_pmut:
        Peq_list = list()

    for ch in array['channels']:

        dc_bias = ch['dc_bias']
        delay = ch['delay']

        for m in ch['membranes']:

            if isinstance(m, abstract.SquareCmutMembrane):
                subc = sub.connector_square_cmut_membrane(m, simulation, dc_bias=dc_bias)

            elif isinstance(m, abstract.CircularCmutMembrane):
                subc = sub.connector_circular_cmut_membrane(m, simulation, dc_bias=dc_bias)

            elif isinstance(m, abstract.SquarePmutMembrane):
                subc = sub.connector_square_pmut_membrane(m, simulation)

            elif isinstance(m, abstract.CircularPmutMembrane):
                subc = sub.connector_circular_pmut_membrane(m, simulation)

            # add general matrices
            M_list.append(subc['M'])
            B_list.append(subc['B'])
            K_list.append(subc['K'])
            nodes_list.append(subc['nodes'])
            e_mask_list.append(subc['e_mask'])
            if use_preconditioner:
                Zr1_list.append(subc['Zr1'])

            # add CMUT matrices
            if is_cmut:

                Kss_list.append(subc['Kss'])
                t_ratios_list.append(subc['t_ratios'])
                u0_list.append(subc['u0'])

            # determine node excitations
            if use_pressure_load:

                nnodes = len(subc['nodes'])
                P_list.append(np.ones(nnodes) * np.exp(-1j * omega * delay))

            elif is_cmut:

                t_ratios = subc['t_ratios']
                P_list.append(t_ratios * np.exp(-1j * omega * delay))

            elif is_pmut:
                pass

    if is_cmut:
        Gmech_list = [-omega ** 2 * M + 1j * omega * B + K - Kss for M, B, K, Kss in
                      zip(M_list, B_list, K_list, Kss_list)]

    elif is_pmut:
        Gmech_list = [-omega ** 2 * M + 1j * omega * B + K for M, B, K in
                      zip(M_list, B_list, K_list)]

    if use_preconditioner:

        G1_list = [G1 + 1j * omega * Zr1 for G1, Zr1 in zip(Gmech_list, Zr1_list)]
        G1inv_list = [bem.g1inv_matrix(G1) for G1 in G1_list]

    # form full (sparse) matrices
    M = sps.block_diag(M_list, format='csr')
    B = sps.block_diag(B_list, format='csr')
    K = sps.block_diag(K_list, format='csr')
    nodes = np.concatenate(nodes_list, axis=0)
    e_mask = np.concatenate(e_mask_list)
    Gmech = sps.block_diag(Gmech_list, format='csr')
    P = np.concatenate(P_list, format='csr')

    if use_preconditioner:

        G1 = sps.block_diag(G1_list, format='csr')
        G1inv = sps.block_diag(G1inv_list, format='csr')

    if is_cmut:

        Kss = sps.block_diag(Kss_list, format='csr')
        t_ratios = np.concatenate(t_ratios_list, axis=0)
        u0 = np.concatenate(u0_list, axis=0)

    # elif is_pmut:
        # Peq = np.concatenate(Peq_list, axis=0)


    result = dict()
    result.update(simulation)
    result['nodes'] = nodes
    result['nnodes'] = len(nodes)
    result['node_area'] = dx * dy
    result['M'] = M
    result['B'] = B
    result['K'] = K
    result['Gmech'] = Gmech
    result['electrode_mask'] = e_mask
    result['P'] = P

    if is_cmut:

        result['Kss'] = Kss
        result['transformer_ratios'] = t_ratios
        result['static_displacement'] = u0

    if use_preconditioner:

        result['G1inv'] = G1inv
        result['G1'] = G1

    return result