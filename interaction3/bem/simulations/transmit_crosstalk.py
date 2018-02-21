
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
from . import sim_functions as sim


@attr.s
class TransmitCrosstalk(object):

    ## INSTANCE ATTRIBUTES, INIT ##

    nodes = attr.ib(repr=False)
    bounding_box = attr.ib()
    frequency = attr.ib()
    node_area = attr.ib(repr=False)
    orders_db = attr.ib()
    translations_db = attr.ib()
    density = attr.ib()
    sound_speed = attr.ib()
    Gmech = attr.ib(repr=False)
    P = attr.ib(repr=False)

    ## INSTANCE ATTRIBUTES, INIT, OPTIONAL ##

    max_level = attr.ib(default=6)
    Ginv = attr.ib(default=None, repr=False)
    use_preconditioner = attr.ib(default=True)
    use_pressure_load = attr.ib(default=False)
    tolerance = attr.ib(default=0.01)
    max_iterations = attr.ib(default=100)

    ## INSTANCE ATTRIBUTES, NO INIT ##

    wavenumber = attr.ib(init=False, repr=False)
    result = attr.ib(init=False, default=attr.Factory(dict), repr=False)
    _tree = attr.ib(init=False, repr=False)
    _linear_operator = attr.ib(init=False, repr=False)

    @wavenumber.default
    def _wavenumber_default(self):
        return 2 * np.pi * self.frequency / self.sound_speed

    @_tree.default
    def _tree_default(self):

        kwargs = dict()
        kwargs['nodes'] = self.nodes
        kwargs['bounding_box'] = self.bounding_box
        kwargs['max_level'] = self.max_level
        kwargs['frequency'] = self.frequency
        kwargs['node_area']= self.node_area
        kwargs['orders_db'] = self.orders_db
        kwargs['translations_db'] = self.translations_db
        kwargs['density'] = self.density
        kwargs['sound_speed'] = self.sound_speed

        result = self.result

        t0 = timer()
        tree = FmaQuadTree(**kwargs)
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

    ## PUBLIC METHODS ##

    def solve(self):
        '''
        Solve.
        '''
        linear_operator = self._linear_operator
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

        result['displacement'] = x
        result['number_of_iterations'] = niter.count
        result['solve_time'] = solve_time

    @staticmethod
    def get_objects_from_spec(*files):

        spec = list()

        for file in files:
            obj = abstract.load(file)
            if isinstance(obj, list):
                spec += obj
            else:
                spec.append(obj)

        if len(spec) != 2:
            raise Exception

        for obj in spec:
            if isinstance(obj, abstract.Array):
                array = obj
            elif isinstance(obj, abstract.BemArrayTransmitSimulation):
                simulation = obj

        return simulation, array

    @staticmethod
    def connector(simulation, array):

        # set simulation defaults
        use_preconditioner = simulation.get('use_preconditioner', True)
        use_pressure_load = simulation.get('use_pressure_load', False)
        tolerance = simulation.get('tolerance', 0.01)
        max_iterations = simulation.get('max_iterations', 100)
        max_level = simulation.get('max_level', 6)

        f = simulation['frequency']
        rho = simulation['density']
        c = simulation['sound_speed']
        is_cmut = isinstance(array['channels'][0]['elements'][0]['membranes'][0], (abstract.SquareCmutMembrane,
                                                                                   abstract.CircularCmutMembrane))
        is_pmut = isinstance(array['channels'][0]['elements'][0]['membranes'][0], (abstract.SquarePmutMembrane,
                                                                                   abstract.CircularPmutMembrane))
        omega = 2 * np.pi * f

        # General
        M_list = list()
        B_list = list()
        K_list = list()
        P_list = list()
        nodes_list = list()
        e_mask_list = list()
        channel_id_list = list()
        element_id_list = list()
        membrane_id_list = list()

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
            raise NotImplementedError

        for ch in array['channels']:

            dc_bias = ch['dc_bias']
            delay = ch['delay']

            for elem in ch['elements']:
                for mem in elem['membranes']:

                    if isinstance(mem, abstract.SquareCmutMembrane):
                        subc = sim.connector_square_cmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                  dc_bias=dc_bias,
                                                                  use_preconditioner=use_preconditioner)
                    elif isinstance(mem, abstract.CircularCmutMembrane):
                        subc = sim.connector_circular_cmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                    dc_bias=dc_bias,
                                                                    use_preconditioner=use_preconditioner)
                    elif isinstance(mem, abstract.SquarePmutMembrane):
                        subc = sim.connector_square_pmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                  use_preconditioner=use_preconditioner)
                    elif isinstance(mem, abstract.CircularPmutMembrane):
                        subc = sim.connector_circular_pmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                    use_preconditioner=use_preconditioner)

                    # add general matrices
                    M_list.append(subc['M'])
                    B_list.append(subc['B'])
                    K_list.append(subc['K'])
                    nodes_list.append(subc['nodes'])
                    e_mask_list.append(subc['electrode_mask'])
                    if use_preconditioner:
                        Zr1_list.append(subc['Zr1'])

                    nnodes = len(subc['nodes'])
                    membrane_id_list.append(np.ones(nnodes, dtype=int) * mem['id'])
                    element_id_list.append(np.ones(nnodes, dtype=int) * elem['id'])
                    channel_id_list.append(np.ones(nnodes, dtype=int) * ch['id'])

                    # add CMUT matrices
                    if is_cmut:
                        Kss_list.append(subc['Kss'])
                        t_ratios_list.append(subc['transformer_ratios'])
                        u0_list.append(subc['static_displacement'])

                    # determine node excitations
                    if ch['active'] and ch['kind'].lower() in ['tx', 'transmit', 'both', 'txrx']:

                        if use_pressure_load:
                            P_list.append(np.ones(nnodes) * np.exp(-1j * omega * delay))

                        elif is_cmut:
                            t_ratios = subc['transformer_ratios']
                            P_list.append(t_ratios * np.exp(-1j * omega * delay))

                        elif is_pmut:
                            pass
                    else:
                        P_list.append(np.zeros(nnodes))

        s_n = subc['node_area']

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
        nodes = np.concatenate(nodes_list)
        e_mask = np.concatenate(e_mask_list)
        Gmech = sps.block_diag(Gmech_list, format='csr')
        P = np.concatenate(P_list)
        membrane_id = np.concatenate(membrane_id_list)
        element_id = np.concatenate(element_id_list)
        channel_id = np.concatenate(channel_id_list)

        if use_preconditioner:
            G1 = sps.block_diag(G1_list, format='csr')
            G1inv = sps.block_diag(G1inv_list, format='csr')

        if is_cmut:
            Kss = sps.block_diag(Kss_list, format='csr')
            t_ratios = np.concatenate(t_ratios_list, axis=0)
            u0 = np.concatenate(u0_list, axis=0)

        # elif is_pmut:
            # Peq = np.concatenate(Peq_list, axis=0)

        # output required ArrayTransmitSimulation init arguments
        output = dict()
        output['nodes'] = nodes
        output['node_area'] = s_n
        output['Gmech'] = Gmech
        output['P'] = P
        output['frequency'] = simulation['frequency']
        output['density'] = simulation['density']
        output['sound_speed'] = simulation['sound_speed']
        output['bounding_box'] = simulation['bounding_box']
        output['orders_db'] = simulation['orders_db']
        output['translations_db'] = simulation['translations_db']
        output['max_level'] = max_level
        output['use_preconditioner'] = use_preconditioner
        output['use_pressure_load'] = use_pressure_load
        output['tolerance'] = tolerance
        output['max_iterations'] = max_iterations

        # metadata
        meta = dict()
        meta['nnodes'] = len(nodes)
        meta['M'] = M
        meta['B'] = B
        meta['K'] = K
        meta['electrode_mask'] = e_mask
        meta['membrane_id'] = membrane_id
        meta['element_id'] = element_id
        meta['channel_id'] = channel_id

        if is_cmut:
            meta['Kss'] = Kss
            meta['transformer_ratios'] = t_ratios
            meta['static_displacement'] = u0

        if use_preconditioner:
            output['Ginv'] = G1inv
            meta['G1'] = G1

        return output, meta