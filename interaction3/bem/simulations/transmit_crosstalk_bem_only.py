## interaction3 / bem / simulations / transmit_crosstalk_bem_only.py

import numpy as np
from scipy import linalg, sparse as sps
from timeit import default_timer as timer
import attr

try:
    import resource
    _RESOURCE_IMPORTED = True
except ImportError:
    _RESOURCE_IMPORTED = False

from interaction3 import abstract
from .. core import bem_functions as bem
from . import sim_functions as sim


@attr.s
class TransmitCrosstalkBemOnly(object):

    ## INSTANCE ATTRIBUTES, INIT ##

    G = attr.ib(repr=False)
    P = attr.ib(repr=False)

    ## INSTANCE ATTRIBUTES, NO INIT ##

    result = attr.ib(init=False, default=attr.Factory(dict), repr=False)

    ## PUBLIC METHODS ##

    def solve(self):
        '''
        Solve.
        '''
        result = self.result
        G = self.G
        P = self.P

        t0 = timer()
        x = linalg.solve(G, P)
        solve_time = timer() - t0

        # determine max RAM usage
        if _RESOURCE_IMPORTED:
            result['ram_usage'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000

        result['displacement'] = x
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
            elif isinstance(obj, abstract.BemSimulation):
                simulation = obj

        return simulation, array

    @staticmethod
    def connector(simulation, array):

        # set simulation defaults
        use_pressure_load = simulation.get('use_pressure_load', False)

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

        # CMUTs only
        if is_cmut:
            Kss_list = list()
            t_ratios_list = list()
            u0_list = list()

        # PMUTs only
        if is_pmut:
            raise NotImplementedError

        for ch in array['channels']:

            dc_bias = ch['dc_bias']
            delay = ch['delay']

            for elem in ch['elements']:
                for mem in elem['membranes']:

                    if isinstance(mem, abstract.SquareCmutMembrane):
                        subc = sim.connector_square_cmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                  dc_bias=dc_bias, use_preconditioner=False)
                    elif isinstance(mem, abstract.CircularCmutMembrane):
                        subc = sim.connector_circular_cmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                    dc_bias=dc_bias, use_preconditioner=False)
                    elif isinstance(mem, abstract.SquarePmutMembrane):
                        subc = sim.connector_square_pmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                  use_preconditioner=False)
                    elif isinstance(mem, abstract.CircularPmutMembrane):
                        subc = sim.connector_circular_pmut_membrane(mem, frequency=f, density=rho, sound_speed=c,
                                                                    use_preconditioner=False)

                    # add general matrices
                    M_list.append(subc['M'])
                    B_list.append(subc['B'])
                    K_list.append(subc['K'])
                    nodes_list.append(subc['nodes'])
                    e_mask_list.append(subc['electrode_mask'])

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

        # form full matrices
        M = sps.block_diag(M_list).todense()
        B = sps.block_diag(B_list).todense()
        K = sps.block_diag(K_list).todense()
        Gmech = sps.block_diag(Gmech_list)
        nodes = np.concatenate(nodes_list)
        e_mask = np.concatenate(e_mask_list)
        P = np.concatenate(P_list)
        membrane_id = np.concatenate(membrane_id_list)
        element_id = np.concatenate(element_id_list)
        channel_id = np.concatenate(channel_id_list)

        if is_cmut:
            Kss = sps.block_diag(Kss_list).todense()
            t_ratios = np.concatenate(t_ratios_list, axis=0)
            u0 = np.concatenate(u0_list, axis=0)

        # form Zr matrix
        Zr = bem.zr_matrix(nodes, f, rho, c, s_n)

        # form G matrix
        G = Gmech + 1j * omega * Zr

        # output required for TransmitCrosstalkBemOnly init arguments
        output = dict()
        output['G'] = G
        output['P'] = P

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
        meta['frequency'] = simulation['frequency']
        meta['density'] = simulation['density']
        meta['sound_speed'] = simulation['sound_speed']
        meta['nodes'] = nodes
        meta['node_area'] = s_n

        if is_cmut:
            meta['Kss'] = Kss
            meta['transformer_ratios'] = t_ratios
            meta['static_displacement'] = u0

        return output, meta