
import numpy as np
from scipy import sparse as sps

from . core . bem_functions import *
from interaction3 . abstract import *


def connector(array, simulation):

    rho = simulation['density']
    c = simulation['sound_speed']
    f = simulation['frequency']
    use_preconditioner = simulation['use_preconditioner']
    use_pressure_load = simulation['use_pressure_load']
    tol = simulation['tolerance']
    maxiter = simulation['max_iterations']

    is_cmut = isinstance(array['channels']['membranes'][0], (SquareCmutMembrane, CircularCmutMembrane))
    is_pmut = isinstance(array['channels']['membranes'][0], (SquarePmutMembrane, CircularPmutMembrane))

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

            if isinstance(m, SquareCmutMembrane):
                conn = _connect_square_cmut_membrane(m, simulation, dc_bias=dc_bias)

            elif isinstance(m, CircularCmutMembrane):
                conn = _connect_circular_cmut_membrane(m, simulation, dc_bias=dc_bias)

            elif isinstance(m, SquarePmutMembrane):
                conn = _connect_square_pmut_membrane(m, simulation)

            elif isinstance(m, CircularPmutMembrane):
                conn = _connect_circular_pmut_membrane(m, simulation)

            # add general matrices
            M_list.append(conn['M'])
            B_list.append(conn['B'])
            K_list.append(conn['K'])
            nodes_list.append(conn['nodes'])
            e_mask_list.append(conn['e_mask'])
            if use_preconditioner:
                Zr1_list.append(conn['Zr1'])

            # add CMUT matrices
            if is_cmut:

                Kss_list.append(conn['Kss'])
                t_ratios_list.append(conn['t_ratios'])
                u0_list.append(conn['u0'])

            # determine node excitations
            if use_pressure_load:

                nnodes = len(conn['nodes'])
                P_list.append(np.ones(nnodes) * np.exp(-1j * omega * delay))

            elif is_cmut:

                t_ratios = conn['t_ratios']
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
        G1inv_list = [g1inv_matrix(G1) for G1 in G1_list]

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
    result['nodes'] = nodes
    result['nnodes'] = len(nodes)
    # result['node_area'] = dx * dy
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


def _connect_square_cmut_membrane(obj, simulation, **kwargs):

    f = simulation['frequency']
    rho = simulation['density']
    c = simulation['sound_speed']
    use_preconditioner = simulation['use_preconditioner']

    length_x = obj['length_x']
    length_y = obj['length_y']
    electrode_x = obj['electrode_x']
    electrode_y = obj['electrode_y']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    dc_bias = kwargs['dc_bias']

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = _generate_square_nodes(length_x, length_y, electrode_x, electrode_y,
                                                                          nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:

        filename = obj['k_matrix_comsol_file']
        K = k_matrix_comsol(filename)

    else:

        E = obj['y_modulus']
        eta = obj['p_ratio']
        shape = nnodes_x - 2, nnodes_y - 2
        K = k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # spring-softening matrix
    h_gap = obj['gap']
    h_isol = obj['isolation']
    e_r = obj['permittivity']
    Kss, is_collapsed, u0, t_ratios = kss_matrix(K, e_mask, dc_bias, h_gap, h_isol, e_r, nnodes)
    if is_collapsed:
        raise Exception

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    # mem_no = obj['membrane_no']

    output = dict()

    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    # output['nodes_idx'] = np.arange(mem_no * nnodes, mem_no * nnodes + nnodes)
    output['M'] = M
    output['B'] = B
    output['K'] = K
    output['Kss'] = Kss
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['electrode_mask'] = e_mask
    output['static_displacement'] = u0
    output['transformer_ratios'] = t_ratios

    return output


def _connect_circular_cmut_membrane(obj, simulation, **kwargs):

    f = simulation['frequency']
    rho = simulation['density']
    c = simulation['sound_speed']
    use_preconditioner = simulation['use_preconditioner']

    radius = obj['radius']
    electrode_r = obj['electrode_r']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    dc_bias = kwargs['dc_bias']

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = _generate_square_nodes(radius, electrode_r, nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:

        filename = obj['k_matrix_comsol_file']
        K = k_matrix_comsol(filename)

    else:

        raise Exception # finite-differences not supported for circular membranes yet

    # spring-softening matrix
    h_gap = obj['gap']
    h_isol = obj['isolation']
    e_r = obj['permittivity']
    Kss, is_collapsed, u0, t_ratios = kss_matrix(K, e_mask, dc_bias, h_gap, h_isol, e_r, nnodes)
    if is_collapsed:
        raise Exception

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    # mem_no = obj['membrane_no']

    output = dict()

    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    # output['nodes_idx'] = np.arange(mem_no * nnodes, mem_no * nnodes + nnodes)
    output['M'] = M
    output['B'] = B
    output['K'] = K
    output['Kss'] = Kss
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['electrode_mask'] = e_mask
    output['static_displacement'] = u0
    output['transformer_ratios'] = t_ratios

    return output


def _connect_square_pmut_membrane(obj, simulation, **kwargs):

    f = simulation['frequency']
    rho = simulation['density']
    c = simulation['sound_speed']
    use_preconditioner = simulation['use_preconditioner']

    length_x = obj['length_x']
    length_y = obj['length_y']
    electrode_x = obj['electrode_x']
    electrode_y = obj['electrode_y']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = _generate_square_nodes(length_x, length_y, electrode_x, electrode_y,
                                                                          nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:

        filename = obj['k_matrix_comsol_file']
        K = k_matrix_comsol(filename)

    else:

        E = obj['y_modulus']
        eta = obj['p_ratio']
        shape = nnodes_x - 2, nnodes_y - 2
        K = k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    # piezoelectric actuating load matrix
    filename = obj['peq_matrix_file']
    Peq = peq_matrix(filename)

    # mem_no = obj['membrane_no']

    output = dict()

    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    # output['nodes_idx'] = np.arange(mem_no * nnodes, mem_no * nnodes + nnodes)
    output['M'] = M
    output['B'] = B
    output['K'] = K
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['Peq'] = Peq
    output['electrode_mask'] = e_mask

    return output



def _connect_circular_pmut_membrane(obj, simulation, **kwargs):

    f = simulation['frequency']
    rho = simulation['density']
    c = simulation['sound_speed']
    use_preconditioner = simulation['use_preconditioner']

    radius = obj['radius']
    electrode_r = obj['electrode_r']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = _generate_square_nodes(radius, electrode_r, nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:

        filename = obj['k_matrix_comsol_file']
        K = k_matrix_comsol(filename)

    else:

        E = obj['y_modulus']
        eta = obj['p_ratio']
        shape = nnodes_x - 2, nnodes_y - 2
        K = k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    # piezoelectric actuating load matrix
    filename = obj['peq_matrix_file']
    Peq = peq_matrix(filename)

    # mem_no = obj['membrane_no']

    output = dict()

    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    # output['nodes_idx'] = np.arange(mem_no * nnodes, mem_no * nnodes + nnodes)
    output['M'] = M
    output['B'] = B
    output['K'] = K
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['Peq'] = Peq
    output['electrode_mask'] = e_mask

    return output

def _connect_channel():
    pass


def _connect_defocused_channel():
    pass


## HELPER FUNCTIONS ##

def _generate_square_nodes(length_x, length_y, electrode_x, electrode_y, nnodes_x, nnodes_y):

    # define nodes on x-y plane
    xv = np.linspace(-length_x / 2, length_x / 2, nnodes_x)
    yv = np.linspace(-length_y / 2, length_y / 2, nnodes_y)
    zv = 0.
    x, y, z = np.meshgrid(xv[1:-1], yv[1:-1], zv) # remove boundary nodes
    nodes = np.c_[x.ravel(), y.ravel(), z.ravel()]
    nnodes = len(nodes)

    # calculate node spacing and node area
    dx = length_x / (nnodes_x - 1)
    dy = length_y / (nnodes_y - 1)
    s_n = dx * dy

    # flag whether node is in electrode or not
    x_condition = np.abs(nodes[:, 0]) <= electrode_x / 2.
    y_condition = np.abs(nodes[:, 1]) <= electrode_y / 2.
    e_mask = np.logical_and(x_condition, y_condition)

    return nodes, nnodes, dx, dy, s_n, e_mask


def _generate_circular_nodes(radius, electrode_r, nnodes_x, nnodes_y):

    # define nodes on x-y plane
    length_x = radius * 2
    length_y = radius * 2
    xv = np.linspace(-length_x / 2, length_x / 2, nnodes_x)
    yv = np.linspace(-length_y / 2, length_y / 2, nnodes_y)
    zv = 0
    x, y, z = np.meshgrid(xv[1:-1], yv[1:-1], zv)  # remove boundary nodes
    nodes = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # calculate node spacing and node area
    dx = length_x / (nnodes_x - 1)
    dy = length_y / (nnodes_y - 1)
    s_n = dx * dy

    # keep only nodes inside radius with edge buffer
    edge_buffer = np.sqrt((dx / 2) ** 2 + (dy / 2) ** 2)
    r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    edge_mask = r <= radius - edge_buffer
    nodes = nodes[edge_mask, :]
    nnodes = len(nodes)

    # flag whether node is in electrode or not
    r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    e_mask = r <= electrode_r

    return nodes, nnodes, dx, dy, s_n, e_mask


def rotation_matrix(vec, angle):

    x, y, z = vec
    a = angle

    r = np.zeros((3, 3))
    r[0, 0] = np.cos(a) + x**2 * (1 - np.cos(a))
    r[0, 1] = x * y * (1 - np.cos(a)) - z * np.sin(a)
    r[0, 2] = x * z * (1 - np.cos(a)) + y * np.sin(a)
    r[1, 0] = y * x * (1 - np.cos(a)) + z * np.sin(a)
    r[1, 1] = np.cos(a) + y**2 * (1 - np.cos(a))
    r[1, 2] = y * z * (1 - np.cos(a)) - x * np.sin(a)
    r[2, 0] = z * x * (1 - np.cos(a)) - z * np.sin(a)
    r[2, 1] = z * y * (1 - np.cos(a)) + x * np.sin(a)
    r[2, 2] = np.cos(a) + z**2 * (1 - np.cos(a))

    return r


def rotate_nodes(nodes, vec, angle):

    rmatrix = rotation_matrix(vec, angle)
    return rmatrix.dot(nodes.T).T