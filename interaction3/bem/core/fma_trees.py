## bem / core / fma_trees.py
'''
Data structures for the multi-level fast multipole algorithm.
Author: Bernard Shieh (bshieh@gatech.edu)
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
import os.path
import h5py
import sys
from ast import literal_eval

import . import fma_functions as fma

# extend recursion limit
sys.setrecursionlimit(30000)


class Group():
    '''
    Group (Box) has a center, type (root, branch or leaf), uid (unique ID) and 
    size; also keeps track of its parent and child Groups.
    '''
    ROOT = 0
    BRANCH = 1
    LEAF = 2
    maxlevel = 1

    def __init__(self, parent, bbox, uid=(0,0,0)):

        # set level
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1

        # set type
        if parent is None:
            self.type = Group.ROOT
        elif self.level == Group.maxlevel:
            self.type = Group.LEAF
        else:
            self.type = Group.BRANCH

        # set uid
        if parent is None:
            self.uid = (0,0,0)
        else:
            self.uid = uid

        # set bbox
        self.bbox = map(float, bbox)

        # set center
        x0, y0, x1, y1 = bbox
        self.center = np.array([x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2, 0.0])

        # set nodes
        self.nodes = None

        # set references
        self.parent = parent
        self.children = [None, None, None, None]

    def __repr__(self):

        types = ('Root', 'Branch', 'Leaf')

        return 'Group <uid: (%d, %d, %d), type: %s>' % (self.uid[0], 
            self.uid[1], self.uid[2], types[self.type])

    def subdivide(self):
        '''
        Create tree by instantiating child Groups recursively.
        '''
        # termination condition
        if self.type == Group.LEAF:
            return

        # calculate childrens' bbox
        x0, y0, x1, y1 = self.bbox
        xmid, ymid = (x1 - x0) / 2, (y1 - y0) / 2

        bbox00 = x0, y0, x0 + xmid, y0 + ymid
        bbox10 = x0 + xmid, y0, x1, y0 + ymid
        bbox01 = x0, y0 + ymid, x0 + xmid, y1
        bbox11 = x0 + xmid, y0 + ymid, x1, y1

        # calculate childrens' uid
        level, xid, yid = self.uid

        uid00 = level + 1, 2 * xid + 0, 2 * yid + 0
        uid10 = level + 1, 2 * xid + 1, 2 * yid + 0
        uid01 = level + 1, 2 * xid + 0, 2 * yid + 1
        uid11 = level + 1, 2 * xid + 1, 2 * yid + 1

        # check if group contains any nodes?
        # spawn children
        self.children[0] = Group(self, bbox00, uid00)
        self.children[0].subdivide() # << recursion!
        self.children[1] = Group(self, bbox10, uid10)
        self.children[1].subdivide() # << recursion!
        self.children[2] = Group(self, bbox01, uid01)
        self.children[2].subdivide() # << recursion!
        self.children[3] = Group(self, bbox11, uid11)
        self.children[3].subdivide() # << recursion!

    def find(self, uid):
        '''
        Returns the Group with the specified uid, or None if not found.
        '''
        if self.uid == uid:
            return self

        if self.level == uid[0]:
            return None

        for child in [c for c in self.children if c is not None]:

            res = child.find(uid)

            if res is not None:
                return res

        return None


class QuadTree():
    '''
    A QuadTree has up to four children per parent box. Used for 2D problems. 
    This is a recursive implementation meant to be simpler and more readable 
    (but probably a bit slower). 
    '''
    _parameters = ['maxlevel', 'wavenumber', 'node_area', 'density', 
        'sound_speed', 'path_to_translation_order_file', 
        'path_to_translation_repository']
    
    def __init__(self, nodes, bounding_box, maxlevel, **kwargs):
        
        # Read in configuration parameters
        config = dict.fromkeys(QuadTree._parameters)

        for k, v in kwargs.items():
            if k in config:
                config[k] = v
                
        config['maxlevel'] = maxlevel
        self.config = config

        # Set tree parameters
        Group.maxlevel = maxlevel
        self.bbox = map(float, bounding_box)
        self.apply_counter = 0

        # Plant and grow full tree
        root = Group(parent=None, bbox=bounding_box)
        root.subdivide()

        # Setup tree and prune
        self.root = root
        self.allgroups = []
        self.leaves = []

        self._traverse(root)
        self._add_nodes(nodes)
        self._prune(root)

        self._find_neighbors(root)

        for group in self.allgroups:
            self._find_ntnn(group)

    def _find(self, uid, group=None):
        '''
        Returns the Group with the specified uid, or None if not found.
        '''
        if group is None:
            group = self.root

        return group.find(uid)

    def _add_nodes(self, nodes):
        '''
        Add nodes to the QuadTree and assigns the nodes to their corresponding
        leaf group.
        '''
        self.nodes = nodes

        x0, y0, x1, y1 = self.root.bbox
        maxid = 2**self.root.maxlevel
        xdim, ydim = (x1 - x0)/maxid, (y1 - y0)/maxid

        # calculate usid of each node: the unique single digit id which 
        # identifies the group it belongs to
        xid = np.floor((nodes[:, 0] - x0) / xdim).astype(np.intc)
        yid = np.floor((nodes[:, 1] - y0) / ydim).astype(np.intc)

        # handle cases where node is on top and right boundaries
        xid[xid == maxid] -= 1
        yid[yid == maxid] -= 1

        usid = xid + maxid * yid

        for group in self.leaves:

            l, xid, yid = group.uid
            group_usid = xid + maxid * yid

            group.node_ids = np.nonzero(usid == group_usid)[0]

            if group.node_ids.size > 0:
                group.nodes = nodes[group.node_ids, :]

    def _prune(self, group):
        '''
        Prunes the QuadTree by removing leaf Groups with no nodes and their 
        corresponding branches. 
        '''
        if group.type == Group.LEAF:

            if group.nodes is None:

                self.allgroups.remove(group)
                self.leaves.remove(group)

                return True

            return False

        res = []
        for idx, child in enumerate(group.children):
            if child is not None:

                if self._prune(child): # << recursion!

                    group.children[idx] = None
                    res.append(True)
                else:

                    res.append(False)
            else:
                res.append(True)

        if all(res):

            self.allgroups.remove(group)
            return True

        return False

    def _traverse(self, group):
        '''
        Traverses the tree in order to add all Groups in the tree to a master
        list (called allgroups).
        '''
        self.allgroups.append(group)

        if group.type == Group.LEAF:
            self.leaves.append(group)

        for child in group.children:
            if child is not None:
                self._traverse(child) # << recursion!

    def _find_neighbors(self, group):
        '''
        Has each group in the tree find its touching neighbors.
        '''
        if group.type == Group.ROOT:
            group.neighbors = []

        if group.level > 0:

            level, xid, yid = group.uid
            maxid = 2**level

            # set uid search range
            istart = max(xid - 1, 0)
            istop = min(xid + 1, maxid)
            jstart = max(yid - 1, 0)
            jstop = min(yid + 1, maxid)

            group.neighbors = []

            if level == 1:

                for i in range(istart, istop + 1):
                    for j in range(jstart, jstop + 1):

                        # skip if uid is itself
                        if i == xid and j == yid:
                            continue

                        res = self._find((level, i, j), group=self.root)

                        if res is not None:
                            group.neighbors.append(res)

            else:

                parent = group.parent

                for i in range(istart, istop + 1):
                    for j in range(jstart, jstop + 1):

                        # skip if uid is itself
                        if i == xid and j == yid:
                            continue

                        for neighbor in (parent.neighbors + [parent,]):
                            res = self._find((level, i, j), group=neighbor)

                            if res is not None:
                                group.neighbors.append(res)
                                break

        for child in group.children:
            if child is not None:
                self._find_neighbors(child) # << recursion!

    def _find_ntnn(self, group):
        '''
        Find the non-touching nearest neighbors for the specified Group
        '''
        parent = group.parent

        group.ntnn = []

        if parent is None:
            return

        for neighbor in parent.neighbors:
            for child in neighbor.children:
                if child is not None:
                    if child not in group.neighbors:
                        group.ntnn.append(child)

    def setup(self):
        '''
        Set the QuadTree up for solving. This function calls the individual
        setup functions in the correct order.
        '''
        leaves = self.leaves
        config = self.config
        
        for k, v in config.items():
            if v is None:
                print(k)
                raise Exception('One or more configuration parameters not set')
                
        # zero apply counter
        self.apply_counter = 0

        # setup fmm (quadrature rule etc.)
        self._setup_fmm()

        # setup translators
        self._setup_translators()

        #setup shifters
        self._setup_shifters()

        # precompute distances and exp part for leaves
        for group in leaves:

            self._calc_self_dist(group)
            self._calc_neighbor_dist(group)
            self._calc_exp_part(group)

    def _setup_fmm(self):
        '''
        Setup quadrature rules and translation operator order.
        '''
        config = self.config
        
        k = config['wavenumber']
        trans_order_filepath = config['path_to_translation_order_file']
        maxlevel = config['maxlevel']
        
        x0, y0, x1, y1 = self.bbox
        xlength, ylength = x1 - x0, y1 - y0

        self.ldata = {}
        # compute far-field angles for each level
        for l in range(2, maxlevel + 1):

            # load translation order file
            with h5py.File(trans_order_filepath, 'r') as root:
                
                ks = root['wavenumbers'][:]
                orders = root[str(l)]['order'][:]
            
            orders_interp_func = interp1d(ks, orders)
            order = int(orders_interp_func(k))
            if order % 2 == 0:
                order += 1

            self.ldata[l] = fma.fft_quadrule(order, order)
            self.ldata[l]['trans_order'] = order
            self.ldata[l]['group_dims'] = xlength / (2**l), ylength / (2**l)

    def _setup_translators(self):
        '''
        Setup translation operators (precalculated) by loading them from a 
        database.
        '''
        config = self.config
        ldata = self.ldata
        
        k = config['wavenumber']
        maxlevel = config['maxlevel']
        filepath = os.path.normpath(config['path_to_translation_repository'])
        
        x0, y0, x1, y1 = self.bbox
        xlength, ylength = x1 - x0, y1 - y0

        # load translations for every level
        self.translators = {}
        
        with h5py.File(filepath, 'r') as root:
            
            for l in range(2, maxlevel + 1):
                
                cache = {}
                key = '{:0.4f}/{:n}'.format(k, l)
                
                for vec, trans in root[key].items():
                    cache[literal_eval(vec)] = trans[:]
                    
                expanded_cache = {}
    
                for vec, translation in cache.items():
                    
                    try:
                        x, y, z = vec
                    except Exception:
                        print(vec)
                        raise Exception
                        
                    ntheta, nphi = translation.shape
    
                    # Quadrant II
                    a = np.flipud(translation)[:, nphi/2:]
                    b = translation[:, :nphi/2]
                    expanded_cache[(-y, x, z)] = np.ascontiguousarray(np.concatenate((a, b), axis=1))
    
                    # Quadrant III
                    expanded_cache[(-x, -y, z)] = np.ascontiguousarray(np.flipud(translation))
    
                    # Quadrant IV
                    a = translation[:, nphi/2:]
                    b = np.flipud(translation)[:, :nphi/2]
                    expanded_cache[(y, -x, z)] =  np.ascontiguousarray(np.concatenate((a, b), axis=1))
    
                cache.update(expanded_cache)
                
                self.translators[l] = cache

        # assign each group's translators
        allgroups = self.allgroups

        for group in allgroups:

            group.translators = []
            l = group.level

            if l < 2:
                continue

            xdim, ydim = ldata[l]['group_dims']

            for fargroup in group.ntnn:

                rx, ry, _ = group.center - fargroup.center

                x = int(round(rx/xdim))
                y = int(round(ry/ydim))
                z = 0

                group.translators.append(self.translators[l][(x, y, z)])

    def _setup_shifters(self):
        '''
        Setup shift operators (calculated here).
        '''
        config = self.config
        ldata = self.ldata
        
        k = config['wavenumber']
        maxlevel = config['maxlevel']

        self.shifters = {}

        for l in range(2, maxlevel + 1):

            xdim, ydim = ldata[l]['group_dims']
            kcoordT = ldata[l]['kcoord'].transpose((0, 2, 1))
            r = np.sqrt(xdim**2 + ydim**2) / 2

            # define direction unit vectors for the four quadrants
            rhat00 = np.array([1, 1, 0]) / np.sqrt(2) # lower left group
            rhat10 = np.array([-1, 1, 0]) / np.sqrt(2) # lower right group
            rhat01 = np.array([1, -1, 0]) / np.sqrt(2) # upper left group
            rhat11 = np.array([-1, -1, 0]) / np.sqrt(2) # upper right group

            # calculate shifters from magnitude and angle
            shift00 = fma.ff2ff_op(r, rhat00.dot(kcoordT), k)
            shift10 = fma.ff2ff_op(r, rhat10.dot(kcoordT), k)
            shift01 = fma.ff2ff_op(r, rhat01.dot(kcoordT), k)
            shift11 = fma.ff2ff_op(r, rhat11.dot(kcoordT), k)

            self.shifters[l] = []
            self.shifters[l].append(shift00)
            self.shifters[l].append(shift10)
            self.shifters[l].append(shift01)
            self.shifters[l].append(shift11)

    def _calc_exp_part(self, group):
        '''
        Calculate the exponential part.
        '''
        config = self.config
        ldata = self.ldata
        
        k = config['wavenumber']
        #k = self.wavenumber

        nodes = group.nodes
        center = group.center
        l = group.level
        kcoord = ldata[l]['kcoord']

        group.exp_part = fma.calc_exp_part(nodes, center, kcoord, k)

    def _calc_self_dist(self, group):
        '''
        Calculates distances between nodes in the specified Group.
        '''
        nodes = group.nodes
        group.self_dist = fma.distance(nodes, nodes)

    def _calc_neighbor_dist(self, group):
        '''
        Calculates distances between nodes in the specified Group and nodes in
        neighbor Groups.
        '''
        nodes = group.nodes
        neighbors = group.neighbors
        group.neighbor_dist = []

        for neighbor in neighbors:
            group.neighbor_dist.append(fma.distance(nodes, neighbor.nodes))

    def apply(self, strengths):
        
        config = self.config
        for v in config.values():
            if v is None:
                raise Exception('One or more configuration parameters not set')
                
        self.apply_counter += 1

        root = self.root
        leaves = self.leaves
        nnodes = self.nodes.shape[0]

        for group in leaves:
            self._calc_coeffs(group, strengths)

        self._uptree(root)
        self._downtree(root)

        # calculate pressures
        pres = np.zeros(nnodes, dtype=np.complex128)

        for group in self.leaves:
            self._calc_pres(group, strengths, pres)

        return pres

    def _calc_coeffs(self, group, strengths):
        '''
        Calculates the far-field coefficients for the specified Group.
        '''
        node_ids = group.node_ids
        exp_part = group.exp_part

        q = np.ascontiguousarray(strengths[node_ids])
        group.coeffs = fma.ff_coeff(q, exp_part)

    def _uptree(self, group):
        '''
        Upward traversal of the QuadTree. Starting with Leaf Groups, far-field
        coefficients are shifted, interpolated and collected by the Parent.
        '''
        # skip leaves
        if group.type == Group.LEAF:
            return

        for child in group.children:
            if child is not None:
                self._uptree(child) # << recursion!

        # skip levels 0 and 1
        if group.level < 2:
            return

        shifters = self.shifters

        ntheta1, nphi1 = shifters[group.level + 1][0].shape
        ntheta2, nphi2 = shifters[group.level][0].shape

        sum_coeffs = np.zeros((ntheta1, nphi1), dtype=np.complex128)

        for child, shifter in zip(group.children, shifters[group.level + 1]):
            if child is not None:
                sum_coeffs += child.coeffs*shifter

        if ntheta2 > ntheta1:
            group.coeffs = fma.fft_interpolate(sum_coeffs, ntheta2, nphi2)
        else:
            group.coeffs = sum_coeffs

    def _downtree(self, group):
        '''
        Downward traversal of the QuadTree. Starting at L2, the coefficients of 
        non-touching neighbor Groups are translated and aggregated. These 
        near-field coefficients are then shifted, filtered, and assigned to
        child Groups.
        '''
        # skip levels 0 and 1
        if group.level > 1:

            if group.level == 2:
                aggr_coeffs = np.zeros_like(group.coeffs, dtype=np.complex128)
            else:
                aggr_coeffs = group.aggr_coeffs

            for fargroup, translator in zip(group.ntnn, group.translators):
                aggr_coeffs += fargroup.coeffs * translator

            # skip this part for leaves
            if group.type != Group.LEAF:

                shifters = self.shifters

                ntheta1, nphi1 = shifters[group.level][0].shape
                ntheta2, nphi2 = shifters[group.level + 1][0].shape

                if ntheta2 < ntheta1:
                    aggr_coeffs = fma.fft_filter(aggr_coeffs, ntheta2, nphi2)

                for child, shifter in zip(group.children, 
                    shifters[group.level + 1]):
                    if child is not None:
                        child.aggr_coeffs = np.conj(shifter) * aggr_coeffs

        for child in group.children:
            if child is not None:
                self._downtree(child) # << recursion!

    def _calc_pres(self, group, strengths, pres):
        '''
        Pressure evaluation. After uptree and downtree traversals, pressure
        is calculated at every node either directly or by evaluation of 
        near-field coefficients.
        '''
        config = self.config
        ldata = self.ldata
        
        k = config['wavenumber']
        maxlevel = config['maxlevel']
        rho = config['density']
        c = config['sound_speed']
        s_n = config['node_area']
        
        node_ids = group.node_ids
        q = np.ascontiguousarray(strengths[node_ids])
        self_dist = group.self_dist
        weight = ldata[maxlevel]['weights'][0, 0]

        # pressure from nodes within group
        pres[node_ids] += fma.direct_eval(q, self_dist, k, rho, c)

        # pressure from neighbor groups
        for neighbor, dist in zip(group.neighbors, group.neighbor_dist):

            q1 = np.ascontiguousarray(strengths[neighbor.node_ids])
            pres[node_ids] += fma.direct_eval(q1, dist, k, rho, c)

        # pressure from far groups
        pres[node_ids] += weight * fma.nf_eval(group.aggr_coeffs, group.exp_part, k, rho, c)

        # self pressures (piston radiation)
        a_eff = np.sqrt(s_n / np.pi)
        pres[node_ids] += rho * c * (0.5 * (k * a_eff)**2 + 1j * 8 / (3 * np.pi) * k * a_eff) / 2 * (q / s_n)

    
    def _draw_group(self, group, ax, **kwargs):
        
        x0, y0, x1, y1 = group.bbox
        xdim, ydim = x1 - x0, y1 - y0
        
        if 'facecolor' not in kwargs: kwargs['facecolor'] = 'none'
            
        ax.add_patch(Rectangle((x0, y0), xdim, ydim, **kwargs))
        
    def draw(self, ax=None, **kwargs):

        if ax is None:
            fig = plt.figure(tight_layout=True)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
            
        self._draw(self.root, ax)
        
        return fig, ax

    def _draw(self, group, ax, **kwargs):

        self._draw_group(group, ax, **kwargs)

        for child in group.children:
            if child is not None:
                self._draw(child, ax, **kwargs)
                  
    def draw_interaction_map(self, uid, ax=None, **kwargs):
        
        if ax is None:
            fig = plt.figure(tight_layout=True)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
            
        group = self._find(uid)
        
        self._draw_interaction_map(group, ax, **kwargs)
        
        return fig, ax
        
    def _draw_interaction_map(self, group, ax, **kwargs):
        
        if group is None:
            return
            
        if group.type == Group.LEAF:
        
            self._draw_group(group, ax, **kwargs)
            
            for neighbor in group.ntnn:
                self._draw_group(neighbor, ax, **kwargs)
        
        else:
            
            for neighbor in group.ntnn:
                self._draw_group(neighbor, ax, **kwargs)
        
        self._draw_interaction_map(group.parent, ax, **kwargs)

