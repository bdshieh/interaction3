## bem / core / fma_trees.py
'''
Data structures for the multi-level fast multipole algorithm.
Author: Bernard Shieh (bshieh@gatech.edu)
'''
import numpy as np
import os.path
import sys
import attr

from . import fma_functions as fma
from . import db_functions as db

# extend recursion limit
sys.setrecursionlimit(30000)


@attr.s
class Group():

    # class attributes
    ROOT = 0
    BRANCH = 1
    LEAF = 2
    MAXLEVEL = 1

    # instance attributes, init
    parent = attr.ib(repr=False)
    bounding_box = attr.ib()
    uid = attr.ib(default=(0, 0, 0))

    # instance attributes, no init
    level = attr.ib(init=False)
    type = attr.ib(init=False)
    center = attr.ib(init=False)
    children = attr.ib(init=False, default=attr.Factory(lambda: [None] * 4), repr=False,)
    nodes = attr.ib(init=False, default=None, repr=False)

    @level.default
    def _level_default(self):

        if self.parent is None:
            return 0
        else:
            return self.parent.level + 1
    
    @type.default
    def _type_default(self):

        if self.parent is None:
            return Group.ROOT
        elif self.level == Group.MAXLEVEL:
            return Group.LEAF
        else:
            return Group.BRANCH

    @center.default
    def _center_default(self):

        x0, y0, x1, y1 = self.bounding_box
        return x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2, 0

    # public methods
    def subdivide(self):
        '''
        Create tree by instantiating child Groups recursively.
        '''
        # termination condition
        if self.type == Group.LEAF:
            return

        # calculate childrens' bbox
        x0, y0, x1, y1 = self.bounding_box
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


@attr.s
class QuadTree(object):

    ## INSTANCE ATTRIBUTES, INIT ##

    nodes = attr.ib(repr=False)
    bounding_box = attr.ib()
    max_level = attr.ib()

    ## INSTANCE ATTRIBUTES, NO INIT ##

    _groups = attr.ib(init=False, default=attr.Factory(list), repr=False)
    _root = attr.ib(init=False, default=None, repr=False)
    _leaves = attr.ib(init=False, default=attr.Factory(list), repr=False)

    def __attrs_post_init__(self):

        # Set tree parameters
        Group.MAXLEVEL = self.max_level

        # Plant and grow full tree
        root = Group(parent=None, bounding_box=self.bounding_box)
        root.subdivide()

        # Setup tree and prune
        self._root = root
        self._traverse(root)
        self._add_nodes(self.nodes)
        self._prune(root)
        self._find_neighbors(root)

        for group in self._groups:
            self._find_ntnn(group)

    ## PRIVATE METHODS ##

    def _find(self, uid, group=None):
        '''
        Returns the Group with the specified uid, or None if not found.
        '''
        if group is None:
            group = self._root

        return group.find(uid)

    def _add_nodes(self, nodes):
        '''
        Add nodes to the QuadTree and assigns the nodes to their corresponding
        leaf group.
        '''
        self.nodes = nodes

        x0, y0, x1, y1 = self._root.bounding_box
        maxid = 2 ** self._root.MAXLEVEL
        xdim, ydim = (x1 - x0) / maxid, (y1 - y0) / maxid

        # calculate usid of each node: the unique single digit id which 
        # identifies the group it belongs to
        xid = np.floor((nodes[:, 0] - x0) / xdim).astype(int)
        yid = np.floor((nodes[:, 1] - y0) / ydim).astype(int)

        # handle cases where node is on top and right boundaries
        xid[xid == maxid] -= 1
        yid[yid == maxid] -= 1

        usid = xid + maxid * yid

        for group in self._leaves:

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

                self._groups.remove(group)
                self._leaves.remove(group)

                return True

            return False

        res = list()
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

            self._groups.remove(group)
            return True

        return False

    def _traverse(self, group):
        '''
        Traverses the tree in order to add all Groups in the tree to a master
        list (called allgroups).
        '''
        self._groups.append(group)

        if group.type == Group.LEAF:
            self._leaves.append(group)

        for child in group.children:
            if child is not None:
                self._traverse(child) # << recursion!

    def _find_neighbors(self, group):
        '''
        Has each group in the tree find its touching neighbors.
        '''
        if group.type == Group.ROOT:
            group.neighbors = list()

        if group.level > 0:

            level, xid, yid = group.uid
            maxid = 2**level

            # set uid search range
            istart = max(xid - 1, 0)
            istop = min(xid + 1, maxid)
            jstart = max(yid - 1, 0)
            jstop = min(yid + 1, maxid)

            group.neighbors = list()

            if level == 1:

                for i in range(istart, istop + 1):
                    for j in range(jstart, jstop + 1):

                        # skip if uid is itself
                        if i == xid and j == yid:
                            continue

                        res = self._find((level, i, j), group=self._root)

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

        group.ntnn = list()

        if parent is None:
            return

        for neighbor in parent.neighbors:
            for child in neighbor.children:
                if child is not None:
                    if child not in group.neighbors:
                        group.ntnn.append(child)


@attr.s
class FmaQuadTree(QuadTree):

    ## INSTANCE ATTRIBUTES, INIT ##

    frequency = attr.ib()
    node_area = attr.ib(repr=False)
    orders_db = attr.ib(converter=os.path.normpath, repr=False)
    translations_db = attr.ib(converter=os.path.normpath, repr=False)
    density = attr.ib(repr=False)
    sound_speed = attr.ib(repr=False)

    ## INSTANCE ATTRIBUTES, NO INIT ##

    wavenumber = attr.ib(init=False, repr=False)
    _apply_counter = attr.ib(init=False, default=0, repr=False)
    _level_data = attr.ib(init=False, default=attr.Factory(dict), repr=False)
    _translators = attr.ib(init=False, default=attr.Factory(dict), repr=False)
    _shifters = attr.ib(init=False, default=attr.Factory(dict), repr=False)

    @wavenumber.default
    def wavenumber_default(self):
        return 2 * np.pi * self.frequency / self.sound_speed
    
    def __attrs_post_init__(self):
        '''
        Set the QuadTree up for solving. This function calls the individual
        setup functions in the correct order.
        '''
        super().__attrs_post_init__()

        self._setup_fma() # setup fma (quadrature rule etc.)
        self._setup_translators() # setup translators
        self._setup_shifters() # setup shifters

        # precompute distances and exp part for leaves
        for group in self._leaves:

            self._calc_self_dist(group)
            self._calc_neighbor_dist(group)
            self._calc_exp_part(group)

    ## PRIVATE METHODS ##

    def _setup_fma(self):
        '''
        Setup quadrature rules and translation operator order.
        '''
        f = self.frequency
        orders_db = self.orders_db
        maxlevel = self.max_level
        x0, y0, x1, y1 = self.bounding_box
        ldata = self._level_data

        xlength, ylength = x1 - x0, y1 - y0

        # compute far-field angles for each level
        for l in range(2, maxlevel + 1):

            order = db.get_order(orders_db, f, l)

            ldata[l] = fma.fft_quadrule(order, order)
            ldata[l]['order'] = order
            ldata[l]['group_dims'] = xlength / (2 ** l), ylength / (2 ** l)

    def _setup_translators(self):
        '''
        Setup translation operators (precalculated) by loading them from a database.
        '''
        f = self.frequency
        max_level = self.max_level
        translations_db = self.translations_db
        translators = self._translators
        groups = self._groups
        ldata = self._level_data
        
        # load translations for every level
        for l in range(2, max_level + 1):

            cache = dict()

            for vec in fma.get_unique_coords():
                cache[tuple(vec)] = db.get_translation(translations_db, f, l, vec)

            expanded_cache = dict()

            for vec, translation in cache.items():

                x, y, z = vec
                ntheta, nphi = translation.shape

                # Quadrant II
                a = np.flipud(translation)[:, nphi // 2:]
                b = translation[:, :nphi // 2]
                expanded_cache[(-y, x, z)] = np.ascontiguousarray(np.concatenate((a, b), axis=1))

                # Quadrant III
                expanded_cache[(-x, -y, z)] = np.ascontiguousarray(np.flipud(translation))

                # Quadrant IV
                a = translation[:, nphi // 2:]
                b = np.flipud(translation)[:, :nphi // 2]
                expanded_cache[(y, -x, z)] =  np.ascontiguousarray(np.concatenate((a, b), axis=1))

            cache.update(expanded_cache)

            translators[l] = cache

        # assign each group's translators
        for group in groups:

            group.translators = list()
            l = group.level

            if l < 2:
                continue

            xdim, ydim = ldata[l]['group_dims']

            for fargroup in group.ntnn:

                rx, ry, _ = [x1 - x2 for x1, x2 in zip(group.center, fargroup.center)]

                x = int(round(rx / xdim))
                y = int(round(ry / ydim))
                z = 0

                group.translators.append(translators[l][(x, y, z)])

    def _setup_shifters(self):
        '''
        Setup shift operators (calculated here).
        '''
        k = self.wavenumber
        max_level = self.max_level
        ldata = self._level_data
        shifters = self._shifters

        for l in range(2, max_level + 1):

            xdim, ydim = ldata[l]['group_dims']
            kcoordT = ldata[l]['kcoordT']
            r = np.sqrt(xdim ** 2 + ydim ** 2) / 2

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

            shifters[l] = list()
            shifters[l].append(shift00)
            shifters[l].append(shift10)
            shifters[l].append(shift01)
            shifters[l].append(shift11)

    def _calc_exp_part(self, group):
        '''
        Calculate the exponential part.
        '''
        k = self.wavenumber
        ldata = self._level_data

        nodes = group.nodes
        center = group.center
        l = group.level
        kcoord = ldata[l]['kcoord']

        group.exp_part = fma.calc_exp_part(nodes, np.array(center), kcoord, k)

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

        shifters = self._shifters

        ntheta1, nphi1 = shifters[group.level + 1][0].shape
        ntheta2, nphi2 = shifters[group.level][0].shape

        sum_coeffs = np.zeros((ntheta1, nphi1), dtype=np.complex128)

        for child, shifter in zip(group.children, shifters[group.level + 1]):
            if child is not None:
                sum_coeffs += child.coeffs * shifter

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

                shifters = self._shifters

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
        k = self.wavenumber
        maxlevel = self.max_level
        rho = self.density
        c = self.sound_speed
        s_n = self.node_area
        ldata = self._level_data
        
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
        pres[node_ids] += rho * c * (0.5 * (k * a_eff) ** 2 + 1j * 8 / (3 * np.pi) * k * a_eff) / 2 * (q / s_n)

    ## PUBLIC METHODS ##

    def apply(self, strengths):

        root = self._root
        leaves = self._leaves
        nnodes = len(self.nodes)

        self._apply_counter += 1

        for group in leaves:
            self._calc_coeffs(group, strengths)

        self._uptree(root)
        self._downtree(root)

        # calculate pressures
        pres = np.zeros(nnodes, dtype=np.complex128)

        for group in self._leaves:
            self._calc_pres(group, strengths, pres)

        return pres