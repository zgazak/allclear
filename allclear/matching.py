"""Triangle-hash blind matching for star identification."""

import numpy as np
from itertools import combinations

from scipy.spatial import KDTree


def build_triangle_hashes(positions, indices=None, n_brightest=40):
    """Build scale/rotation-invariant triangle hashes from 2D positions.

    Parameters
    ----------
    positions : ndarray (N, 2)
        (x, y) positions.
    indices : ndarray (N,), optional
        Original indices of the positions. Defaults to range(N).
    n_brightest : int
        Use only the first n_brightest positions.

    Returns
    -------
    hashes : ndarray (M, 2)
        Hash values (a/c, b/c) for each triangle.
    tri_indices : ndarray (M, 3)
        Indices of the three stars in each triangle.
    """
    n = min(len(positions), n_brightest)
    pos = np.asarray(positions[:n], dtype=np.float64)
    if indices is None:
        indices = np.arange(n)
    else:
        indices = np.asarray(indices[:n])

    hashes = []
    tri_indices = []

    for i, j, k in combinations(range(n), 3):
        # Side lengths
        sides = np.array([
            np.linalg.norm(pos[i] - pos[j]),
            np.linalg.norm(pos[j] - pos[k]),
            np.linalg.norm(pos[i] - pos[k]),
        ])
        # Sort: a <= b <= c
        order = np.argsort(sides)
        a, b, c = sides[order]
        if c < 1e-10:
            continue

        hashes.append([a / c, b / c])

        # Map sorted side indices back to star indices
        idx = [indices[i], indices[j], indices[k]]
        # Reorder triangle vertices to match sorted sides
        tri_verts = [(i, j), (j, k), (i, k)]
        sorted_verts = [tri_verts[order[0]], tri_verts[order[1]], tri_verts[order[2]]]
        tri_indices.append([indices[i], indices[j], indices[k]])

    if not hashes:
        return np.empty((0, 2)), np.empty((0, 3), dtype=int)

    return np.array(hashes), np.array(tri_indices)


def build_hash_index(hashes):
    """Build a KDTree over 2D triangle hash space.

    Parameters
    ----------
    hashes : ndarray (M, 2)
        Triangle hashes.

    Returns
    -------
    KDTree
    """
    return KDTree(hashes)


def match_triangles(det_positions, cat_positions, det_indices=None,
                    cat_indices=None, n_brightest=30, hash_tol=0.02,
                    min_votes=3):
    """Find star correspondences via triangle hash matching.

    Parameters
    ----------
    det_positions : ndarray (N, 2)
        Detected star positions in pixels.
    cat_positions : ndarray (M, 2)
        Catalog star projected positions (initial guess).
    det_indices : ndarray, optional
        Indices into the detection table.
    cat_indices : ndarray, optional
        Indices into the catalog table.
    n_brightest : int
        Number of brightest stars to use.
    hash_tol : float
        Hash space matching tolerance.
    min_votes : int
        Minimum votes for a correspondence to be accepted.

    Returns
    -------
    matches : list of (det_idx, cat_idx)
        Matched pairs.
    """
    if det_indices is None:
        det_indices = np.arange(len(det_positions))
    if cat_indices is None:
        cat_indices = np.arange(len(cat_positions))

    det_hashes, det_tris = build_triangle_hashes(
        det_positions, det_indices, n_brightest
    )
    cat_hashes, cat_tris = build_triangle_hashes(
        cat_positions, cat_indices, n_brightest
    )

    if len(det_hashes) == 0 or len(cat_hashes) == 0:
        return []

    cat_tree = build_hash_index(cat_hashes)

    # Vote matrix: votes[det_idx][cat_idx] = count
    from collections import defaultdict
    votes = defaultdict(lambda: defaultdict(int))

    for i, det_hash in enumerate(det_hashes):
        nearby = cat_tree.query_ball_point(det_hash, hash_tol)
        for j in nearby:
            # Each matching triangle pair votes for 3 possible correspondences.
            # Since we don't know vertex ordering, vote for all possible pairings
            # of the 3 detected vs 3 catalog stars.
            for di in det_tris[i]:
                for ci in cat_tris[j]:
                    votes[int(di)][int(ci)] += 1

    # Extract matches above min_votes
    matches = []
    det_used = set()
    cat_used = set()

    # Sort by vote count
    all_pairs = []
    for di, cv in votes.items():
        for ci, count in cv.items():
            if count >= min_votes:
                all_pairs.append((count, di, ci))
    all_pairs.sort(reverse=True)

    for count, di, ci in all_pairs:
        if di not in det_used and ci not in cat_used:
            matches.append((di, ci))
            det_used.add(di)
            cat_used.add(ci)

    return matches


def match_sources(det_positions, cat_positions, max_dist=10.0):
    """Nearest-neighbor matching given aligned positions.

    Parameters
    ----------
    det_positions : ndarray (N, 2)
        Detected star (x, y) positions.
    cat_positions : ndarray (M, 2)
        Catalog star projected (x, y) positions.
    max_dist : float
        Maximum match distance in pixels.

    Returns
    -------
    matches : list of (det_idx, cat_idx)
        Matched pairs.
    distances : ndarray
        Match distances for each pair.
    """
    if len(det_positions) == 0 or len(cat_positions) == 0:
        return [], np.array([])

    cat_tree = KDTree(np.asarray(cat_positions))
    dists, idxs = cat_tree.query(np.asarray(det_positions))

    matches = []
    match_dists = []
    cat_used = set()

    # Sort by distance to prioritize closer matches
    order = np.argsort(dists)
    for di in order:
        ci = idxs[di]
        if dists[di] <= max_dist and ci not in cat_used:
            matches.append((int(di), int(ci)))
            match_dists.append(dists[di])
            cat_used.add(ci)

    return matches, np.array(match_dists)
