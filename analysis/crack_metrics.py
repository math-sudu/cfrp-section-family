"""
Crack metrics extraction from AT1 staggered damage results.

Provides:
  - Crack length estimation via alpha=threshold isoline tracing
  - Dissipated fracture energy (approximate)
  - Damaged element statistics
  - Crack path polyline for visualization
"""

import numpy as np
from collections import defaultdict


def crack_length_from_elements(alpha_elem, centroids, T_conn, X, Y,
                               threshold=0.5, min_cluster=3):
    """Estimate crack length by tracing connected damaged elements.

    Builds element adjacency graph, finds connected components of
    elements with alpha > threshold, then measures the extent of
    each component along its principal axis.

    Parameters
    ----------
    alpha_elem : (M,) element damage values
    centroids : (M, 2) element centroids [mm]
    T_conn : (M, 3) triangle connectivity
    X, Y : (N,) nodal coordinates [mm]
    threshold : float, alpha threshold for "cracked"
    min_cluster : int, minimum cluster size to count as a crack

    Returns
    -------
    cracks : list of dict, each with keys:
        'length_mm': crack length in mm
        'centroid': (2,) mean position of the crack cluster
        'elem_indices': array of element indices in this crack
        'path': (K, 2) ordered path points along the crack [mm]
    total_length_mm : float, sum of all crack lengths
    """
    cracked = np.where(alpha_elem > threshold)[0]
    if len(cracked) == 0:
        return [], 0.0

    # Build element adjacency (share an edge = share 2 nodes)
    edge_to_elem = defaultdict(list)
    for ei in cracked:
        tri = T_conn[ei]
        for j in range(3):
            edge = tuple(sorted([int(tri[j]), int(tri[(j + 1) % 3])]))
            edge_to_elem[edge].append(ei)

    adj = defaultdict(set)
    for edge, elems in edge_to_elem.items():
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                adj[elems[i]].add(elems[j])
                adj[elems[j]].add(elems[i])

    # Connected components (BFS)
    visited = set()
    components = []
    for start in cracked:
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        comp = []
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(comp) >= min_cluster:
            components.append(np.array(comp))

    cracks = []
    total = 0.0
    for comp in components:
        pts = centroids[comp]  # (K, 2) mm
        mean_pt = pts.mean(axis=0)

        # Principal axis via PCA
        centered = pts - mean_pt
        if len(centered) < 2:
            length = 0.0
            path = pts
        else:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Project onto major axis
            major = eigvecs[:, -1]
            proj = centered @ major
            length = float(proj.max() - proj.min())
            # Order points along major axis for path
            order = np.argsort(proj)
            path = pts[order]

        cracks.append({
            'length_mm': length,
            'centroid': mean_pt,
            'elem_indices': comp,
            'path': path,
            'n_elements': len(comp),
            'alpha_max': float(alpha_elem[comp].max()),
            'alpha_mean': float(alpha_elem[comp].mean()),
        })
        total += length

    # Sort by length descending
    cracks.sort(key=lambda c: c['length_mm'], reverse=True)
    return cracks, total
