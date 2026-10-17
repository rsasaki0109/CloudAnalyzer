"""Experimental fixed-match target-normal objective; no solver or state feedback."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

NORMAL_POLICY = dict(neighbors=12, radius_m=1., minimum_neighbors=6,
    minimum_middle_variance_m2=1e-8, minimum_middle_largest_ratio=.1,
    maximum_smallest_middle_ratio=.1, broad_phase_padding_m=1e-10)


def points(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("expected finite (N,3) coordinates")
    return values


def canonical_map(values):
    values = points(values)
    return values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))].copy()


def neighborhoods(map_points, targets, *, workers=4):
    """Radius-limited nearest 12, distance ties resolved by canonical map index."""
    cloud, targets = points(map_points), points(targets)
    if workers not in {1, 4}:
        raise ValueError("use the declared serial or four-worker query")
    selected = np.full((len(targets), NORMAL_POLICY["neighbors"]), -1, dtype=np.int64)
    if not len(targets) or not len(cloud):
        return selected
    policy = NORMAL_POLICY
    candidates = cKDTree(cloud).query_ball_point(targets,
        policy["radius_m"]+policy["broad_phase_padding_m"], workers=workers, return_sorted=True)
    for row, hits in enumerate(candidates):
        ids = np.asarray(hits, dtype=np.int64)
        difference = cloud[ids]-targets[row]
        squared = np.sum(difference*difference, axis=1)
        keep = squared <= policy["radius_m"]**2
        ids, squared = ids[keep], squared[keep]
        order = np.lexsort((ids, squared))[:policy["neighbors"]]
        selected[row, :len(order)] = ids[order]
    return selected


def normal_geometry(map_points, neighbors):
    cloud = points(map_points)
    neighbors = np.asarray(neighbors)
    if neighbors.ndim != 2 or neighbors.shape[1] != NORMAL_POLICY["neighbors"] or neighbors.dtype.kind != "i":
        raise ValueError("invalid neighborhood index array")
    if np.any(neighbors < -1) or np.any(neighbors >= len(cloud)):
        raise ValueError("neighborhood index outside canonical map")
    count = np.sum(neighbors >= 0, axis=1)
    coordinates = np.zeros((*neighbors.shape, 3))
    present = neighbors >= 0
    coordinates[present] = cloud[neighbors[present]]
    centroids = coordinates.sum(axis=1)/np.maximum(count, 1)[:, None]
    centered = np.where(present[:, :, None], coordinates-centroids[:, None], 0.)
    covariance = np.einsum("nki,nkj->nij", centered, centered)/np.maximum(count, 1)[:, None, None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].copy()
    if len(normals):
        sign = np.sign(normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)])
        normals *= sign[:, None]
    policy = NORMAL_POLICY
    labels = np.full(len(neighbors), "accepted", dtype="<U24")
    labels[(eigenvalues[:, 1] <= policy["minimum_middle_variance_m2"]) |
           (eigenvalues[:, 1] < policy["minimum_middle_largest_ratio"]*eigenvalues[:, 2])] = "line_or_point"
    labels[(labels == "accepted") & (eigenvalues[:, 0] > policy["maximum_smallest_middle_ratio"]*eigenvalues[:, 1])] = "not_planar"
    labels[count < policy["minimum_neighbors"]] = "insufficient_neighbors"
    return dict(neighbor_count=count, centroids=centroids, covariance=covariance,
                eigenvalues=eigenvalues, normals=normals, labels=labels, accepted=labels == "accepted")


def spectrum(h, weight_sum, radius):
    if not weight_sum > 0 or radius is None or not radius > 0:
        return None
    d = np.array([1., 1., 1., 1./radius, 1./radius, 1./radius])
    scaled = h*d[:, None]*d[None, :]/weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh((scaled+scaled.T)/2)
    if not eigenvalues[-1] > 0:
        return None
    return dict(eigenvalues=eigenvalues.tolist(), eigenvectors=eigenvectors.tolist(),
                min_max_ratio=float(eigenvalues[0]/eigenvalues[-1]),
                weak_directions_below_1e_3=int(np.count_nonzero(eigenvalues/eigenvalues[-1] < 1e-3)))


def systems(source_world, targets, normals, weights, center):
    """Compare plane/P2P surrogates on exactly the same rows, weights and center."""
    q, p, n = map(points, [source_world, targets, normals])
    w, center = np.asarray(weights, dtype=float), np.asarray(center, dtype=float)
    if (q.shape != p.shape or q.shape != n.shape or w.shape != (len(q),) or center.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(w).all() or np.any(w < 0)
            or not np.allclose(np.linalg.norm(n, axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError("invalid fixed-match inputs, weights or unit normals")
    arm, residual = q-center, q-p
    plane_residual = np.sum(n*residual, axis=1)
    plane_jacobian = np.column_stack([n, np.cross(arm, n)])
    h_plane = plane_jacobian.T@(w[:, None]*plane_jacobian)
    b_plane = -plane_jacobian.T@(w*plane_residual)
    jacobian = np.zeros((len(q), 3, 6))
    jacobian[:, :, :3] = np.eye(3)
    x, y, z = arm.T
    jacobian[:, 0, 4], jacobian[:, 0, 5] = z, -y
    jacobian[:, 1, 3], jacobian[:, 1, 5] = -z, x
    jacobian[:, 2, 3], jacobian[:, 2, 4] = y, -x
    h_point = np.einsum("nri,n,nrj->ij", jacobian, w, jacobian)
    b_point = -np.einsum("nri,n,nr->i", jacobian, w, residual)
    support = float(w.sum())
    radius = float(np.sqrt(np.sum(w*np.sum(arm*arm, axis=1))/support)) if support > 0 else None
    return dict(plane_residual=plane_residual, plane_jacobian=plane_jacobian,
                plane_normal_matrix=h_plane, plane_right_hand_side=b_plane,
                point_normal_matrix=h_point, point_right_hand_side=b_point,
                weight_sum=support, radius_m=radius,
                plane_cost=float(np.sum(w*plane_residual**2)), point_cost=float(np.sum(w*np.sum(residual**2, axis=1))),
                plane_geometry=spectrum(h_plane, support, radius), point_geometry=spectrum(h_point, support, radius))
