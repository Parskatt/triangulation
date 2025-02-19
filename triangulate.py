import numpy as np
import scipy

def rt_to_pose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def from_hom(X):
    return X[..., :-1] / X[..., -1:]


def to_hom(x):
    return np.concatenate((x, np.ones_like(x[..., -1:])), axis=-1)


def point_cam_towards(x, up):
    # T world to cam
    # x point in cam coords
    x = to_hom(x)
    front = x / np.linalg.norm(x)
    up = up - (front * up).sum()*front
    up = up / np.linalg.norm(up)
    right = -np.cross(front, up)

    R = np.zeros((3, 3))
    R[0] = right
    R[1] = up
    R[2] = front
    T_point = np.eye(4)
    T_point[:3, :3] = R
    return T_point


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def midpoint_triangulation(
    T1: np.ndarray, T2: np.ndarray, point1: np.ndarray, point2: np.ndarray
):
    not_batched = len(point1.shape) == 1
    point1 = to_hom(np.atleast_2d(point1))
    point2 = to_hom(np.atleast_2d(point2))

    R1 = T1[:3, :3]  # 3x3 rotation matrix
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]  # 3x1 translation vector
    t2 = T2[:3, 3]

    # Calculate camera centers (single position for each camera)
    C1 = -R1.T @ t1  # (3,)
    C2 = -R2.T @ t2  # (3,)

    # # Transform view vectors from local to world coordinates
    # # World vector = R * local_vector

    v1s_world = normalize(point1 @ R1)  # (N x 3)
    v2s_world = normalize(point2 @ R2)  # (N x 3)

    # # Vector between camera centers (broadcast to match number of points)
    b = C2 - C1  # (3,)
    num_points = point1.shape[0]
    bs = np.broadcast_to(b[None], (num_points, 3))  # (N x 3)

    # Compute direction vectors between closest points on rays
    cross1 = np.cross(v1s_world, v2s_world)  # N x 3
    cross2 = np.cross(bs, v2s_world)  # N x 3

    # Calculate parameters using cross products
    denom = np.sum(cross1 * cross1, axis=1)
    s = np.sum(cross2 * cross1, axis=1) / denom
    t = np.sum(np.cross(bs, v1s_world) * cross1, axis=1) / denom

    # Find points on each ray in world coordinates
    P1s = C1[None] + s[:, None] * v1s_world  # (N x 3)
    P2s = C2[None] + t[:, None] * v2s_world  # (N x 3)

    # For parallel rays, use camera midpoints
    # midpoint = (C1 + C2) / 2
    # midpoints = midpoint.unsqueeze(0).expand(num_points, -1)
    midpoint = (P1s + P2s) / 2
    if not_batched:
        return midpoint[0]
    else:
        return midpoint


def lindstrom_find_image_observations(E, point1, point2):
    # Thanks to Georg Bökman for c++ implementation, ported by Claude to python (and somewhat checked for correctness by me)
    # Convert 2D points to homogeneous coordinates (3,) arrays
    point1_homogeneous = np.append(point1, 1)
    point2_homogeneous = np.append(point2, 1)

    # Extract 2x2 block from essential matrix
    E_tilde = E[:2, :2]

    # Calculate epipolar lines
    # n2 = E_tilde @ point1 + E[:2, 2]
    n2 = E_tilde @ point1 + E[:2, 2]
    # n1 = E_tilde.T @ point2 + E[2, :2].T
    n1 = E_tilde.T @ point2 + E[2:3, :2].T.flatten()

    # Calculate intermediate values
    a = n2.T @ E_tilde @ n1
    b = (np.sum(n1**2) + np.sum(n2**2)) / 2.0
    c = point2_homogeneous.T @ E @ point1_homogeneous
    d = np.sqrt(b * b - a * c)
    if np.isnan(d):
        d = 0.
        # print("hej")

    # Calculate lambda
    lambda_val = c / (b + d)

    # Calculate deltas
    delta1 = lambda_val * n1
    delta2 = lambda_val * n2

    # Update n1 and n2
    n2 = n2 - E_tilde @ delta1
    n1 = n1 - E_tilde.T @ delta2

    # Update lambda
    lambda_val *= (2.0 * d) / (np.sum(n1**2) + np.sum(n2**2))

    # Calculate optimal points
    optimal_point1 = point1 - lambda_val * n1
    optimal_point2 = point2 - lambda_val * n2

    return optimal_point1, optimal_point2


def skew(t):
    t = t.flatten()
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def essential_from_pose(R, t):
    return skew(t) @ R


def random_inplane_rot():
    theta = 2 * np.pi * np.random.rand()
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

def random_rotation(n):
    Q = np.linalg.qr(np.random.randn(n,n))[0]
    Q[0] *= np.linalg.det(Q)
    return Q

def small_random_rotation(n, scale=0.1):
    # Generate random skew-symmetric matrix
    A = np.random.randn(n, n) * scale
    A = (A - A.T) / 2  # Make it skew-symmetric
    
    # Matrix exponential gives rotation matrix
    return scipy.linalg.expm(A)


if __name__ == "__main__":
    up = np.array([0, 1., 0])
    sigma_position = .01
    sigma_obs = .0
    sigma_rot = .01 # 0.1 rads ish 
    N = 100
    T = 100
    fractions = np.zeros((N,T))
    dists_niter2 = np.zeros((N,T))
    dists_niter2_norm = np.zeros((N,T))
    dists_midpoint = np.zeros((N,T))
    dists_fused = np.zeros((N,T))

    preds = np.zeros((N, T, 3, 3))

    for n in range(N):
        c_A = np.random.randn(3)  # world position
        R_A = random_rotation(3)  # cam to world rotation
        T_A = np.linalg.inv(rt_to_pose(R_A, c_A))

        R_B = random_rotation(3)  # cam to world rotation

        c_B = np.random.randn(3)  # world position

        T_B = np.linalg.inv(rt_to_pose(R_B, c_B))
        relpose = T_B @ np.linalg.inv(T_A)


        X = np.array([0, 0, 0.0, 1.0])
        x_A = from_hom(from_hom(T_A @ X))
        x_B = from_hom(from_hom(T_B @ X))

        for i in range(T):
            obs_A = x_A + sigma_obs * np.random.randn(2)
            obs_B = x_B + sigma_obs * np.random.randn(2)
            # E = essential_from_pose(relpose[:3, :3], relpose[:3, 3])
            T_noisy_A = np.linalg.inv(
                rt_to_pose(small_random_rotation(3, sigma_rot)@R_A, c_A + sigma_position * np.random.randn(3))
            )
            T_noisy_B = np.linalg.inv(
                rt_to_pose(small_random_rotation(3, sigma_rot)@R_B, c_B + sigma_position * np.random.randn(3))
            )
            point_transform_A = point_cam_towards(obs_A, up)
            point_transform_B = point_cam_towards(obs_B, up)
            T_norm_noisy_A = point_transform_A @ T_noisy_A
            T_norm_noisy_B = point_transform_B @ T_noisy_B


            # no align
            X_midpoint_pred = midpoint_triangulation(T_noisy_A, T_noisy_B, obs_A, obs_B)
            relpose_noisy = T_noisy_B @ np.linalg.inv(T_noisy_A)
            E_noisy = essential_from_pose(relpose_noisy[:3, :3], relpose_noisy[:3, 3])
            niter2_pred_A, niter2_pred_B = lindstrom_find_image_observations(E_noisy, obs_A, obs_B)
            X_niter2_pred = midpoint_triangulation(T_noisy_A, T_noisy_B, niter2_pred_A, niter2_pred_B)


            # principal align
            principal_point = np.zeros(2)
            # X_midpoint_pred = midpoint_triangulation(T_noisy_norm_A, T_noisy_norm_B, principal_point, principal_point)
            relpose_norm_noisy = T_norm_noisy_B @ np.linalg.inv(T_norm_noisy_A)
            E_norm_noisy = essential_from_pose(relpose_norm_noisy[:3, :3], relpose_norm_noisy[:3, 3])
            niter2_pred_A, niter2_pred_B = lindstrom_find_image_observations(E_norm_noisy, principal_point, principal_point)
            X_niter2_norm_pred = midpoint_triangulation(T_norm_noisy_A, T_norm_noisy_B, niter2_pred_A, niter2_pred_B)
            
            dist_niter2 = np.linalg.norm(X_niter2_pred-from_hom(X))
            dist_midpoint = np.linalg.norm(X_midpoint_pred-from_hom(X))
            dist_fused = np.linalg.norm((X_midpoint_pred+X_niter2_norm_pred)/2-from_hom(X))
            
            dist_niter2_norm = np.linalg.norm(X_niter2_norm_pred-from_hom(X))

            dists_niter2[n,i] = dist_niter2
            dists_niter2_norm[n,i] = dist_niter2_norm
            dists_midpoint[n,i] = dist_midpoint
            dists_fused[n,i] = dist_fused

            preds[n,i] = np.stack((X_niter2_pred, X_niter2_norm_pred, X_midpoint_pred), axis = -1) 
    corr = np.einsum("tndu, tndv -> uv", preds, preds)
    print(corr)
    print("niter2 vs niter2 norm vs midpoint", dists_niter2.mean(), dists_niter2_norm.mean(), dists_midpoint.mean(), dists_fused.mean())
    print("niter2 vs niter2 norm vs midpoint", np.median(dists_niter2), np.median(dists_niter2_norm), np.median(dists_midpoint), np.median(dists_fused))
