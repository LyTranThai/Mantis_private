import numpy as np
import torch
import torch.nn.functional as F

def compute_kinematics_numpy(traj_3d, dt=1.0, eps=1e-5):
    """
    Computes kinematics for 3D trajectory data.
    
    Args:
        traj_3d: np.ndarray of shape (N, 3, T) or (3, T)
        dt: time step (default 1.0)
        eps: epsilon to avoid division by zero
        
    Returns:
        dict containing:
            'vel': velocity (N, 3, T)
            'acc': acceleration (N, 3, T)
            'speed': scalar tangential speed (N, 1, T)
            'acc_mag': acceleration magnitude (N, 1, T)
            'curvature': spatial curvature (N, 1, T)
    """
    is_single = False
    if traj_3d.ndim == 2:
        traj_3d = traj_3d[None, ...]
        is_single = True
        
    N, C, T = traj_3d.shape
    assert C == 3, f"Expected 3 channels (X, Y, Z), got {C}"
    
    # Numerical velocity via central difference
    vx = np.gradient(traj_3d[:, 0, :], dt, axis=-1)
    vy = np.gradient(traj_3d[:, 1, :], dt, axis=-1)
    vz = np.gradient(traj_3d[:, 2, :], dt, axis=-1)
    vel = np.stack([vx, vy, vz], axis=1) # (N, 3, T)
    
    # Speed magnitude ||v(t)||
    speed = np.sqrt(vx**2 + vy**2 + vz**2)[:, None, :] # (N, 1, T)
    
    # Numerical acceleration
    ax = np.gradient(vx, dt, axis=-1)
    ay = np.gradient(vy, dt, axis=-1)
    az = np.gradient(vz, dt, axis=-1)
    acc = np.stack([ax, ay, az], axis=1) # (N, 3, T)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)[:, None, :] # (N, 1, T)
    
    # Cross product v x a
    cx = vy * az - vz * ay
    cy = vz * ax - vx * az
    cz = vx * ay - vy * ax
    cross_mag = np.sqrt(cx**2 + cy**2 + cz**2) # (N, T)
    
    # Spatial curvature kappa = ||v x a|| / (||v||^3 + eps)
    denom = (speed[:, 0, :] ** 3) + eps
    curvature = (cross_mag / denom)[:, None, :] # (N, 1, T)
    
    # Clip extreme curvature spikes from near-zero stationary points
    curv_p99 = np.percentile(curvature, 99)
    if curv_p99 > 0:
        curvature = np.clip(curvature, 0, curv_p99 * 3.0)
        
    out = {
        'vel': vel,
        'acc': acc,
        'speed': speed,
        'acc_mag': acc_mag,
        'curvature': curvature
    }
    
    if is_single:
        for k in out:
            out[k] = out[k][0]
            
    return out


def augment_trajectory(traj_3d, config='c7', dt=1.0, eps=1e-5):
    """
    Augments 3D trajectory with kinematic channels.
    
    Args:
        traj_3d: np.ndarray of shape (N, 3, T) or (3, T)
        config: one of:
            'c3': [X, Y, Z] (3 channels)
            'c4': [X, Y, Z, speed] (4 channels)
            'c6': [X, Y, Z, vx, vy, vz] (6 channels)
            'c7': [X, Y, Z, vx, vy, vz, speed] (7 channels)
            'c8': [X, Y, Z, vx, vy, vz, speed, curvature] (8 channels)
            'c10': [X, Y, Z, vx, vy, vz, ax, ay, az, speed] (10 channels)
            
    Returns:
        np.ndarray of shape (N, C_out, T) or (C_out, T)
    """
    is_single = False
    if traj_3d.ndim == 2:
        traj_3d = traj_3d[None, ...]
        is_single = True
        
    config = config.lower()
    if config == 'c3':
        result = traj_3d
    else:
        kin = compute_kinematics_numpy(traj_3d, dt=dt, eps=eps)
        if config == 'c4':
            result = np.concatenate([traj_3d, kin['speed']], axis=1)
        elif config == 'c6':
            result = np.concatenate([traj_3d, kin['vel']], axis=1)
        elif config == 'c7':
            result = np.concatenate([traj_3d, kin['vel'], kin['speed']], axis=1)
        elif config == 'c8':
            result = np.concatenate([traj_3d, kin['vel'], kin['speed'], kin['curvature']], axis=1)
        elif config == 'c10':
            result = np.concatenate([traj_3d, kin['vel'], kin['acc'], kin['speed']], axis=1)
        else:
            raise ValueError(f"Unknown config: {config}. Supported: c3, c4, c6, c7, c8, c10")
            
    if is_single:
        result = result[0]
        
    return result.astype(np.float32)
