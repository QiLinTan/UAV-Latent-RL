import numpy as np
import torch
import torch.nn.functional as F


KIN_MEAN = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
KIN_STD = np.array([0.5, 0.5, 0.5, np.pi, np.pi, np.pi, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0], dtype=np.float32)

DEFAULT_HOVER_ANGLE_RAD = 0.15
DEFAULT_HOVER_VZ_MPS = 0.25
DEFAULT_DIVE_PITCH_RAD = 0.35
DEFAULT_DIVE_VZ_MPS = -0.60


def posture_labels_np(
    states: np.ndarray,
    hover_angle_rad: float = DEFAULT_HOVER_ANGLE_RAD,
    hover_vz_mps: float = DEFAULT_HOVER_VZ_MPS,
    dive_pitch_rad: float = DEFAULT_DIVE_PITCH_RAD,
    dive_vz_mps: float = DEFAULT_DIVE_VZ_MPS,
) -> list[str]:
    """Classify normalized replay states into coarse flight-posture labels."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 12:
        return ["unknown"] * int(states.shape[0] if states.ndim > 0 else 0)

    roll = states[:, 3] * KIN_STD[3] + KIN_MEAN[3]
    pitch = states[:, 4] * KIN_STD[4] + KIN_MEAN[4]
    vz = states[:, 8] * KIN_STD[8] + KIN_MEAN[8]

    labels = []
    for r, p, v in zip(roll, pitch, vz):
        if abs(r) < hover_angle_rad and abs(p) < hover_angle_rad and abs(v) < hover_vz_mps:
            labels.append("hover")
        elif v < dive_vz_mps or abs(p) > dive_pitch_rad:
            labels.append("dive")
        elif v > abs(dive_vz_mps):
            labels.append("climb")
        elif abs(r) > dive_pitch_rad:
            labels.append("banked")
        else:
            labels.append("cruise")
    return labels


def posture_separation_loss(
    states: torch.Tensor,
    latent: torch.Tensor,
    margin: float = 1.0,
    hover_angle_rad: float = DEFAULT_HOVER_ANGLE_RAD,
    hover_vz_mps: float = DEFAULT_HOVER_VZ_MPS,
    dive_pitch_rad: float = DEFAULT_DIVE_PITCH_RAD,
    dive_vz_mps: float = DEFAULT_DIVE_VZ_MPS,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Encourage hover and dive latent cluster centers to be at least `margin` apart."""
    zero = latent.sum() * 0.0
    if states.ndim != 2 or states.shape[1] < 12 or latent.ndim != 2 or latent.shape[0] == 0:
        return zero, {
            "posture_sep_loss": 0.0,
            "posture_center_distance": 0.0,
            "posture_hover_count": 0.0,
            "posture_dive_count": 0.0,
        }

    roll = states[:, 3] * float(KIN_STD[3]) + float(KIN_MEAN[3])
    pitch = states[:, 4] * float(KIN_STD[4]) + float(KIN_MEAN[4])
    vz = states[:, 8] * float(KIN_STD[8]) + float(KIN_MEAN[8])

    hover_mask = (roll.abs() < hover_angle_rad) & (pitch.abs() < hover_angle_rad) & (vz.abs() < hover_vz_mps)
    dive_mask = (vz < dive_vz_mps) | (pitch.abs() > dive_pitch_rad)
    hover_count = int(hover_mask.sum().detach().item())
    dive_count = int(dive_mask.sum().detach().item())

    if hover_count == 0 or dive_count == 0:
        return zero, {
            "posture_sep_loss": 0.0,
            "posture_center_distance": 0.0,
            "posture_hover_count": float(hover_count),
            "posture_dive_count": float(dive_count),
        }

    latent_norm = F.normalize(latent, dim=1, eps=1e-6)
    hover_center = latent_norm[hover_mask].mean(dim=0)
    dive_center = latent_norm[dive_mask].mean(dim=0)
    center_distance = torch.norm(hover_center - dive_center, p=2)
    loss = F.relu(float(margin) - center_distance).pow(2)

    return loss, {
        "posture_sep_loss": float(loss.detach().item()),
        "posture_center_distance": float(center_distance.detach().item()),
        "posture_hover_count": float(hover_count),
        "posture_dive_count": float(dive_count),
    }
