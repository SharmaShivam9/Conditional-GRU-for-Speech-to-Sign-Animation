import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

bone_min = torch.tensor([
    0.1690, 0.1202, 0.0447, 0.1032, 0.0588,
    0.0077, 0.0077, 0.0022, 0.0160, 0.0085, 0.0067, 0.0012,
    0.0058, 0.0011, 0.0051, 0.0010, 0.0038, 0.0008,
    0.0091, 0.0108, 0.0018, 0.0188, 0.0085, 0.0067, 0.0011,
    0.0051, 0.0011, 0.0043, 0.0010, 0.0033, 0.0008
], dtype=torch.float32, device=DEVICE)

bone_max = torch.tensor([
    0.2053, 0.1752, 0.1681, 0.1723, 0.1751,
    0.0317, 0.0500, 0.0170, 0.0581, 0.0393, 0.0466, 0.0145,
    0.0505, 0.0157, 0.0466, 0.0142, 0.0379, 0.0122,
    0.0323, 0.0513, 0.0173, 0.0588, 0.0391, 0.0467, 0.0145,
    0.0496, 0.0158, 0.0461, 0.0145, 0.0374, 0.0123
], dtype=torch.float32, device=DEVICE)

w_vel = torch.tensor([
    4.1544, 4.0894, 2.4006, 1.9128, 1.1587, 1.0943, 0.9793, 0.8791, 0.7948, 
    0.9270, 0.8152, 0.7539, 0.6997, 0.9284, 0.8163, 0.7660, 0.7152, 0.9213,
    0.8221, 0.7846, 0.7430, 0.9045, 0.8217, 0.7882, 0.7527, 1.0055, 0.9412,
    0.8470, 0.7664, 0.6973, 0.8029, 0.7069, 0.6551, 0.6096, 0.8008, 0.7098,
    0.6744, 0.6350, 0.7951, 0.7141, 0.6896, 0.6593, 0.7852, 0.7171, 0.6948,
    0.6698
], dtype=torch.float32, device=DEVICE)

w_acc = torch.tensor([
    3.8846, 3.8279, 2.5316, 2.2244, 1.1324, 1.0781, 0.9470, 0.8246, 0.7148,
    0.9554, 0.8326, 0.7506, 0.6668, 0.9945, 0.8669, 0.7832, 0.6890, 0.9794,
    0.8628, 0.7962, 0.7145, 0.9172, 0.8163, 0.7621, 0.7002, 1.0498, 0.9724,
    0.8515, 0.7426, 0.6444, 0.8462, 0.7270, 0.6530, 0.5789, 0.8631, 0.7480,
    0.6825, 0.6046, 0.8513, 0.7470, 0.7000, 0.6367, 0.8127, 0.7214, 0.6809,
    0.6327
], dtype=torch.float32, device=DEVICE)


def get_bone_pairs_31():
    """
    Returns the list of (joint_a, joint_b) indices forming each of the 31 bones.
    These correspond to your skeleton structure:
      - 5 body bones (shoulders, arms)
      - 13 left-hand bones
      - 13 right-hand bones
    """
    bone_pairs = []

    # === Upper body (5 bones) ===
    bone_pairs += [(0, 1)]   # shoulder_line
    bone_pairs += [(0, 2)]   # left_upper_arm
    bone_pairs += [(2, 4)]   # left_lower_arm
    bone_pairs += [(1, 3)]   # right_upper_arm
    bone_pairs += [(3, 25)]  # right_lower_arm

    # === Hand bones (left: base index 4, right: base index 25)
    left = 4
    right = 25

    def add_hand_bones(start):
        hand_pairs = []
        # Palm (3 bones)
        hand_pairs += [(start + 0, start + 1)]
        hand_pairs += [(start + 1, start + 5)]
        hand_pairs += [(start + 5, start + 9)]
        # Fingers (2 bones per finger)
        fingers = {
            "thumb": [0, 2, 4],
            "index": [5, 7, 8],
            "middle": [9, 11, 12],
            "ring": [13, 15, 16],
            "pinky": [17, 19, 20],
        }
        for _, (a, b, c) in fingers.items():
            hand_pairs.append((start + a, start + b))
            hand_pairs.append((start + b, start + c))
        return hand_pairs

    # Add both hands
    bone_pairs += add_hand_bones(left)
    bone_pairs += add_hand_bones(right)

    assert len(bone_pairs) == 31, f"Expected 31 bones, got {len(bone_pairs)}"
    return bone_pairs


bone_pairs = get_bone_pairs_31()
_idx_a = torch.tensor([a for (a, _) in bone_pairs], dtype=torch.long, device=DEVICE)
_idx_b = torch.tensor([b for (_, b) in bone_pairs], dtype=torch.long, device=DEVICE)
_eps = 1e-8

def bone_consistency_loss(Y_pred, mask, alpha=1.0, beta=0.5):
    """Fully vectorized bone range + temporal consistency loss."""
    B, L, D = Y_pred.shape
    mask_b = mask[:, :, 0] if mask.dim() == 3 else mask

    pa = torch.stack([Y_pred[:, :, 2 * i:2 * i + 2] for i in _idx_a.tolist()], dim=2)
    pb = torch.stack([Y_pred[:, :, 2 * i:2 * i + 2] for i in _idx_b.tolist()], dim=2)
    la = torch.sqrt(((pa - pb) ** 2).sum(-1) + _eps)

    below_min = (bone_min.view(1, 1, -1) - la).clamp(min=0.0)
    above_max = (la - bone_max.view(1, 1, -1)).clamp(min=0.0)
    range_violation = below_min ** 2 + above_max ** 2
    range_loss = ((range_violation * mask_b.unsqueeze(-1)).sum() / 
                  mask_b.sum().clamp(min=1e-6)).mean()

    if L > 1:
        diff_temp = (la[:, 1:, :] - la[:, :-1, :]) ** 2
        mask_temp = mask_b[:, 1:].unsqueeze(-1)
        temp_loss = ((diff_temp * mask_temp).sum() / 
                     mask_temp.sum().clamp(min=1e-6)).mean()
    else:
        temp_loss = torch.tensor(0.0, device=DEVICE)

    total_loss = alpha * range_loss + beta * temp_loss
    total_loss = total_loss.mean()
    return torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=0.0)


def velocity_loss(Y_pred, Y_target, mask):
    if Y_pred.shape[1] < 2:
        return torch.tensor(0.0, device=DEVICE)
    mask_v = mask[:, 1:, :] if mask.dim() == 3 else mask[:, 1:].unsqueeze(-1)
    vel_pred = Y_pred[:, 1:, :] - Y_pred[:, :-1, :]
    vel_tgt = Y_target[:, 1:, :] - Y_target[:, :-1, :]
    diff_v = (vel_pred - vel_tgt) * mask_v
    w = torch.repeat_interleave(w_vel, 2).view(1, 1, -1)
    loss = ((diff_v ** 2) * w).sum() / mask_v.sum().clamp(min=1e-6)
    return torch.nan_to_num(loss, nan=0.0)

def acceleration_loss(Y_pred, Y_target, mask):
    if Y_pred.shape[1] < 3:
        return torch.tensor(0.0, device=DEVICE)
    mask_a = mask[:, 2:, :] if mask.dim() == 3 else mask[:, 2:].unsqueeze(-1)
    acc_pred = Y_pred[:, 2:, :] - 2 * Y_pred[:, 1:-1, :] + Y_pred[:, :-2, :]
    acc_tgt = Y_target[:, 2:, :] - 2 * Y_target[:, 1:-1, :] + Y_target[:, :-2, :]
    diff_a = (acc_pred - acc_tgt) * mask_a
    w = torch.repeat_interleave(w_acc, 2).view(1, 1, -1)
    loss = ((diff_a ** 2) * w).sum() / mask_a.sum().clamp(min=1e-6)
    return torch.nan_to_num(loss, nan=0.0)

def pose_l2_loss(Y_pred, Y_target, mask):
    mask_e = mask.expand_as(Y_target) if mask.dim() == 3 else mask.unsqueeze(-1).expand_as(Y_target)
    diff = (Y_pred - Y_target) * mask_e
    loss = (diff ** 2).sum() / mask_e.sum().clamp(min=1e-6)
    return torch.nan_to_num(loss, nan=0.0)


def frame_length_loss(Y_pred, mask):
    B, L, D = Y_pred.shape
    if L < 2:
        return torch.tensor(0.0, device=DEVICE)
    vel_pred = Y_pred[:, 1:, :] - Y_pred[:, :-1, :]
    mask_v = mask[:, 1:, :] if mask.dim() == 3 else mask[:, 1:].unsqueeze(-1)
    invalid_v = (1.0 - mask_v)
    vel_mag2 = (vel_pred ** 2).sum(-1, keepdim=True)
    loss_vel = (vel_mag2 * invalid_v).sum() / invalid_v.sum().clamp(min=1e-6)

    if L >= 3:
        acc_pred = Y_pred[:, 2:, :] - 2 * Y_pred[:, 1:-1, :] + Y_pred[:, :-2, :]
        mask_a = mask[:, 2:, :] if mask.dim() == 3 else mask[:, 2:].unsqueeze(-1)
        invalid_a = (1.0 - mask_a)
        acc_mag2 = (acc_pred ** 2).sum(-1, keepdim=True)
        loss_acc = (acc_mag2 * invalid_a).sum() / invalid_a.sum().clamp(min=1e-6)
    else:
        loss_acc = torch.tensor(0.0, device=DEVICE)

    return 0.5 * (loss_vel + loss_acc)


def compute_total_loss(Y_pred, Y_target, mask, alpha_pose=1e3,
                       alpha_vel=0.0, alpha_acc=0.0, alpha_bone=500.0, alpha_len=10.0):
    

    Y_target = Y_target.to(DEVICE)
    mask = mask.to(DEVICE)
    Y_pred = Y_pred.to(DEVICE)

    # 1️ Individual components
    loss_pose = pose_l2_loss(Y_pred, Y_target, mask)*alpha_pose
    #loss_vel  = velocity_loss(Y_pred, Y_target, mask)*alpha_vel
    #loss_acc  = acceleration_loss(Y_pred, Y_target, mask)*alpha_acc
    loss_bone = bone_consistency_loss(Y_pred, Y_target, mask)*alpha_bone
    loss_len  = frame_length_loss(Y_pred, mask)*alpha_len

    # 2️ Weighted combination
    total_loss = (
        loss_pose
        + loss_bone
        + loss_len
    )

    # 3️ Return both total and component losses
    loss_dict = {
        "pose": float(loss_pose.detach().cpu().item()),
        "vel":  0.0,
        "acc":  0.0,
        "bone": float(loss_bone.detach().cpu().item()),
        "len":  float(loss_len.detach().cpu().item()),
        "total": float(total_loss.detach().cpu().item())
    }

    return total_loss,loss_dict