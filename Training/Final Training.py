import os, time, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import math
from make_gif import generate_stick_figure_gif_from_array
from PoseSeqModel import PoseSeqModel
from Loss_func import (compute_total_loss)

# ---------- PATHS ----------
train_pose_dir = r"D:\Des646\AI\data_split\train\pose"
train_sent_dir = r"D:\Des646\AI\data_split\train\sentences"
test_pose_dir = r"D:\Des646\AI\data_split\test\pose"
test_sent_dir = r"D:\Des646\AI\data_split\test\sentences"
CHECKPOINT_DIR = r"D:\Des646\AI\checkpoints"
GIF_DIR = r"D:\Des646\AI\mid-training-gifs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pt")

# ---------- CONFIG ----------
OUT_DIM = 92          # per frame features
MAX_FRAMES = None     # or set e.g. 120 to cap very long sequences
EPOCHS = 400
LR = 1e-4             # Good for RNNs
ALPHA = 1e-4
HIDDEN = 512
PRINT_EVERY = 25
BATCH_SIZE = 128
TF_DECAY_K = 5.0

# ---------- DEVICE ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")
torch.backends.cudnn.benchmark = True

# ---------- LOAD MATCHING FILES ----------
pose_uids = {os.path.splitext(f)[0] for f in os.listdir(train_pose_dir) if f.endswith(".npy")}
sent_uids = {os.path.splitext(f)[0] for f in os.listdir(train_sent_dir) if f.endswith(".npy")}
common_uids = sorted(list(pose_uids & sent_uids))
print(f"Found {len(common_uids)} paired samples in training data.")

X_list, Y_list = [], []

for uid in common_uids:
    sent_path = os.path.join(train_sent_dir, f"{uid}.npy")
    pose_path = os.path.join(train_pose_dir, f"{uid}.npy")

    # Load embedding and pose
    x = np.load(sent_path).astype("float32")          # (512,)
    y = np.load(pose_path).astype("float32")          # (frames, 92)

    # Optional: trim or pad sequences
    if MAX_FRAMES is not None and y.shape[0] > MAX_FRAMES:
        y = y[:MAX_FRAMES]

    X_list.append(x)
    Y_list.append(y)

print(f" Loaded {len(X_list)} samples.")



def load_split(pose_dir, sent_dir):
    pose_uids = {os.path.splitext(f)[0] for f in os.listdir(pose_dir) if f.endswith(".npy")}
    sent_uids = {os.path.splitext(f)[0] for f in os.listdir(sent_dir) if f.endswith(".npy")}
    common_uids = sorted(list(pose_uids & sent_uids))
    Xs, Ys = [], []
    for uid in common_uids:
        Xs.append(np.load(os.path.join(sent_dir, f"{uid}.npy")).astype("float32"))
        Ys.append(np.load(os.path.join(pose_dir, f"{uid}.npy")).astype("float32"))
    return Xs, Ys

X_test_list, Y_test_list = load_split(test_pose_dir, test_sent_dir)
print(f"Loaded test samples: {len(X_test_list)}")

# ---------------- EVALUATION AFTER TRAINING ----------------
def evaluate_model(model, X_list, Y_list, Lmax): # Note: Lmax is added
    model.eval()
    total_comps = {"pose": 0, "vel": 0, "acc": 0, "bone": 0, "len": 0, "total": 0}
    count = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for X, Y_true in zip(X_list, Y_list): # Y_true is the unpadded numpy array
            
            # 1. PREPARE INPUT
            Xb = torch.tensor(X, dtype=torch.float32, device=DEVICE).unsqueeze(0) # (1, Ein)
            
            # 2. PREPARE TARGETS (Pad to Lmax)
            L_true = Y_true.shape[0]
            Yb_pad = torch.zeros((1, Lmax, OUT_DIM), dtype=torch.float32, device=DEVICE)
            Yb_pad[0, :L_true, :] = torch.tensor(Y_true, dtype=torch.float32, device=DEVICE)
            
            # Fill padding with last frame
            if L_true > 0 and L_true < Lmax:
                last_frame = torch.tensor(Y_true[-1, :], dtype=torch.float32, device=DEVICE)
                Yb_pad[0, L_true:, :] = last_frame

            Mb = torch.zeros((1, Lmax, 1), dtype=torch.float32, device=DEVICE)
            Mb[0, :L_true, 0] = 1.0

            # 3. GET PREDICTION (Full length, no cropping)
            es= time.time()
            Ypred = model(Xb, Lmax, target_sequence=None)
            en= time.time()
            print(f" Inference time for one sample: {en - es:.4f}s")
            # 4. COMPUTE LOSS (Using full padded tensors)
            total_loss,comps = compute_total_loss(
                Ypred, Yb_pad, Mb
            )
            
            for k in total_comps:
                total_comps[k] += comps[k]
            count += 1

    avg_comps = {k: v / count for k, v in total_comps.items()}
    print("\n TEST SET LOSS BREAKDOWN") 
    print(f" Total: {avg_comps['total']:.6f}")
    print(f"pose={avg_comps['pose']:.6f}, vel={avg_comps['vel']:.6f}, acc={avg_comps['acc']:.6f}, bone={avg_comps['bone']:.6f}, len={avg_comps['len']:.6f}")
    return avg_comps

def create_gif(model, ep, X_list, X_test_list, Lmax, OUT_DIM, DEVICE):
    """
    Generates one train and one test GIF directly from Y_pred arrays.
    """
    model.eval()

    train_idx = random.randint(0, len(X_list) - 1)
    test_idx = random.randint(0, len(X_test_list) - 1)

    with torch.no_grad():
        X_train_sample = torch.tensor(X_list[train_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        X_test_sample = torch.tensor(X_test_list[test_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        Ypred_train_flat = model(X_train_sample, Lmax)
        Ypred_test_flat = model(X_test_sample, Lmax)

        Ypred_train = Ypred_train_flat.cpu().numpy()[0]
        Ypred_test = Ypred_test_flat.cpu().numpy()[0]

    train_gif_path = os.path.join(GIF_DIR, f"train_epoch{ep}.gif")
    test_gif_path = os.path.join(GIF_DIR, f"test_epoch{ep}.gif")


    generate_stick_figure_gif_from_array(Ypred_train, train_gif_path, fps=10)
    generate_stick_figure_gif_from_array(Ypred_test, test_gif_path, fps=10)


# ---------------- PREPARE DATA ----------------
OUT_DIM = 92
Lmax = max(y.shape[0] for y in Y_list)

X = np.stack(X_list).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)

all_poses = np.concatenate(Y_list, axis=0)

Y_list = [y.astype(np.float32) for y in Y_list]

lengths = np.array([y.shape[0] for y in Y_list])
Y_pad = np.zeros((len(Y_list), Lmax, OUT_DIM), np.float32)
mask = np.zeros((len(Y_list), Lmax, 1), np.float32)

for i, y in enumerate(Y_list):
    L = y.shape[0]
    Y_pad[i, :L, :] = y
    mask[i, :L, 0] = 1.0
    if L > 0 and L < Lmax:
        last_frame = y[-1, :] # Get the last frame (shape [92])
        Y_pad[i, L:, :] = last_frame # Broadcast this frame to the entire padded area

X = torch.tensor(X, dtype=torch.float32)
Y_pad = torch.tensor(Y_pad, dtype=torch.float32)
mask = torch.tensor(mask, dtype=torch.float32)
dataset = TensorDataset(X, Y_pad, mask)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
print(f"Data: total={len(dataset)}, train={len(dataset)}, Ein={X.shape[1]}, Lmax={Lmax}, OUT_DIM={OUT_DIM}")

# ---------------- MODEL ----------------
Ein = X.shape[1]
N_GRU_LAYERS = 2 # You can tune this (1, 2, or 3)

model = PoseSeqModel(
    sentence_embed_dim=Ein,
    hidden_dim=HIDDEN,
    pose_dim=OUT_DIM,
    n_gru_layers=N_GRU_LAYERS,
    dropout=0.2
).to(DEVICE)

print(f"Loaded PoseSeqModel with {N_GRU_LAYERS} GRU layers.")


optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=ALPHA)



# ---------------- TRAINING LOOP ----------------
best_loss = float("inf")
patience, no_improve = 50, 0

print(f"\n Starting training on {DEVICE} for {EPOCHS} epochs...\n")

for ep in range(1, EPOCHS + 1):

    epoch_start_time = time.time()

    progress = (ep - 1) / (EPOCHS - 1)
    sigmoid_input = TF_DECAY_K * (1.0 - 2.0 * progress)
    current_ratio = 1.0 / (1.0 + math.exp(-sigmoid_input))

    model.train()
    train_loss_total = 0.0
    train_comps_sum = {"pose": 0, "vel": 0, "acc": 0, "bone": 0, "len": 0, "total": 0}

    for i,(Xb, Yb, Mb) in enumerate(train_loader):
        Xb, Yb, Mb = Xb.to(DEVICE), Yb.to(DEVICE), Mb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        Ypred = model(Xb, Lmax, target_sequence=Yb, teacher_forcing_ratio=current_ratio)
        # Compute all losses (returns total + components)
        total_loss, comps = compute_total_loss(
            Ypred, Yb, Mb )

        if torch.isnan(total_loss):
            print(f" NaN detected at epoch {ep}, skipping batch.")
            continue
    

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss_total += total_loss.item() * Xb.size(0)
        for k in train_comps_sum:
            train_comps_sum[k] += comps[k] * Xb.size(0)


    # Average train losses
    avg_train_loss = train_loss_total / len(dataset)
    avg_train_comps = {k: v / len(dataset) for k, v in train_comps_sum.items()}

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # ---------------- LOGGING ----------------
    print(f"\n Epoch {ep:03d}/{EPOCHS} | Time: {epoch_duration:.2f}s | TF Ratio: {current_ratio:.2f}")
    print(f"Train Total Loss: {avg_train_comps['total']:.6f}")
    print(f"pose={avg_train_comps['pose']:.6f}, vel={avg_train_comps['vel']:.6f}, acc={avg_train_comps['acc']:.6f}, bone={avg_train_comps['bone']:.6f}, len={avg_train_comps['len']:.6f}")
    # ---------------- CHECKPOINTS ----------------
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    if ep % 50 == 0:
        e1 = time.time()
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{ep}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        e2 = time.time()
        print(f" Saved checkpoint: {checkpoint_path}. Checkpoint saved in {e2 - e1:.2f}s.")

        e3=time.time()
        model.eval()
        
        evaluate_model(model, X_test_list, Y_test_list, Lmax)
        e4=time.time()
        print(f" Evaluate Model took {e4 - e3:.2f}s")

        e5=time.time()
        create_gif(model, ep, X_list, X_test_list, Lmax, OUT_DIM, DEVICE)
        e6=time.time()
        print(f" GIF generation took {e6 - e5:.2f}s")

    if ep % PRINT_EVERY == 0 or ep == 1:
        print(f" Completed {ep}/{EPOCHS} epochs | Best training loss so far = {best_loss:.6f}")

print("\n Training complete!")
print(f"Best training loss achieved: {best_loss:.6f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")

