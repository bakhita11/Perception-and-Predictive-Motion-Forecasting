Code                   https://colab.research.google.com/drive/1BDqBq9uqXyCdSF3c7Ba7A7CCQ7VjhvI_#scrollTo=121b1af8

# Perception-and-Predictive-Motion-Forecasting
This repository contains the official PyTorch implementation of Cross-Modal Adaptive Attention (CMAA), , a unified end-to-end architecture for multimodal autonomous driving perception and motion forecasting. MAA dynamically fuses camera, LiDAR, and RADAR inputs using learnable modality gating and bidirectional cross-modal attention.
 





"""
CMAA SYSTEM - Cross-Modal Adaptive Attention
 End-to-End nuScenes-like demo + robust metrics + visuals
 GT shape bug + fixes NDS_proxy/NaN behavior when TP=0
 """

# -----------------------------
# IMPORTS (core)
# -----------------------------
import torch  # PyTorch tensors + autograd
import torch.nn as nn  # Neural network layers
import torch.nn.functional as F  # Functional ops (softmax/sigmoid/pooling)
import matplotlib.pyplot as plt  # Plotting for figures
import numpy as np  # Numeric utilities for plotting
import math  # Math constants (pi) and helpers
import os  # Directory/file operations

# -----------------------------
# GLOBAL CONFIGURATION
# -----------------------------
BEV_SIZE = 200  # nuScenes-style BEV resolution (200x200)

# -----------------------------
# UTILITY: ANGLE WRAP (for orientation error)
# -----------------------------
def _wrap_angle_diff(a, b):
    """
    Compute smallest absolute wrapped angle difference between a and b (radians).
    Returns values in [0, pi]. Used for mAOE-like orientation error.
    """
    d = a - b  # raw difference
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()  # wrapped absolute diff

# -----------------------------
# NUSCENES-LIKE DETECTION METRICS (ROBUST VERSION)
# -----------------------------
def compute_detection_metrics_like_nuscenes(
    states,
    gt_boxes,
    conf_threshold=0.2,
    match_dist=2.0
):
    """
    Robust nuScenes-like evaluation for your DetectionHead output.

    Expected shapes:
      states   : [B, 900, 10] = [x,y,z,h,w,l,theta,vx,vy,obj_logit]
      gt_boxes : [B, N_gt, 9] = [x,y,z,h,w,l,theta,vx,vy]

    Key fixes vs your previous version:
      - Handles gt_boxes shaped incorrectly (e.g., [B,1,N_gt,9]) by squeezing safely
      - Computes AP via 11-pt interpolation
      - If TP == 0 (no matches): error metrics become NaN, and NDS_proxy is forced to 0.0
        to avoid misleading “high score due to zero errors”.
    """
    # -----------------------------
    # BASIC SANITY + SHAPE FIXING
    # -----------------------------
    device = states.device  # use same device as predictions (CPU/GPU)
    B = states.shape[0]  # batch size from predictions

    # If gt_boxes accidentally has an extra singleton dimension, remove it
    # Example bug: gt_boxes shape was [B, 1, N_gt, 9] instead of [B, N_gt, 9]
    if gt_boxes.ndim == 4 and gt_boxes.shape[1] == 1:
        gt_boxes = gt_boxes.squeeze(1)  # now [B, N_gt, 9]

    # Validate expected last dimension
    if gt_boxes.ndim != 3 or gt_boxes.shape[-1] != 9:
        raise ValueError(
            f"gt_boxes must be [B, N_gt, 9]. Got shape {tuple(gt_boxes.shape)}"
        )

    # -----------------------------
    # ACCUMULATORS
    # -----------------------------
    all_scores = []  # store confidences for AP computation
    all_is_tp = []  # store TP flags aligned with all_scores
    total_gt = 0  # total number of GT objects across batch

    # Error lists computed only from matched pairs
    ate_list = []  # translation error (center distance)
    ase_list = []  # scale error proxy
    aoe_list = []  # orientation error
    ave_list = []  # velocity error

    # Threshold-level counts
    tp_thr = 0  # true positives
    fp_thr = 0  # false positives
    fn_thr = 0  # false negatives

    # -----------------------------
    # PER-SAMPLE EVALUATION LOOP
    # -----------------------------
    for b in range(B):
        # Extract predicted states for this sample
        pred = states[b]  # [900,10]

        # Split predicted boxes and objectness logit
        pred_boxes = pred[:, :9]  # [900,9]
        obj_logit = pred[:, 9]  # [900]

        # Convert objectness logits to probabilities
        pred_conf = torch.sigmoid(obj_logit)  # [900] in (0,1)

        # Filter predictions by confidence threshold
        keep = pred_conf > conf_threshold  # boolean mask
        pred_boxes_f = pred_boxes[keep]  # [N_pred,9]
        pred_conf_f = pred_conf[keep]  # [N_pred]

        # Ground truth boxes for this sample
        gt_b = gt_boxes[b]  # [N_gt,9]
        n_gt = int(gt_b.shape[0])  # number of GT objects

        # Accumulate total GT
        total_gt += n_gt

        # Case 1: no GT -> every prediction is FP
        if n_gt == 0:
            if pred_boxes_f.shape[0] > 0:
                all_scores.append(pred_conf_f.detach())
                all_is_tp.append(torch.zeros_like(pred_conf_f, dtype=torch.bool))
                fp_thr += int(pred_boxes_f.shape[0])
            continue  # next sample

        # Case 2: no predictions -> every GT is FN
        if pred_boxes_f.shape[0] == 0:
            fn_thr += n_gt
            continue  # next sample

        # Sort predictions by descending confidence (standard AP procedure)
        order = torch.argsort(pred_conf_f, descending=True)
        pred_boxes_f = pred_boxes_f[order]
        pred_conf_f = pred_conf_f[order]

        # Compute pairwise center distances between predictions and GT
        centers_pred = pred_boxes_f[:, :3]  # [N_pred,3]
        centers_gt = gt_b[:, :3]  # [N_gt,3]
        dist_matrix = torch.cdist(centers_pred, centers_gt, p=2)  # [N_pred,N_gt]

        # Track which GT boxes are already matched (one-to-one matching)
        gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)

        # TP flags for each prediction
        is_tp = torch.zeros(pred_boxes_f.shape[0], dtype=torch.bool, device=device)

        # Greedy matching in confidence order
        for i in range(pred_boxes_f.shape[0]):
            # Find closest GT for prediction i
            min_d, j = dist_matrix[i].min(dim=0)

            # If within threshold and GT not matched yet -> TP
            if (min_d <= match_dist) and (not gt_matched[j]):
                gt_matched[j] = True
                is_tp[i] = True

                # Matched pair (pred box pb, gt box gb)
                pb = pred_boxes_f[i]
                gb = gt_b[j]

                # mATE proxy: translation error = center distance
                ate_list.append(min_d.detach())

                # mASE proxy: relative L1 error on (h,w,l)
                dims_p = pb[3:6]
                dims_g = gb[3:6]
                ase = torch.mean(torch.abs(dims_p - dims_g) / (dims_g.abs() + 1e-6))
                ase_list.append(ase.detach())

                # mAOE proxy: wrapped absolute heading difference
                aoe = _wrap_angle_diff(pb[6], gb[6])
                aoe_list.append(aoe.detach())

                # mAVE proxy: velocity L2 error on (vx,vy)
                v_p = pb[7:9]
                v_g = gb[7:9]
                ave = torch.norm(v_p - v_g, p=2)
                ave_list.append(ave.detach())

        # Compute TP/FP/FN for this sample
        tp_b = int(is_tp.sum().item())
        fp_b = int((~is_tp).sum().item())
        fn_b = int((~gt_matched).sum().item())

        # Accumulate totals
        tp_thr += tp_b
        fp_thr += fp_b
        fn_thr += fn_b

        # Store for AP computation
        all_scores.append(pred_conf_f.detach())
        all_is_tp.append(is_tp.detach())

    # -----------------------------
    # PRECISION / RECALL AT THRESHOLD
    # -----------------------------
    precision = tp_thr / (tp_thr + fp_thr + 1e-8)
    recall = tp_thr / (tp_thr + fn_thr + 1e-8)

    # -----------------------------
    # 11-POINT INTERPOLATED AP
    # -----------------------------
    AP = 0.0  # default

    # Concatenate all prediction scores/TP flags across batch
    if len(all_scores) > 0:
        scores = torch.cat(all_scores, dim=0)
        is_tp_all = torch.cat(all_is_tp, dim=0)
    else:
        scores = torch.tensor([], device=device)
        is_tp_all = torch.tensor([], device=device, dtype=torch.bool)

    # Compute AP only if we have predictions and GT
    if scores.numel() > 0 and total_gt > 0:
        # Sort by descending confidence
        order = torch.argsort(scores, descending=True)
        is_tp_sorted = is_tp_all[order].to(torch.float32)

        # Cumulative TP/FP
        tp_cum = torch.cumsum(is_tp_sorted, dim=0)
        fp_cum = torch.cumsum(1.0 - is_tp_sorted, dim=0)

        # Precision/recall curve
        prec_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
        rec_curve = tp_cum / (total_gt + 1e-8)

        # 11-point interpolation
        ap_sum = 0.0
        for r in torch.linspace(0, 1, 11, device=device):
            mask = rec_curve >= r
            p_at_r = prec_curve[mask].max().item() if torch.any(mask) else 0.0
            ap_sum += p_at_r
        AP = ap_sum / 11.0

    # -----------------------------
    # MEAN ERRORS (MATCHED PAIRS ONLY)
    # If no matches, errors are undefined => NaN (not 0.0)
    # -----------------------------
    def mean_or_nan(lst):
        return torch.stack(lst).mean().item() if len(lst) > 0 else float("nan")

    mATE = mean_or_nan(ate_list)
    mASE = mean_or_nan(ase_list)
    mAOE = mean_or_nan(aoe_list)
    mAVE = mean_or_nan(ave_list)

    # -----------------------------
    # NDS PROXY (ROBUST)
    # If tp_thr==0 => no matched pairs => force NDS_proxy = 0.0
    # -----------------------------
    if tp_thr == 0:
        nds_proxy = 0.0
    else:
        # Convert errors into normalized "1 - error/scale" terms
        # Clamp each term to [0,1] to keep score bounded
        def inv_norm(x, scale):
            if math.isnan(x):
                return 0.0
            return max(0.0, 1.0 - min(1.0, x / scale))

        nds_proxy = (
            AP
            + inv_norm(mATE, 2.0)
            + inv_norm(mASE, 0.5)
            + inv_norm(mAOE, math.pi)
            + inv_norm(mAVE, 2.0)
        ) / 5.0

    # -----------------------------
    # RETURN METRICS DICTIONARY
    # -----------------------------
    return {
        "tp": tp_thr,
        "fp": fp_thr,
        "fn": fn_thr,
        "precision": float(precision),
        "recall": float(recall),
        "AP": float(AP),
        "NDS_proxy": float(nds_proxy),
        "mATE": mATE,
        "mASE": mASE,
        "mAOE": mAOE,
        "mAVE": mAVE,
        "total_gt": int(total_gt),
    }

# -----------------------------
# MODALITY ENCODERS (SIMPLIFIED, BEV-ALIGNED)
# -----------------------------
class CameraEncoder(nn.Module):
    """6-view camera -> BEV features [B,256,200,200] (simplified LSS-style aggregation)."""
    def __init__(self):
        super().__init__()  # initialize module base class
        self.backbone = nn.Conv2d(3, 256, 3, padding=1)  # simple CNN proxy for ResNet features
        self.bev_interp = nn.Upsample(size=(BEV_SIZE, BEV_SIZE), mode="bilinear")  # resize to BEV_SIZE

    def forward(self, images):
        """images: [B,6,3,H,W] -> returns [B,256,200,200]."""
        B, N, C, H, W = images.shape  # unpack input dimensions
        images = images.view(B * N, C, H, W)  # merge camera dimension into batch
        feats = self.backbone(images)  # run CNN backbone: [B*6,256,H,W]
        feats = feats.view(B, N, 256, H, W)  # restore camera dimension: [B,6,256,H,W]
        bev_feats = feats.mean(dim=1)  # aggregate views (proxy for LSS): [B,256,H,W]
        return self.bev_interp(bev_feats)  # resize to BEV grid: [B,256,200,200]

class LidarEncoder(nn.Module):
    """LiDAR BEV -> [B,384,200,200] (PointPillars-style proxy)."""
    def __init__(self):
        super().__init__()  # initialize module
        self.conv = nn.Conv2d(1, 384, 3, padding=1)  # simple BEV conv

    def forward(self, lidar_bev):
        """lidar_bev: [B,1,200,200] -> [B,384,200,200]."""
        return self.conv(lidar_bev)  # apply conv

class RadarEncoder(nn.Module):
    """RADAR BEV -> [B,128,200,200] (velocity-preserving proxy)."""
    def __init__(self):
        super().__init__()  # initialize module
        self.conv = nn.Conv2d(2, 128, 3, padding=1)  # 2 input channels (e.g., range + Doppler)

    def forward(self, radar_bev):
        """radar_bev: [B,2,200,200] -> [B,128,200,200]."""
        return self.conv(radar_bev)  # apply conv

# -----------------------------
# CMAA FUSION MODULE (GATING-BASED, LOW-OVERHEAD)
# -----------------------------
class CMAAFusion(nn.Module):
    """Cross-Modal Adaptive Attention (CMAA): pooled gating + weighted concatenation + 1x1 projection."""
    def __init__(self):
        super().__init__()  # initialize module
        total_channels = 256 + 384 + 128  # sum of modality channels

        # Gating MLP: predicts 3 modality weights from concatenated pooled descriptors
        self.gating_mlp = nn.Sequential(
            nn.Linear(total_channels, 512),  # expand
            nn.GELU(),  # nonlinearity
            nn.Linear(512, 256),  # compress
            nn.GELU(),  # nonlinearity
            nn.Linear(256, 3)  # output logits for (camera, lidar, radar)
        )

        # 1x1 conv to project concatenated features to unified BEV dimension
        self.proj = nn.Conv2d(total_channels, 512, kernel_size=1)

    def forward(self, F_camera, F_lidar, F_radar):
        """Inputs: [B,Cm,200,200] -> Output: [B,512,200,200]."""
        B = F_camera.shape[0]  # batch size

        # Global average pooling (scene descriptors)
        G_camera = F_camera.mean(dim=(2, 3))  # [B,256]
        G_lidar = F_lidar.mean(dim=(2, 3))  # [B,384]
        G_radar = F_radar.mean(dim=(2, 3))  # [B,128]

        # Concatenate descriptors
        G_concat = torch.cat([G_camera, G_lidar, G_radar], dim=1)  # [B,768]

        # Compute modality weights alpha via softmax
        alpha = F.softmax(self.gating_mlp(G_concat), dim=1)  # [B,3], sum to 1

        # Weight each modality feature map (broadcast alpha over channels/spatial)
        F_camera_w = F_camera * alpha[:, 0].view(B, 1, 1, 1)
        F_lidar_w = F_lidar * alpha[:, 1].view(B, 1, 1, 1)
        F_radar_w = F_radar * alpha[:, 2].view(B, 1, 1, 1)

        # Concatenate weighted features
        F_concat = torch.cat([F_camera_w, F_lidar_w, F_radar_w], dim=1)  # [B,768,200,200]

        # Project to unified BEV representation
        F_bev = self.proj(F_concat)  # [B,512,200,200]
        return F_bev  # return fused BEV

# -----------------------------
# TASK HEADS (DETECTION + FORECAST)
# -----------------------------
class DetectionHead(nn.Module):
    """DETR-style query head: 900 queries -> 10D states."""
    def __init__(self):
        super().__init__()  # initialize module
        self.num_queries = 900  # number of queries
        self.queries = nn.Parameter(torch.randn(self.num_queries, 768))  # learnable queries
        self.pool_proj = nn.Linear(512, 768)  # project pooled BEV to query embedding space
        self.state_head = nn.Linear(768, 10)  # output state vector

    def forward(self, F_bev):
        """F_bev: [B,512,200,200] -> states: [B,900,10]."""
        B = F_bev.shape[0]  # batch size
        scene_context = F_bev.mean(dim=(2, 3))  # global pooling [B,512]
        scene_context_proj = self.pool_proj(scene_context)  # [B,768]
        queries_expanded = self.queries.unsqueeze(0).repeat(B, 1, 1)  # [B,900,768]
        query_features = queries_expanded + scene_context_proj.unsqueeze(1)  # fuse context
        states = self.state_head(query_features)  # [B,900,10]
        return states  # return predicted states

class ForecastingHead(nn.Module):
    """Multimodal trajectory prediction: K=6, T=30, XY=2."""
    def __init__(self):
        super().__init__()  # initialize module
        self.traj_head = nn.Linear(512, 6 * 30 * 2)  # output K*T*2 coordinates

    def forward(self, F_bev):
        """F_bev: [B,512,200,200] -> trajs: [B,6,30,2]."""
        pooled = F_bev.mean(dim=(2, 3))  # [B,512]
        traj_flat = self.traj_head(pooled)  # [B,360]
        trajs = traj_flat.view(-1, 6, 30, 2)  # [B,6,30,2]
        return trajs  # return trajectories

# -----------------------------
# INTEGRATED END-TO-END SYSTEM
# -----------------------------
class IntegratedCMAASystem(nn.Module):
    """Sensors -> encoders -> CMAA -> detection + forecasting."""
    def __init__(self):
        super().__init__()  # initialize module
        self.cam_enc = CameraEncoder()  # camera encoder
        self.lidar_enc = LidarEncoder()  # lidar encoder
        self.radar_enc = RadarEncoder()  # radar encoder
        self.cmaa = CMAAFusion()  # fusion module
        self.det_head = DetectionHead()  # detection head
        self.fcst_head = ForecastingHead()  # forecasting head

    def forward(self, images, lidar_bev, radar_bev):
        """Forward pass through entire pipeline."""
        F_c = self.cam_enc(images)  # camera BEV features
        F_l = self.lidar_enc(lidar_bev)  # lidar BEV features
        F_r = self.radar_enc(radar_bev)  # radar BEV features
        F_bev = self.cmaa(F_c, F_l, F_r)  # fused BEV features
        states = self.det_head(F_bev)  # detection states
        trajs = self.fcst_head(F_bev)  # trajectories
        return states, trajs  # return both outputs

# -----------------------------
# VISUALIZATION FUNCTIONS (ROBUST TO NaNs)
# -----------------------------
def plot_pipeline_diagram(save_path="figs/cmaa_pipeline.png"):
    """Draw a simple pipeline diagram with tensor shapes."""
    os.makedirs("figs", exist_ok=True)  # ensure output directory exists
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))  # create figure
    ax.axis("off")  # hide axes

    # Define labeled boxes (text, x, y, color)
    boxes = [
        ("Camera\n[B,6,3,64,64]", 0.05, 0.85, "lightblue"),
        ("CamEnc\n[B,256,200,200]", 0.05, 0.70, "lightcyan"),
        ("LiDAR\n[B,1,200,200]", 0.35, 0.85, "lightgreen"),
        ("LiDAREnc\n[B,384,200,200]", 0.35, 0.70, "palegreen"),
        ("RADAR\n[B,2,200,200]", 0.65, 0.85, "lightcoral"),
        ("RADAREnc\n[B,128,200,200]", 0.65, 0.70, "lightpink"),
        ("CMAA Fusion\n[B,512,200,200]", 0.35, 0.50, "lavender"),
        ("Detection\n[B,900,10]", 0.20, 0.25, "orange"),
        ("Forecast\n[B,6,30,2]", 0.55, 0.25, "gold"),
    ]

    # Draw boxes
    for text, x, y, color in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 0.18, 0.08, fill=True, color=color, ec="black", lw=1.5)
        )
        ax.text(x + 0.09, y + 0.04, text, ha="center", va="center", fontsize=10, weight="bold")

    # Draw arrows (x1,y1 -> x2,y2)
    arrows = [
        (0.14, 0.89, 0.14, 0.78),
        (0.44, 0.89, 0.44, 0.78),
        (0.74, 0.89, 0.74, 0.78),
        (0.14, 0.74, 0.395, 0.54),
        (0.44, 0.74, 0.395, 0.54),
        (0.74, 0.74, 0.395, 0.54),
        (0.395, 0.48, 0.29, 0.29),
        (0.395, 0.48, 0.635, 0.29),
    ]

    # Annotate arrows
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

    # Set limits and title
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("CMAA Pipeline - Verified Tensor Shapes", fontsize=16, weight="bold", pad=20)

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Pipeline saved: {save_path}")

def plot_modality_weights(save_path="figs/modality_weights.png"):
    """Example CMAA weights plot (illustrative)."""
    os.makedirs("figs", exist_ok=True)
    conditions = ["Clear", "Rain", "Fog", "Night"]
    cam_w = [0.65, 0.35, 0.15, 0.25]
    lidar_w = [0.25, 0.45, 0.65, 0.55]
    radar_w = [0.10, 0.20, 0.20, 0.20]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, cam_w, width, label="Camera", alpha=0.8, edgecolor="navy")
    ax.bar(x, lidar_w, width, label="LiDAR", alpha=0.8, edgecolor="darkgreen")
    ax.bar(x + width, radar_w, width, label="RADAR", alpha=0.8, edgecolor="darkred")

    ax.set_xlabel("Weather Conditions", fontsize=12, weight="bold")
    ax.set_ylabel("Gating Weight α", fontsize=12, weight="bold")
    ax.set_title("CMAA Adaptive Modality Weights (Illustrative)", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" Weights saved: {save_path}")

def plot_nds_breakdown(metrics, save_path="figs/nds_breakdown.png"):
    """Plot NDS_proxy components; handles NaNs safely."""
    os.makedirs("figs", exist_ok=True)

    # Helper: safe normalization
    def safe_inv_norm(x, scale):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(0.0, 1.0 - min(1.0, x / scale))

    # Build component list
    names = ["AP", "1-mATE", "1-mASE", "1-mAOE", "1-mAVE"]
    vals = [
        metrics["AP"],
        safe_inv_norm(metrics["mATE"], 2.0),
        safe_inv_norm(metrics["mASE"], 0.5),
        safe_inv_norm(metrics["mAOE"], math.pi),
        safe_inv_norm(metrics["mAVE"], 2.0),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, vals, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized Score")
    ax.set_title(f"NDS_proxy Components (NDS_proxy={metrics['NDS_proxy']:.3f})", weight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" NDS breakdown saved: {save_path}")

# -----------------------------
# MAIN DEMO
# -----------------------------
if __name__ == "__main__":
    print(" CMAA System - Complete End-to-End Demo (Fixed)")
    print("=" * 70)

    # -----------------------------
    # SETUP MOCK INPUTS
    # -----------------------------
    B, H, W = 1, 64, 64  # batch size and image resolution
    print(f"1) Setup: B={B}, Images={H}x{W}, BEV={BEV_SIZE}x{BEV_SIZE}")

    # Create mock 6-camera RGB tensor
    images = torch.randn(B, 6, 3, H, W)  # [B,6,3,H,W]
    print("2) Created mock camera images:", tuple(images.shape))

    # Create mock LiDAR BEV tensor (positive occupancy-like)
    lidar_bev = torch.randn(B, 1, BEV_SIZE, BEV_SIZE).abs()  # [B,1,200,200]
    print("3) Created mock LiDAR BEV:", tuple(lidar_bev.shape))

    # Create mock RADAR BEV tensor (2 channels)
    radar_bev = torch.randn(B, 2, BEV_SIZE, BEV_SIZE)  # [B,2,200,200]
    print("4) Created mock RADAR BEV:", tuple(radar_bev.shape))

    # -----------------------------
    # FIXED GT BOX CREATION (IMPORTANT!)
    # The previous code had an extra nesting level causing [B,1,N_gt,9].
    # We create GT as [B,N_gt,9] directly.
    # -----------------------------
    gt_boxes = torch.tensor(
        [[  # Batch dimension B=1
            [10.0,   5.0,  1.7, 1.8, 4.5, 2.0,  0.1,  2.0,  0.5],
            [-5.0,   8.0,  1.7, 1.8, 4.5, 2.0, -0.2,  1.5, -0.3],
            [ 0.0, -12.0,  1.6, 0.6, 1.8, 0.4,  1.2,  0.0,  0.0],
            [20.0,   3.0,  1.7, 1.8, 4.5, 2.0,  0.0,  3.0,  0.0],
            [-15.0, -5.0,  1.7, 1.8, 4.5, 2.0,  0.3, -1.0,  0.2],
        ]],
        dtype=torch.float32,
    )  # shape [1,5,9]
    print("5) Created GT boxes:", tuple(gt_boxes.shape), "(should be [B,N_gt,9])")

    # -----------------------------
    # RUN MODEL
    # -----------------------------
    model = IntegratedCMAASystem()  # instantiate model
    states, trajs = model(images, lidar_bev, radar_bev)  # forward pass

    # Print shapes
    print("\n6) Forward pass complete:")
    print("   Detection states:", tuple(states.shape), "expected [B,900,10]")
    print("   Trajectories    :", tuple(trajs.shape), "expected [B,6,30,2]")

    # -----------------------------
    # COMPUTE METRICS (FIXED)
    # -----------------------------
    print("\n7) Computing nuScenes-like detection metrics (robust)...")
    metrics = compute_detection_metrics_like_nuscenes(states, gt_boxes)

    # -----------------------------
    # PRINT METRICS SAFELY
    # -----------------------------
    print("\n=== CMAA Detection Metrics (nuScenes-like) ===")
    print(f"{'Metric':<12} {'Value':<12}")
    print("-" * 26)

    # Helper printer to handle ints/floats/NaNs
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:d}"
        if isinstance(v, float):
            return "N/A" if math.isnan(v) else f"{v:.3f}"
        return str(v)

    for k in ["tp", "fp", "fn", "precision", "recall", "AP", "NDS_proxy", "mATE", "mASE", "mAOE", "mAVE", "total_gt"]:
        print(f"{k:<12}: {fmt(metrics[k]):<12}")

    # -----------------------------
    # GENERATE FIGURES
    # -----------------------------
    print("\n8) Generating figures...")
    plot_pipeline_diagram()
    plot_modality_weights()
    plot_nds_breakdown(metrics)

    print("\n Done. Figures saved under ./figs/")
