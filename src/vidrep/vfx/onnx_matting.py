"""
Local ONNX inference wrappers for:
  1. Depth Anything V2 Small (relative depth + closer foreground occluder detection)
  2. SAM 2.1 Hiera Tiny (box+point prompted semantic foreground matting)

Weights auto-download on first use via `onnx_models.ensure_model_downloaded()`.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_models import ensure_model_downloaded

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _select_providers() -> list[str]:
    """Use CPUExecutionProvider by default for deterministic cross-platform dynamic-shape ONNX."""
    avail = ort.get_available_providers()
    if os.environ.get("VIDREP_ONNX_COREML", "").strip() == "1" and "CoreMLExecutionProvider" in avail:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class DepthAnythingV2Onnx:
    def __init__(self, auto_download: bool = True) -> None:
        if auto_download:
            paths = ensure_model_downloaded("depth")
        else:
            from .onnx_models import get_model_cache_root
            paths = {"model.onnx": get_model_cache_root() / "depth/model.onnx"}
        self.model_path = paths["model.onnx"]
        self.session = ort.InferenceSession(str(self.model_path), providers=_select_providers())

    def predict_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Predict normalized relative depth in [0, 1] at full frame size (1.0 = closest to camera)."""
        h, w = frame_bgr.shape[:2]
        # Aspect-preserving dimensions divisible by 14 (patch size of ViT-S)
        if w >= h:
            in_w = 518
            in_h = max(140, int(round((518 * h / w) / 14.0)) * 14)
        else:
            in_h = 518
            in_w = max(140, int(round((518 * w / h) / 14.0)) * 14)

        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

        raw = self.session.run(None, {"pixel_values": x})[0][0]
        d_min, d_max = float(raw.min()), float(raw.max())
        norm = (raw - d_min) / (d_max - d_min + 1e-6)
        return cv2.resize(norm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def detect_foreground_occluders(
        self,
        orig_bgr: np.ndarray,
        subject_core_mask: np.ndarray,
        candidate_mask: np.ndarray,
        depth_margin: float = 0.14,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Detect pixels inside `candidate_mask` where an object in the untouched original plate
        is significantly closer to the camera than the replaced subject (e.g., foreground tree trunk,
        lamp post, or passing pedestrian). Returns (occluder_mask_01, depth_map_01, occlusion_ratio).
        """
        depth = self.predict_depth(orig_bgr)
        h, w = depth.shape
        core_bool = subject_core_mask > 0.5
        cand_bool = candidate_mask > 0.08
        if int(core_bool.sum()) < 50 or int(cand_bool.sum()) < 50:
            return np.zeros((h, w), dtype=np.float32), depth, 0.0

        # Reference depth of the original subject itself (p95 so the original subject's own closest
        # body/head/shoulders can NEVER be mistaken for an external foreground occluder!)
        ys, xs = np.where(core_bool)
        y_cut = int(np.percentile(ys, 82))
        upper_core = core_bool.copy()
        upper_core[y_cut:, :] = False
        ref_region = upper_core if int(upper_core.sum()) >= 30 else core_bool
        d_subject = float(np.percentile(depth[ref_region], 95))

        # Genuine foreground occluder (e.g., passing tree trunk, pole, pedestrian in front of subject)
        # must be BOTH significantly closer than d_subject (by depth_margin) AND extend continuously
        # beyond the subject core (not an isolated pixel inside the replaced subject itself).
        y_ground = int(np.percentile(ys, 90))
        closer_all = depth > (d_subject + depth_margin)
        closer_all[y_ground:, :] = False

        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(closer_all.astype(np.uint8))
        valid_occ = np.zeros((h, w), dtype=np.uint8)
        for lbl in range(1, num_lbl):
            comp = labels == lbl
            in_cand = comp & cand_bool
            out_core = comp & (~core_bool)
            # Must touch candidate mask AND have substantial exterior anchor (real occluder like tree/pole)
            if int(in_cand.sum()) >= 80 and int(out_core.sum()) >= max(200, int(in_cand.sum()) * 0.5):
                valid_occ[in_cand] = 255

        valid_occ = cv2.morphologyEx(valid_occ, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        valid_occ = cv2.morphologyEx(valid_occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        occ_soft = cv2.GaussianBlur(valid_occ.astype(np.float32) / 255.0, (15, 15), 4.0)
        occ_ratio = float(np.sum(occ_soft > 0.35) / max(1, int(cand_bool.sum())))
        return occ_soft, depth, round(occ_ratio, 4)


class Sam2OnnxMatte:
    def __init__(self, auto_download: bool = True) -> None:
        if auto_download:
            paths = ensure_model_downloaded("sam2")
        else:
            from .onnx_models import get_model_cache_root
            base = get_model_cache_root() / "sam2"
            paths = {
                "vision_encoder.onnx": base / "vision_encoder.onnx",
                "prompt_encoder_mask_decoder.onnx": base / "prompt_encoder_mask_decoder.onnx",
            }
        provs = _select_providers()
        self.enc = ort.InferenceSession(str(paths["vision_encoder.onnx"]), providers=provs)
        self.dec = ort.InferenceSession(str(paths["prompt_encoder_mask_decoder.onnx"]), providers=provs)

    def segment_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, float]:
        """
        Encode `frame_bgr` once and decode crisp semantic masks for all detected subject boxes.
        Returns (union_mask_float32_in_0_1, mean_best_iou).
        """
        h, w = frame_bgr.shape[:2]
        if not boxes_xyxy:
            return np.zeros((h, w), dtype=np.float32), 0.0

        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (1024, 1024), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        sx = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        sx = np.transpose(sx, (2, 0, 1))[None].astype(np.float32)

        e0, e1, e2 = self.enc.run(None, {"pixel_values": sx})

        union = np.zeros((h, w), dtype=np.float32)
        ious: list[float] = []
        for x0, y0, x1, y1 in boxes_xyxy:
            bx0, by0 = x0 / w * 1024.0, y0 / h * 1024.0
            bx1, by1 = x1 / w * 1024.0, y1 / h * 1024.0
            cx, cy = (bx0 + bx1) * 0.5, (by0 + by1) * 0.5
            pts = np.array([[[[cx, cy]]]], dtype=np.float32)
            lbls = np.array([[[1]]], dtype=np.int64)
            boxes = np.array([[[bx0, by0, bx1, by1]]], dtype=np.float32)

            iou_scores, pred_masks, _ = self.dec.run(
                None,
                {
                    "input_points": pts,
                    "input_labels": lbls,
                    "input_boxes": boxes,
                    "image_embeddings.0": e0,
                    "image_embeddings.1": e1,
                    "image_embeddings.2": e2,
                },
            )
            best_idx = int(np.argmax(iou_scores[0, 0]))
            best_iou = float(iou_scores[0, 0, best_idx])
            ious.append(best_iou)
            # Smooth sigmoid probability from raw logits (preserves soft anti-aliased edges!)
            logits = pred_masks[0, 0, best_idx]
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
            prob_full = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            union = np.maximum(union, prob_full)

        return union, round(float(np.mean(ious)), 4)


def propagate_masks_optical_flow(
    frames_bgr: list[np.ndarray],
    keyframe_masks: dict[int, np.ndarray],
) -> list[np.ndarray]:
    """
    Propagate keyframe ONNX masks cleanly to intermediate frames using bidirectional DIS optical flow
    so running SAM2/Depth every N frames remains temporally razor-sharp at full 24fps.
    """
    n = len(frames_bgr)
    if not keyframe_masks:
        h, w = frames_bgr[0].shape[:2]
        return [np.zeros((h, w), dtype=np.float32) for _ in range(n)]

    sorted_kfs = sorted(keyframe_masks.keys())
    if len(sorted_kfs) == n:
        return [keyframe_masks[i] for i in range(n)]

    h, w = frames_bgr[0].shape[:2]
    sw, sh = min(480, w), max(120, int(round(min(480, w) * h / w)))
    grays = [cv2.cvtColor(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    kf_small = {k: cv2.resize(v, (sw, sh), interpolation=cv2.INTER_LINEAR) for k, v in keyframe_masks.items()}

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))

    out: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    for k in sorted_kfs:
        out[k] = keyframe_masks[k]

    # Fill leading/trailing frames if any
    for i in range(0, sorted_kfs[0]):
        out[i] = keyframe_masks[sorted_kfs[0]]
    for i in range(sorted_kfs[-1] + 1, n):
        out[i] = keyframe_masks[sorted_kfs[-1]]

    # Interpolate each interval (k0, k1) with bidirectional flow warp
    for idx in range(len(sorted_kfs) - 1):
        k0, k1 = sorted_kfs[idx], sorted_kfs[idx + 1]
        if k1 - k0 <= 1:
            continue
        m0, m1 = kf_small[k0], kf_small[k1]
        for i in range(k0 + 1, k1):
            alpha = (i - k0) / float(k1 - k0)
            flow_i_to_0 = dis.calc(grays[i], grays[k0], None)
            flow_i_to_1 = dis.calc(grays[i], grays[k1], None)
            w0 = cv2.remap(m0, gx + flow_i_to_0[:, :, 0], gy + flow_i_to_0[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(m1, gx + flow_i_to_1[:, :, 0], gy + flow_i_to_1[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            blended = (1.0 - alpha) * w0 + alpha * w1
            out[i] = cv2.resize(blended.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    return out
