"""Automated visual QA contact sheets, 5-column diagnostics, 5fps motion checks, and heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _make_heatmap(a: np.ndarray, b: np.ndarray, gain: float = 3.5) -> np.ndarray:
    d = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2)
    return cv2.applyColorMap(np.clip(d * gain, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def generate_qa_bundle(
    out_dir: Path,
    orig_matched: list[np.ndarray],
    raw_scaled: list[np.ndarray],
    graded_frames: list[np.ndarray],
    comp_frames: list[np.ndarray],
    masks: list[np.ndarray],
    centroids: list[list[float]],
    fps: float,
    tag: str = "final",
    onnx_diag: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(comp_frames)
    h, w = comp_frames[0].shape[:2]

    # 1. Contact sheet (2fps)
    dur = n / max(fps, 1e-3)
    cs_count = max(4, min(16, int(round(dur * 2.0))))
    cs_tiles = []
    for k in range(cs_count):
        gi = min(n - 1, int(round((k + 0.5) / cs_count * (n - 1))))
        tile = cv2.resize(comp_frames[gi], (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{gi/fps:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cs_tiles.append(tile)
    cols = 4
    rows = (len(cs_tiles) + cols - 1) // cols
    while len(cs_tiles) < rows * cols:
        cs_tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
    cs_grid = np.vstack([np.hstack(cs_tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    p_cs = out_dir / f"{tag}-contact-sheet.jpg"
    cv2.imwrite(str(p_cs), cs_grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 2. 5-column comparison sheet across up to 8 timestamps: Orig | Graded | Composite | Mask | Zoom
    n_samples = min(8, max(3, n))
    sample_gis = sorted({min(n - 1, int(round(k / max(1, n_samples - 1) * (n - 1)))) for k in range(n_samples)})
    comp_rows = []
    for gi in sample_gis:
        t = gi / fps
        fo, fg, fc, m = orig_matched[gi], graded_frames[gi], comp_frames[gi], masks[gi]
        ov = fc.copy()
        red = np.zeros_like(ov)
        red[:, :, 2] = 255
        ov = np.clip(ov.astype(np.float32) * (1.0 - 0.38 * m[:, :, None]) + red * (0.38 * m[:, :, None]), 0, 255).astype(np.uint8)

        zcx, zcy = int(round(centroids[gi][0])), int(round(centroids[gi][1]))
        zw, zh = min(w, max(240, w // 4)), min(h, max(180, h // 3))
        zx0 = max(0, min(w - zw, zcx - zw // 2))
        zy0 = max(0, min(h - zh, zcy - zh // 2))
        zoom = cv2.resize(fc[zy0 : zy0 + zh, zx0 : zx0 + zw], (320, 240), interpolation=cv2.INTER_AREA)

        t1 = cv2.resize(fo, (384, 240))
        t2 = cv2.resize(fg, (384, 240))
        t3 = cv2.resize(fc, (384, 240))
        t4 = cv2.resize(ov, (384, 240))
        for img, label in [
            (t1, f"ORIG {t:.2f}s"),
            (t2, "GRADED"),
            (t3, "COMPOSITE"),
            (t4, "BLEND MASK"),
            (zoom, "SUBJECT ZOOM"),
        ]:
            cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        comp_rows.append(np.hstack([t1, t2, t3, t4, zoom]))
    p_diag = out_dir / f"{tag}-qa-comparison.jpg"
    cv2.imwrite(str(p_diag), np.vstack(comp_rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 3. 5fps consecutive motion strip (12 frames around middle of clip)
    step_f = max(1, int(round(fps / 5.0)))
    mid = n // 2
    start_f = max(0, min(n - 12 * step_f, mid - 6 * step_f))
    mot_tiles = []
    for k in range(12):
        gi = min(n - 1, max(0, start_f + k * step_f))
        zcx, zcy = int(round(centroids[gi][0])), int(round(centroids[gi][1]))
        zw, zh = min(w, max(260, int(w * 0.28))), min(h, max(220, int(h * 0.38)))
        zx0 = max(0, min(w - zw, zcx - zw // 2))
        zy0 = max(0, min(h - zh, zcy - zh // 2))
        crop = cv2.resize(comp_frames[gi][zy0 : zy0 + zh, zx0 : zx0 + zw], (320, 260), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, f"{gi/fps:.2f}s (f{gi})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        mot_tiles.append(crop)
    mot_grid = np.vstack([np.hstack(mot_tiles[r * 4 : (r + 1) * 4]) for r in range(3)])
    p_mot = out_dir / f"{tag}-qa-motion-5fps.jpg"
    cv2.imwrite(str(p_mot), mot_grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 4. Error heatmap comparison at middle frame
    hm_raw = cv2.resize(_make_heatmap(orig_matched[mid], raw_scaled[mid]), (480, 270))
    hm_grd = cv2.resize(_make_heatmap(orig_matched[mid], graded_frames[mid]), (480, 270))
    hm_cmp = cv2.resize(_make_heatmap(orig_matched[mid], comp_frames[mid]), (480, 270))
    cv2.putText(hm_raw, "|RAW - ORIG|", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(hm_grd, "|GRADED - ORIG|", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(hm_cmp, "|COMPOSITE - ORIG|", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    p_hm = out_dir / f"{tag}-qa-heatmaps.jpg"
    cv2.imwrite(str(p_hm), np.hstack([hm_raw, hm_grd, hm_cmp]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    qa_out = {
        "contact_sheet": str(p_cs),
        "comparison_sheet": str(p_diag),
        "motion_5fps": str(p_mot),
        "heatmaps": str(p_hm),
    }

    # 5. Optional ONNX diagnostic sheet (SAM2 crisp matte + Depth Anything V2 heatmap + occluder mask)
    if onnx_diag and (onnx_diag.get("sam2_masks") or onnx_diag.get("depth_samples")):
        sam_list = onnx_diag.get("sam2_masks")
        dep_dict = onnx_diag.get("depth_samples") or {}
        occ_list = onnx_diag.get("occluder_masks")
        dep_keys = sorted(dep_dict.keys())
        pick_gis = dep_keys[:: max(1, len(dep_keys) // 4)][:4] if dep_keys else sample_gis[:4]
        onnx_rows = []
        for gi in pick_gis:
            t = gi / fps
            t_orig = cv2.resize(orig_matched[gi], (384, 216))
            cv2.putText(t_orig, f"ORIG {t:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if gi in dep_dict:
                dm = cv2.applyColorMap(np.clip(dep_dict[gi] * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                t_dep = cv2.resize(dm, (384, 216))
                cv2.putText(t_dep, "DEPTH ANYTHING V2", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                t_dep = np.zeros((216, 384, 3), dtype=np.uint8)

            if sam_list is not None:
                sm = np.clip(sam_list[gi] * 255, 0, 255).astype(np.uint8)
                t_sam = cv2.resize(cv2.cvtColor(sm, cv2.COLOR_GRAY2BGR), (384, 216))
                cv2.putText(t_sam, "SAM 2.1 MATTE", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)
            else:
                t_sam = np.zeros((216, 384, 3), dtype=np.uint8)

            ov = comp_frames[gi].copy()
            # Green = final replace mask, Blue = detected closer occluders kept in front
            g_ch = np.zeros_like(ov); g_ch[:, :, 1] = 255
            ov = np.clip(ov * (1.0 - 0.35 * masks[gi][:, :, None]) + g_ch * (0.35 * masks[gi][:, :, None]), 0, 255).astype(np.uint8)
            if occ_list is not None:
                b_ch = np.zeros_like(ov); b_ch[:, :, 0] = 255
                ov = np.clip(ov * (1.0 - 0.55 * occ_list[gi][:, :, None]) + b_ch * (0.55 * occ_list[gi][:, :, None]), 0, 255).astype(np.uint8)
            t_ov = cv2.resize(ov, (384, 216))
            cv2.putText(t_ov, "FINAL(G) + OCCLUDER(B)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            onnx_rows.append(np.hstack([t_orig, t_dep, t_sam, t_ov]))

        p_onnx = out_dir / f"{tag}-qa-onnx-diagnostics.jpg"
        cv2.imwrite(str(p_onnx), np.vstack(onnx_rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        qa_out["onnx_diagnostics"] = str(p_onnx)

    return qa_out
