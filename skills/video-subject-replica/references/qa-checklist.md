# Universal Visual & Quantitative QA Checklist

Before declaring a preview or final replica complete, verify EVERY item below against actual output files (never assume request parameters equal actual outputs):

## 1. Stream & Delivery Integrity (`metrics.json`)
- [ ] Verify actual `width x height`, `fps`, `duration`, `pix_fmt`, and `color_space / color_primaries / color_transfer` match the source profile (except expected 24fps edit rate).
- [ ] Verify original audio track is present, lossless (`aac`), and matches video duration (`-shortest`, no speed warp).

## 2. Subject Anatomy, Silhouette & Mechanical Motion (`*-qa-motion-5fps.jpg`)
- [ ] Inspect all 12 consecutive 5fps zoomed frames.
- [ ] Confirm posture/anatomy stays consistent (no standing/scooter collapse, no extra limbs, no melting hands/weapons/faces).
- [ ] Confirm mechanical/biological motion progresses naturally across consecutive frames (e.g., alternating pedaling, natural stride, creature breathing/tail/wing motion).
- [ ] Confirm zero residue of original subject, helmet, weapon, or silhouette.

## 3. Lighting, Shadows & Grounding (`*-qa-comparison.jpg`)
- [ ] Check contact points (feet, wheels, paws, hands) stay grounded with no floating or skating.
- [ ] Check cast shadows match scene sun/key-light direction and intensity.
- [ ] Check lighting transitions when subject passes through shade/highlights.

## 4. Background Fidelity & Seam Verification (`*-qa-comparison.jpg` + `*-qa-heatmaps.jpg`)
- [ ] Check `BLEND MASK` column: confirm the red overlay fully covers ALL replaced subjects AND cast shadows AND erased old-subject footprints, without unnecessarily covering distant background details.
- [ ] Check `|COMPOSITE - ORIG|` heatmap: background outside the replaced region should be nearly pitch black (`bg_mae_composited_vs_orig` typically `< 1.0` pre-encode).
- [ ] Check `segmentation.recommend_graded_fallback`: if `true` (high coverage or foreground occlusion), inspect `*-graded-with-audio.mp4` vs `*-composited-with-audio.mp4` side-by-side and choose the cleaner deliverable honestly.

## 5. Honest Reporting
- Always report actual measured metrics and any remaining edge imperfections (e.g., 720p upsampled foreground softness, 24fps vs 60fps cadence, minor occlusion frames) transparently to the user.
