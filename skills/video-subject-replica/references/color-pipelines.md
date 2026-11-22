# Color Pipeline & HDR/HLG Pitfalls Reference

## 1. Never Use `cv2.VideoCapture` to Decode HDR/HLG MP4s on macOS
Newer `opencv-python-headless` builds on macOS default to Apple's `AVFoundation` backend (`CAP_AVFOUNDATION`). When reading an MP4 tagged `bt2020nc / arib-std-b67 (HLG)` or `smpte2084 (PQ)`, `AVFoundation` silently applies HDR-to-SDR tone-mapping into RGB. If you subsequently re-encode those pixels with `bt2020nc / HLG` tags, you cause **double tone-mapping** (neon cyan skies and pink clouds).

**Rule**: `vidrep` ALWAYS decodes frames via `ffmpeg -f rawvideo -pix_fmt bgr24 -` pipe (`vidrep.media.ffmpeg_io.decode_video_frames`), ensuring identical pixel values on macOS, Linux, and Windows.

## 2. Never Blindly Tone-Map 8-bit HLG Sources Before Sending to Seedance
Mobile/social downloads (e.g., X/Twitter or iPhone screen captures) often arrive as `8-bit yuv420p` carrying `bt2020nc / bt2020 / arib-std-b67` tags. Running `ffmpeg zscale=t=linear + tonemap=mobius/hable` before editing frequently clips cloud highlights and washes out midtones.

**Rule**:
- Feed the unmodified source MP4 directly to Seedance 2.5 Flash `edit`.
- Because the edit API returns an untagged MP4 (`color_space=unknown`), macOS QuickTime will display it flat (SDR) next to the original (HDR).
- `vidrep composite` automatically restores the exact source color profile on export.

## 3. Always Pair `zscale` Matrix Conversion with VUI Tags on Encode
Setting `-colorspace bt2020nc` alone in FFmpeg only writes stream metadata while still converting input `bgr24` to `yuv420p` using FFmpeg's default BT.601 matrix — causing a visible ~4–5 level saturation shift on round-trip playback.

`vidrep` automatically pairs matching `zscale` matrix conversion with stream VUI flags and `x264-params`:
- **BT.2020 / HLG**: `-vf zscale=matrix=2020_ncl:range=limited -colorspace bt2020nc -color_primaries bt2020 -color_trc arib-std-b67`
- **BT.2020 / PQ (10-bit)**: `-vf zscale=matrix=2020_ncl:range=limited -pix_fmt yuv420p10le -colorspace bt2020nc -color_primaries bt2020 -color_trc smpte2084`
- **Rec.709 / SDR**: `-vf zscale=matrix=709:range=limited -colorspace bt709 -color_primaries bt709 -color_trc bt709`
