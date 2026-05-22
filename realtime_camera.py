# coding: utf-8

"""
Realtime webcam driving for LivePortrait human portraits.

The script keeps the source portrait cached and uses a camera/video stream as
the driving input. It is intentionally separate from inference.py so the
original offline pipeline stays unchanged.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Apple Silicon needs this for several torch ops used by the original project.
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

np = None
torch = None
CropConfig = None
InferenceConfig = None
LivePortraitWrapper = None
Cropper = None
get_rotation_matrix = None
crop_image_by_bbox = None
parse_bbox_from_landmark = None
paste_back = None
prepare_paste_back = None
calc_motion_multiplier = None
contiguous = None
load_image_rgb = None
resize_to_limit = None
log = print


def load_liveportrait_runtime() -> None:
    """Load heavier dependencies only when realtime inference is requested."""
    global np, torch
    global CropConfig, InferenceConfig, LivePortraitWrapper, Cropper
    global get_rotation_matrix, crop_image_by_bbox, parse_bbox_from_landmark, paste_back, prepare_paste_back
    global calc_motion_multiplier, contiguous, load_image_rgb, resize_to_limit, log

    import numpy as _np
    import torch as _torch

    from src.config.crop_config import CropConfig as _CropConfig
    from src.config.inference_config import InferenceConfig as _InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper as _LivePortraitWrapper
    from src.utils.camera import get_rotation_matrix as _get_rotation_matrix
    from src.utils.crop import (
        crop_image_by_bbox as _crop_image_by_bbox,
        parse_bbox_from_landmark as _parse_bbox_from_landmark,
        paste_back as _paste_back,
        prepare_paste_back as _prepare_paste_back,
    )
    from src.utils.helper import calc_motion_multiplier as _calc_motion_multiplier
    from src.utils.io import contiguous as _contiguous
    from src.utils.io import load_image_rgb as _load_image_rgb
    from src.utils.io import resize_to_limit as _resize_to_limit
    from src.utils.rprint import rlog as _log
    from src.utils.cropper import Cropper as _Cropper

    np = _np
    torch = _torch
    CropConfig = _CropConfig
    InferenceConfig = _InferenceConfig
    LivePortraitWrapper = _LivePortraitWrapper
    Cropper = _Cropper
    get_rotation_matrix = _get_rotation_matrix
    crop_image_by_bbox = _crop_image_by_bbox
    parse_bbox_from_landmark = _parse_bbox_from_landmark
    paste_back = _paste_back
    prepare_paste_back = _prepare_paste_back
    calc_motion_multiplier = _calc_motion_multiplier
    contiguous = _contiguous
    load_image_rgb = _load_image_rgb
    resize_to_limit = _resize_to_limit
    log = _log


def _clone_tensor_dct(dct: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in dct.items()
    }


def _parse_capture_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def _capture_backend():
    return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def list_cameras(limit: int) -> None:
    backend = _capture_backend()
    print("OpenCV camera indexes:")
    found = False
    for idx in range(limit):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            found = True
            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  {idx}: {width}x{height}, reported fps={fps:.1f}")
        cap.release()

    if not found:
        print("  No readable OpenCV cameras found.")

    if platform.system() == "Darwin":
        print("\nmacOS system camera report:")
        try:
            proc = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                text=True,
                capture_output=True,
                check=False,
                timeout=8,
            )
            report = proc.stdout.strip()
            print(report if report else "  system_profiler returned no camera entries.")
        except Exception as exc:
            print(f"  Could not read system camera report: {exc}")


def list_avfoundation_devices() -> int:
    """List AVFoundation devices via FFmpeg, including macOS Continuity Camera diagnostics."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        print("ffmpeg is not installed or not on PATH.")
        return 127
    except Exception as exc:
        print(f"Could not list AVFoundation devices: {exc}")
        return 1

    output = (proc.stdout or "") + (proc.stderr or "")
    print(output.strip())
    if "NSCameraUseContinuityCameraDeviceType" in output:
        print(
            "\nContinuity Camera note: macOS requires the host app to declare "
            "NSCameraUseContinuityCameraDeviceType. Terminal-launched Python/OpenCV "
            "may not be allowed to see iPhone Continuity Camera directly. Use a "
            "normal UVC/USB camera, OBS/Camo/EpocCam virtual camera, or an iPhone "
            "RTSP/MJPEG camera app and pass its stream URL to --camera."
        )
    return 0


def test_camera(args) -> int:
    """Open one camera source and keep it alive to trigger macOS permission prompts."""
    cap = open_capture(args)
    window_name = f"Camera test: {args.camera}"
    print("Camera opened. If macOS asks for permission, allow Terminal/iTerm/Codex to access Camera.")
    print("Keys: q quit, ESC quit.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Camera opened, but no frames are arriving.")
                return 1
            if args.mirror_input:
                frame_bgr = cv2.flip(frame_bgr, 1)
            cv2.putText(
                frame_bgr,
                "Camera test | q quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (35, 255, 35),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def ensure_weights_exist(inference_cfg: InferenceConfig, crop_cfg: CropConfig) -> None:
    required = [
        inference_cfg.checkpoint_F,
        inference_cfg.checkpoint_M,
        inference_cfg.checkpoint_G,
        inference_cfg.checkpoint_W,
        crop_cfg.landmark_ckpt_path,
        crop_cfg.insightface_root,
    ]
    if (
        inference_cfg.flag_stitching
        or inference_cfg.flag_eye_retargeting
        or inference_cfg.flag_lip_retargeting
        or inference_cfg.flag_normalize_lip
    ):
        required.append(inference_cfg.checkpoint_S)
    missing = [path for path in required if path and not Path(path).exists()]
    if not missing:
        return

    lines = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "Missing pretrained weights or face-analysis assets:\n"
        f"{lines}\n\n"
        "Download the official weights first:\n"
        '  huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"\n'
        "If Hugging Face is unavailable, use HF_ENDPOINT=https://hf-mirror.com as described in readme_zh_cn.md."
    )


@dataclass
class SourceState:
    image_rgb: np.ndarray
    crop_rgb_256: np.ndarray
    source_lmk: Optional[np.ndarray]
    M_c2o: Optional[np.ndarray]
    mask_ori_float: Optional[np.ndarray]
    x_s_info: dict
    x_c_s: torch.Tensor
    R_s: torch.Tensor
    f_s: torch.Tensor
    x_s: torch.Tensor
    lip_delta_before_animation: Optional[torch.Tensor]


class DrivingFrameCropper:
    """Realtime variant of Cropper.crop_driving_video for one frame at a time."""

    def __init__(
        self,
        cropper: Cropper,
        crop_cfg: CropConfig,
        redetect_interval: int,
        bbox_smoothing: float,
        crop_mode: str,
        center_crop_ratio: float,
    ) -> None:
        self.cropper = cropper
        self.crop_cfg = crop_cfg
        self.redetect_interval = max(1, redetect_interval)
        self.bbox_smoothing = min(max(bbox_smoothing, 0.0), 0.98)
        self.crop_mode = crop_mode
        self.center_crop_ratio = min(max(center_crop_ratio, 0.1), 1.0)
        self.prev_lmk: Optional[np.ndarray] = None
        self.smoothed_bbox: Optional[np.ndarray] = None
        self.frame_idx = 0

    def reset(self) -> None:
        self.prev_lmk = None
        self.smoothed_bbox = None
        self.frame_idx = 0

    def _detect_lmk(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        faces = self.cropper.face_analysis_wrapper.get(
            contiguous(frame_rgb[..., ::-1]),
            flag_do_landmark_2d_106=True,
            direction=self.crop_cfg.direction,
        )
        if len(faces) == 0:
            return None
        if len(faces) > 1:
            log(f"More than one driving face detected, pick one by rule {self.crop_cfg.direction}.")
        lmk = faces[0].landmark_2d_106
        return self.cropper.human_landmark_runner.run(frame_rgb, lmk)

    def _track_lmk(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.prev_lmk is None:
            return None
        try:
            return self.cropper.human_landmark_runner.run(frame_rgb, self.prev_lmk)
        except Exception:
            return None

    def _crop_by_bbox(self, frame_rgb: np.ndarray, bbox, lmk=None):
        ret_dct = crop_image_by_bbox(
            frame_rgb,
            bbox.tolist() if hasattr(bbox, "tolist") else bbox,
            lmk=lmk,
            dsize=self.crop_cfg.dsize,
            flag_rot=False,
            borderValue=(0, 0, 0),
        )
        crop_rgb = ret_dct["img_crop"]
        crop_rgb_256 = cv2.resize(crop_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        return crop_rgb_256, ret_dct["lmk_crop"]

    def _crop_center(self, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        size = max(16, int(min(h, w) * self.center_crop_ratio))
        cx, cy = w / 2.0, h / 2.0
        bbox = np.array(
            [cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0],
            dtype=np.float32,
        )
        return self._crop_by_bbox(frame_rgb, bbox, None)

    def crop(self, frame_rgb: np.ndarray):
        if self.crop_mode == "center":
            return self._crop_center(frame_rgb)
        if self.crop_mode == "static" and self.smoothed_bbox is not None:
            return self._crop_by_bbox(frame_rgb, self.smoothed_bbox, None)

        should_redetect = self.prev_lmk is None or self.frame_idx % self.redetect_interval == 0
        lmk = self._detect_lmk(frame_rgb) if should_redetect else self._track_lmk(frame_rgb)
        if lmk is None and not should_redetect:
            lmk = self._detect_lmk(frame_rgb)
        if lmk is None:
            self.prev_lmk = None
            self.frame_idx += 1
            return None

        self.prev_lmk = lmk
        self.frame_idx += 1

        ret_bbox = parse_bbox_from_landmark(
            lmk,
            scale=self.crop_cfg.scale_crop_driving_video,
            vx_ratio_crop_driving_video=self.crop_cfg.vx_ratio_crop_driving_video,
            vy_ratio=self.crop_cfg.vy_ratio_crop_driving_video,
        )["bbox"]
        bbox = np.array(
            [ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1]],
            dtype=np.float32,
        )
        if self.smoothed_bbox is not None and self.bbox_smoothing > 0:
            bbox = self.bbox_smoothing * self.smoothed_bbox + (1.0 - self.bbox_smoothing) * bbox
        self.smoothed_bbox = bbox

        return self._crop_by_bbox(frame_rgb, bbox, lmk)


class RealtimePortraitAnimator:
    def __init__(
        self,
        wrapper: LivePortraitWrapper,
        source: SourceState,
        inference_cfg: InferenceConfig,
        motion_smoothing: float,
    ) -> None:
        self.wrapper = wrapper
        self.source = source
        self.cfg = inference_cfg
        self.motion_smoothing = min(max(motion_smoothing, 0.0), 0.98)
        self.anchor_info: Optional[dict] = None
        self.anchor_R: Optional[torch.Tensor] = None
        self.anchor_x_d_new: Optional[torch.Tensor] = None
        self.motion_multiplier: float = 1.0
        self.prev_x_d_new: Optional[torch.Tensor] = None

    def reset_anchor(self) -> None:
        self.anchor_info = None
        self.anchor_R = None
        self.anchor_x_d_new = None
        self.motion_multiplier = 1.0
        self.prev_x_d_new = None

    def _kp_info_from_driving_crop(self, driving_crop_rgb_256: np.ndarray):
        I_d = self.wrapper.prepare_source(driving_crop_rgb_256)
        x_d_i_info = self.wrapper.get_kp_info(I_d)
        R_d_i = get_rotation_matrix(x_d_i_info["pitch"], x_d_i_info["yaw"], x_d_i_info["roll"])
        if self.anchor_info is None:
            self.anchor_info = _clone_tensor_dct(x_d_i_info)
            self.anchor_R = R_d_i.clone()
        return x_d_i_info, R_d_i

    def _build_driving_keypoints(self, x_d_i_info: dict, R_d_i: torch.Tensor) -> torch.Tensor:
        src = self.source
        cfg = self.cfg
        anchor = self.anchor_info
        assert anchor is not None and self.anchor_R is not None

        delta_new = src.x_s_info["exp"].clone()
        if cfg.flag_relative_motion:
            if cfg.animation_region in ("all", "pose"):
                R_new = (R_d_i @ self.anchor_R.permute(0, 2, 1)) @ src.R_s
            else:
                R_new = src.R_s

            if cfg.animation_region in ("all", "exp"):
                delta_new = src.x_s_info["exp"] + (x_d_i_info["exp"] - anchor["exp"])
            elif cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (
                        src.x_s_info["exp"] + (x_d_i_info["exp"] - anchor["exp"])
                    )[:, lip_idx, :]
            elif cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = (
                        src.x_s_info["exp"] + (x_d_i_info["exp"] - anchor["exp"])
                    )[:, eyes_idx, :]

            if cfg.animation_region == "all":
                scale_new = src.x_s_info["scale"] * (x_d_i_info["scale"] / anchor["scale"])
            else:
                scale_new = src.x_s_info["scale"]

            if cfg.animation_region in ("all", "pose"):
                t_new = src.x_s_info["t"] + (x_d_i_info["t"] - anchor["t"])
            else:
                t_new = src.x_s_info["t"]
        else:
            if cfg.animation_region in ("all", "pose"):
                R_new = R_d_i
            else:
                R_new = src.R_s

            if cfg.animation_region in ("all", "exp"):
                for idx in [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                    delta_new[:, idx, :] = x_d_i_info["exp"][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info["exp"][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info["exp"][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info["exp"][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info["exp"][:, 9, 1:]
            elif cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info["exp"][:, lip_idx, :]
            elif cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_i_info["exp"][:, eyes_idx, :]

            scale_new = src.x_s_info["scale"]
            t_new = x_d_i_info["t"] if cfg.animation_region in ("all", "pose") else src.x_s_info["t"]

        t_new = t_new.clone()
        t_new[..., 2].fill_(0)
        x_d_i_new = scale_new * (src.x_c_s @ R_new + delta_new) + t_new

        if cfg.flag_relative_motion and cfg.driving_option == "expression-friendly":
            if self.anchor_x_d_new is None:
                self.anchor_x_d_new = x_d_i_new.clone()
                self.motion_multiplier = calc_motion_multiplier(src.x_s, self.anchor_x_d_new)
            x_d_i_new = (x_d_i_new - self.anchor_x_d_new) * self.motion_multiplier + src.x_s

        return x_d_i_new

    def animate(self, driving_crop_rgb_256: np.ndarray, driving_lmk_crop: Optional[np.ndarray]):
        src = self.source
        cfg = self.cfg
        x_d_i_info, R_d_i = self._kp_info_from_driving_crop(driving_crop_rgb_256)
        x_d_i_new = self._build_driving_keypoints(x_d_i_info, R_d_i)

        if not cfg.flag_stitching and not cfg.flag_eye_retargeting and not cfg.flag_lip_retargeting:
            if cfg.flag_normalize_lip and src.lip_delta_before_animation is not None:
                x_d_i_new += src.lip_delta_before_animation
        elif cfg.flag_stitching and not cfg.flag_eye_retargeting and not cfg.flag_lip_retargeting:
            if cfg.flag_normalize_lip and src.lip_delta_before_animation is not None:
                x_d_i_new = self.wrapper.stitching(src.x_s, x_d_i_new) + src.lip_delta_before_animation
            else:
                x_d_i_new = self.wrapper.stitching(src.x_s, x_d_i_new)
        else:
            eyes_delta, lip_delta = None, None
            if cfg.flag_eye_retargeting and src.source_lmk is not None and driving_lmk_crop is not None:
                c_d_eyes_lst, _ = self.wrapper.calc_ratio([driving_lmk_crop])
                combined_eye_ratio_tensor = self.wrapper.calc_combined_eye_ratio(c_d_eyes_lst[0], src.source_lmk)
                eyes_delta = self.wrapper.retarget_eye(src.x_s, combined_eye_ratio_tensor)
            if cfg.flag_lip_retargeting and src.source_lmk is not None and driving_lmk_crop is not None:
                _, c_d_lip_lst = self.wrapper.calc_ratio([driving_lmk_crop])
                combined_lip_ratio_tensor = self.wrapper.calc_combined_lip_ratio(c_d_lip_lst[0], src.source_lmk)
                lip_delta = self.wrapper.retarget_lip(src.x_s, combined_lip_ratio_tensor)

            if cfg.flag_relative_motion:
                x_d_i_new = src.x_s + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
            else:
                x_d_i_new = x_d_i_new + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
            if cfg.flag_stitching:
                x_d_i_new = self.wrapper.stitching(src.x_s, x_d_i_new)

        x_d_i_new = src.x_s + (x_d_i_new - src.x_s) * cfg.driving_multiplier
        if self.prev_x_d_new is not None and self.motion_smoothing > 0:
            x_d_i_new = self.motion_smoothing * self.prev_x_d_new + (1.0 - self.motion_smoothing) * x_d_i_new
        self.prev_x_d_new = x_d_i_new.detach().clone()

        out = self.wrapper.warp_decode(src.f_s, src.x_s, x_d_i_new)
        generated_rgb = self.wrapper.parse_output(out["out"])[0]
        if cfg.flag_pasteback and src.M_c2o is not None and src.mask_ori_float is not None:
            generated_rgb = paste_back(generated_rgb, src.M_c2o, src.image_rgb, src.mask_ori_float)
        return generated_rgb


def prepare_source_state(
    source_path: str,
    wrapper: LivePortraitWrapper,
    cropper: Cropper,
    inference_cfg: InferenceConfig,
    crop_cfg: CropConfig,
) -> SourceState:
    img_rgb = load_image_rgb(source_path)
    img_rgb = resize_to_limit(img_rgb, inference_cfg.source_max_dim, inference_cfg.source_division)

    crop_info = None
    source_lmk = None
    M_c2o = None
    mask_ori_float = None
    if inference_cfg.flag_do_crop:
        crop_info = cropper.crop_source_image(img_rgb, crop_cfg)
        if crop_info is None:
            raise RuntimeError("No face detected in the source portrait.")
        source_lmk = crop_info["lmk_crop"]
        img_crop_256 = crop_info["img_crop_256x256"]
        M_c2o = crop_info["M_c2o"]
        if inference_cfg.flag_pasteback and inference_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(
                inference_cfg.mask_crop,
                M_c2o,
                dsize=(img_rgb.shape[1], img_rgb.shape[0]),
            )
    else:
        source_lmk = cropper.calc_lmk_from_cropped_image(img_rgb)
        img_crop_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    I_s = wrapper.prepare_source(img_crop_256)
    x_s_info = wrapper.get_kp_info(I_s)
    x_c_s = x_s_info["kp"]
    R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s = wrapper.extract_feature_3d(I_s)
    x_s = wrapper.transform_keypoint(x_s_info)

    lip_delta_before_animation = None
    if inference_cfg.flag_normalize_lip and inference_cfg.flag_relative_motion and source_lmk is not None:
        combined_lip_ratio_tensor = wrapper.calc_combined_lip_ratio([0.0], source_lmk)
        if combined_lip_ratio_tensor[0][0] >= inference_cfg.lip_normalize_threshold:
            lip_delta_before_animation = wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

    return SourceState(
        image_rgb=img_rgb,
        crop_rgb_256=img_crop_256,
        source_lmk=source_lmk,
        M_c2o=M_c2o,
        mask_ori_float=mask_ori_float,
        x_s_info=x_s_info,
        x_c_s=x_c_s,
        R_s=R_s,
        f_s=f_s,
        x_s=x_s,
        lip_delta_before_animation=lip_delta_before_animation,
    )


def build_preview(
    display_mode: str,
    driving_crop_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    if display_mode == "generated":
        preview = generated_rgb.copy()
    else:
        target_h = generated_rgb.shape[0]
        target_w = generated_rgb.shape[1]
        driving_resized = cv2.resize(driving_crop_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        preview = np.concatenate([driving_resized, generated_rgb], axis=1)

    preview_bgr = preview[..., ::-1].copy()
    cv2.putText(
        preview_bgr,
        f"FPS {fps:4.1f} | {latency_ms:4.0f} ms | q quit | r reset neutral",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (35, 255, 35),
        2,
        cv2.LINE_AA,
    )
    return preview_bgr


def open_capture(args) -> cv2.VideoCapture:
    source = _parse_capture_source(args.camera)
    backend = _capture_backend() if isinstance(source, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened() and isinstance(source, int) and backend != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        hint = (
            f"Cannot open camera/video source: {args.camera}\n"
            "Run `python realtime_camera.py --list-avfoundation` for macOS device diagnostics. "
            "If you are trying to use iPhone Continuity Camera from Terminal, macOS may hide it "
            "unless the host app declares NSCameraUseContinuityCameraDeviceType. A USB/UVC camera, "
            "OBS/Camo/EpocCam virtual camera, or an iPhone RTSP/MJPEG stream URL is more reliable."
        )
        raise RuntimeError(hint)

    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


class OptionalVirtualCamera:
    def __init__(self, enabled: bool, fps: int) -> None:
        self.enabled = enabled
        self.fps = fps
        self.cam = None

    def send(self, frame_rgb: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.cam is None:
            try:
                import pyvirtualcam
            except ImportError as exc:
                raise RuntimeError("Install pyvirtualcam to use --virtual-camera: pip install pyvirtualcam") from exc
            height, width = frame_rgb.shape[:2]
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=self.fps)
            print(f"Virtual camera started: {self.cam.device} ({width}x{height}@{self.fps})")
        self.cam.send(np.ascontiguousarray(frame_rgb))
        self.cam.sleep_until_next_frame()

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close()
            self.cam = None


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime LivePortrait webcam driving.")
    parser.add_argument("-s", "--source", help="source portrait image path for the generated character")
    parser.add_argument("--camera", default="0", help="camera index, video path, or stream URL (default: 0)")
    parser.add_argument("--list-cameras", action="store_true", help="list readable camera indexes and exit")
    parser.add_argument("--list-avfoundation", action="store_true", help="list macOS AVFoundation devices via ffmpeg")
    parser.add_argument("--test-camera", action="store_true", help="open a camera preview and trigger macOS camera permission")
    parser.add_argument("--camera-scan-limit", type=int, default=8)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--target-fps", type=float, default=0.0, help="cap realtime processing FPS; 0 means run as fast as possible")
    parser.add_argument("--mirror-input", action="store_true", help="mirror camera frames before driving")
    parser.add_argument("--mirror-output", action="store_true", help="mirror generated frames before preview/recording/virtual camera")

    parser.add_argument("--display", choices=["split", "generated"], default="split")
    parser.add_argument("--no-window", action="store_true", help="do not open an OpenCV preview window")
    parser.add_argument("--output", help="optional path to record generated video")
    parser.add_argument("--output-fps", type=int, default=25)
    parser.add_argument("--virtual-camera", action="store_true", help="send generated RGB frames to pyvirtualcam")

    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--no-half", action="store_true", help="disable half precision")
    parser.add_argument("--no-source-crop", action="store_true")
    parser.add_argument("--pasteback", action="store_true", help="paste generated head back to the original source image")
    parser.add_argument("--no-stitching", action="store_true")
    parser.add_argument("--normalize-lip", action="store_true", help="try to close source mouth before animation")
    parser.add_argument("--eye-retargeting", action="store_true")
    parser.add_argument("--lip-retargeting", action="store_true")
    parser.add_argument("--no-relative-motion", action="store_true")
    parser.add_argument("--driving-option", choices=["expression-friendly", "pose-friendly"], default="expression-friendly")
    parser.add_argument("--driving-multiplier", type=float, default=1.0)
    parser.add_argument("--animation-region", choices=["exp", "pose", "lip", "eyes", "all"], default="all")

    parser.add_argument("--source-scale", type=float, default=2.3)
    parser.add_argument("--source-vx", type=float, default=0.0)
    parser.add_argument("--source-vy", type=float, default=-0.125)
    parser.add_argument("--driving-scale", type=float, default=2.2)
    parser.add_argument("--driving-vx", type=float, default=0.0)
    parser.add_argument("--driving-vy", type=float, default=-0.1)
    parser.add_argument(
        "--driving-crop-mode",
        choices=["landmark", "static", "center"],
        default="landmark",
        help="landmark tracks every frame; static reuses the first detected crop; center skips face tracking",
    )
    parser.add_argument("--center-crop-ratio", type=float, default=0.78, help="center crop size relative to min camera dimension")
    parser.add_argument("--redetect-interval", type=int, default=30)
    parser.add_argument("--crop-smoothing", type=float, default=0.65)
    parser.add_argument("--motion-smoothing", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cameras:
        list_cameras(args.camera_scan_limit)
        return 0
    if args.list_avfoundation:
        return list_avfoundation_devices()
    if args.test_camera:
        return test_camera(args)
    if not args.source:
        print("Missing --source. Provide a source portrait image, or use --list-cameras.", file=sys.stderr)
        return 2
    if not Path(args.source).exists():
        print(f"Source portrait not found: {args.source}", file=sys.stderr)
        return 2
    if args.pasteback and args.no_source_crop:
        print("--pasteback requires source cropping; remove --no-source-crop.", file=sys.stderr)
        return 2

    load_liveportrait_runtime()

    inference_cfg = InferenceConfig(
        flag_force_cpu=args.force_cpu,
        device_id=args.device_id,
        flag_use_half_precision=not args.no_half,
        flag_normalize_lip=args.normalize_lip,
        flag_eye_retargeting=args.eye_retargeting,
        flag_lip_retargeting=args.lip_retargeting,
        flag_stitching=not args.no_stitching,
        flag_relative_motion=not args.no_relative_motion,
        flag_pasteback=args.pasteback,
        flag_do_crop=not args.no_source_crop,
        driving_option=args.driving_option,
        driving_multiplier=args.driving_multiplier,
        animation_region=args.animation_region,
        output_fps=args.output_fps,
    )
    crop_cfg = CropConfig(
        device_id=args.device_id,
        flag_force_cpu=args.force_cpu,
        scale=args.source_scale,
        vx_ratio=args.source_vx,
        vy_ratio=args.source_vy,
        scale_crop_driving_video=args.driving_scale,
        vx_ratio_crop_driving_video=args.driving_vx,
        vy_ratio_crop_driving_video=args.driving_vy,
    )
    ensure_weights_exist(inference_cfg, crop_cfg)

    log("Loading LivePortrait models...")
    wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
    cropper = Cropper(crop_cfg=crop_cfg, flag_force_cpu=args.force_cpu, device_id=args.device_id)
    source_state = prepare_source_state(args.source, wrapper, cropper, inference_cfg, crop_cfg)
    if args.driving_crop_mode != "landmark" and (args.eye_retargeting or args.lip_retargeting):
        print("Warning: eye/lip retargeting needs driving landmarks; use --driving-crop-mode landmark for those modes.")
    driving_cropper = DrivingFrameCropper(
        cropper,
        crop_cfg,
        args.redetect_interval,
        args.crop_smoothing,
        args.driving_crop_mode,
        args.center_crop_ratio,
    )
    animator = RealtimePortraitAnimator(wrapper, source_state, inference_cfg, args.motion_smoothing)

    cap = open_capture(args)
    writer = None
    virtual_cam = OptionalVirtualCamera(args.virtual_camera, args.output_fps)
    frame_count = 0
    fps = 0.0
    fps_t0 = time.perf_counter()
    window_name = "LivePortrait Realtime"

    print("Started realtime driving. Keep a neutral frontal face for the first detected frame.")
    print("Keys: q quit, r reset neutral anchor.")

    try:
        while True:
            loop_t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("No frame received from camera/video source.")
                break
            if args.mirror_input:
                frame_bgr = cv2.flip(frame_bgr, 1)

            t0 = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cropped = driving_cropper.crop(frame_rgb)
            if cropped is None:
                if not args.no_window:
                    display = frame_bgr.copy()
                    cv2.putText(display, "No driving face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow(window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        driving_cropper.reset()
                        animator.reset_anchor()
                continue

            driving_crop_rgb_256, driving_lmk_crop = cropped
            generated_rgb = animator.animate(driving_crop_rgb_256, driving_lmk_crop)
            if args.mirror_output:
                generated_rgb = cv2.flip(generated_rgb, 1)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            frame_count += 1
            now = time.perf_counter()
            if now - fps_t0 >= 0.5:
                fps = frame_count / (now - fps_t0)
                frame_count = 0
                fps_t0 = now

            if args.output:
                if writer is None:
                    height, width = generated_rgb.shape[:2]
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, args.output_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot open output writer: {args.output}")
                writer.write(generated_rgb[..., ::-1])

            virtual_cam.send(generated_rgb)

            if not args.no_window:
                preview_bgr = build_preview(args.display, driving_crop_rgb_256, generated_rgb, fps, latency_ms)
                cv2.imshow(window_name, preview_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    driving_cropper.reset()
                    animator.reset_anchor()
                    print("Neutral anchor reset. Keep a neutral frontal face for the next detected frame.")
            elif frame_count == 1:
                print(f"FPS {fps:.1f}, latency {latency_ms:.0f} ms")

            if args.target_fps > 0:
                target_frame_seconds = 1.0 / args.target_fps
                remaining = target_frame_seconds - (time.perf_counter() - loop_t0)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        virtual_cam.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
