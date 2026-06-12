from __future__ import annotations

from types import TracebackType
from typing import Any

import numpy as np
import pyrealsense2 as rs

from .config import CaptureConfig
from .errors import (
    DeviceBusyError,
    DeviceNotFoundError,
    FrameCaptureError,
    StreamConfigurationError,
)
from .models import CapturedFrames


def _camera_info(source: Any, info: rs.camera_info) -> str | None:
    if source.supports(info):
        return source.get_info(info)
    return None


def _intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def _frame_metadata(frame: rs.frame) -> dict[str, Any]:
    metadata = {
        "frame_number": int(frame.get_frame_number()),
        "timestamp_ms": float(frame.get_timestamp()),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
    }
    optional_fields = {
        "frame_timestamp_ms": rs.frame_metadata_value.frame_timestamp,
        "sensor_timestamp_ms": rs.frame_metadata_value.sensor_timestamp,
        "backend_timestamp_ms": rs.frame_metadata_value.backend_timestamp,
        "actual_exposure": rs.frame_metadata_value.actual_exposure,
        "gain_level": rs.frame_metadata_value.gain_level,
        "actual_fps": rs.frame_metadata_value.actual_fps,
    }

    for key, field in optional_fields.items():
        if frame.supports_frame_metadata(field):
            metadata[key] = frame.get_frame_metadata(field)
    return metadata


def _stream_info(frame: rs.video_frame) -> dict[str, Any]:
    profile = frame.get_profile().as_video_stream_profile()
    return {
        "width": profile.width(),
        "height": profile.height(),
        "format": str(profile.format()),
        "fps": profile.fps(),
        "intrinsics": _intrinsics_to_dict(profile.get_intrinsics()),
    }


def _build_rs_config(config: CaptureConfig) -> rs.config:
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.color,
        config.color_width,
        config.color_height,
        rs.format.rgb8,
        config.color_fps,
    )
    rs_config.enable_stream(
        rs.stream.depth,
        config.depth_width,
        config.depth_height,
        rs.format.z16,
        config.depth_fps,
    )
    return rs_config


def _filter_option_state(block: Any) -> dict[str, Any]:
    state = {}
    for option in block.get_supported_options():
        name = str(option)
        try:
            state[name] = block.get_option(option)
        except RuntimeError:
            state[name] = None
    return state


def _translate_runtime_error(error: RuntimeError) -> RuntimeError:
    message = str(error)
    if "No device connected" in message:
        return DeviceNotFoundError("No RealSense device detected. Please connect the camera and try again.")
    if "Device busy" in message or "failed to set power state" in message:
        return DeviceBusyError(
            "Failed to start the RealSense pipeline. Please close RealSense Viewer or other camera clients."
        )
    if "Couldn't resolve requests" in message:
        return StreamConfigurationError(
            "The requested RealSense stream configuration is not supported by the connected device."
        )
    if "Frame didn't arrive" in message:
        return FrameCaptureError(
            "No frames arrived before timeout. Try reducing FPS, lowering color/depth resolution, or closing other camera clients."
        )
    return error


class RealSenseCamera:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile: rs.pipeline_profile | None = None
        self.auto_exposure_state: list[dict[str, Any]] = []
        self.post_processing_blocks = self._build_post_processing_blocks()
        self.colorizer = self._build_colorizer()

    def __enter__(self) -> RealSenseCamera:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        try:
            self.profile = self.pipeline.start(_build_rs_config(self.config))
            self.auto_exposure_state = self.apply_auto_exposure()
        except RuntimeError as error:
            raise _translate_runtime_error(error) from error

    def stop(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None

    def apply_auto_exposure(self) -> list[dict[str, Any]]:
        device = self._device()
        requested_value = None
        if self.config.auto_exposure == "on":
            requested_value = 1.0
        elif self.config.auto_exposure == "off":
            requested_value = 0.0

        states = []
        for sensor in device.query_sensors():
            state = {
                "sensor": _camera_info(sensor, rs.camera_info.name),
                "supported": sensor.supports(rs.option.enable_auto_exposure),
                "requested": self.config.auto_exposure,
                "before": None,
                "after": None,
            }
            if state["supported"]:
                state["before"] = sensor.get_option(rs.option.enable_auto_exposure)
                if requested_value is not None:
                    sensor.set_option(rs.option.enable_auto_exposure, requested_value)
                state["after"] = sensor.get_option(rs.option.enable_auto_exposure)
            states.append(state)
        return states

    def capture_once(self) -> CapturedFrames:
        self.warmup()
        return self.capture_frame()

    def warmup(self) -> None:
        for _ in range(self.config.warmup_frames):
            self.capture_frame()

    def capture_frame(self) -> CapturedFrames:
        color_frame = None
        depth_frame = None
        processed_depth_frame = None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                processed_depth_frame = self._apply_post_processing(depth_frame)
        except RuntimeError as error:
            raise _translate_runtime_error(error) from error

        if not color_frame or not depth_frame or not processed_depth_frame:
            raise FrameCaptureError("Failed to retrieve both color and aligned depth frames.")

        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(processed_depth_frame.get_data()).copy()
        depth_visualization_image = np.asanyarray(self.colorizer.colorize(processed_depth_frame).get_data()).copy()

        device = self._device()
        depth_sensor = device.first_depth_sensor()
        device_info = {
            "device_name": _camera_info(device, rs.camera_info.name),
            "serial_number": _camera_info(device, rs.camera_info.serial_number),
            "firmware_version": _camera_info(device, rs.camera_info.firmware_version),
            "product_id": _camera_info(device, rs.camera_info.product_id),
            "usb_type_descriptor": _camera_info(device, rs.camera_info.usb_type_descriptor),
        }

        return CapturedFrames(
            color_image=color_image,
            depth_image=depth_image,
            depth_visualization_image=depth_visualization_image,
            device_info=device_info,
            depth_scale=depth_sensor.get_depth_scale(),
            settings={
                "auto_exposure": self.auto_exposure_state,
                "warmup_frames": self.config.warmup_frames,
                "stereo_module": self._stereo_module_settings(depth_sensor),
                "depth_visualization": self._depth_visualization_settings(),
                "post_processing_enabled": self.config.enable_post_processing,
                "post_processing": self._post_processing_settings(),
            },
            streams={
                "color": _stream_info(color_frame),
                "depth_aligned_to_color_raw": _stream_info(depth_frame),
                "depth_aligned_to_color": _stream_info(processed_depth_frame),
            },
            frames={
                "color": _frame_metadata(color_frame),
                "depth_raw": _frame_metadata(depth_frame),
                "depth": _frame_metadata(processed_depth_frame),
            },
        )

    def _device(self) -> rs.device:
        if self.profile is None:
            raise DeviceNotFoundError("RealSense pipeline is not started.")
        return self.profile.get_device()

    def _build_post_processing_blocks(self) -> list[tuple[str, Any]]:
        if not self.config.enable_post_processing:
            return []

        blocks = []
        if self.config.enable_decimation_filter:
            blocks.append(("decimation_filter", rs.decimation_filter()))
        if self.config.enable_rotation_filter:
            blocks.append(("rotation_filter", rs.rotation_filter()))
        if self.config.enable_hdr_merge:
            blocks.append(("hdr_merge", rs.hdr_merge()))
        if self.config.enable_sequence_id_filter:
            blocks.append(("sequence_id_filter", rs.sequence_id_filter()))
        if self.config.enable_threshold_filter:
            blocks.append(("threshold_filter", rs.threshold_filter()))
        if self.config.enable_depth_to_disparity:
            blocks.append(("depth_to_disparity", rs.disparity_transform(True)))
        if self.config.enable_spatial_filter:
            blocks.append(("spatial_filter", rs.spatial_filter()))
        if self.config.enable_temporal_filter:
            blocks.append(("temporal_filter", rs.temporal_filter()))
        if self.config.enable_hole_filling_filter:
            blocks.append(("hole_filling_filter", rs.hole_filling_filter()))
        if self.config.enable_disparity_to_depth:
            blocks.append(("disparity_to_depth", rs.disparity_transform(False)))
        return blocks

    def _build_colorizer(self) -> rs.colorizer:
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 0.0)
        colorizer.set_option(rs.option.color_scheme, 0.0)
        colorizer.set_option(
            rs.option.histogram_equalization_enabled,
            1.0 if self.config.stereo_histogram_equalization_enabled else 0.0,
        )
        return colorizer

    def _apply_post_processing(self, depth_frame: rs.depth_frame) -> rs.depth_frame:
        frame: Any = depth_frame
        for _, block in self.post_processing_blocks:
            frame = block.process(frame)
        processed_depth = frame.as_depth_frame()
        if not processed_depth:
            raise FrameCaptureError("Post-processing did not produce a depth frame.")
        return processed_depth

    def _stereo_module_settings(self, depth_sensor: rs.depth_sensor) -> dict[str, Any]:
        settings = {}
        for name, option in {
            "visual_preset": rs.option.visual_preset,
            "emitter_enabled": rs.option.emitter_enabled,
            "enable_auto_exposure": rs.option.enable_auto_exposure,
        }.items():
            if depth_sensor.supports(option):
                settings[name] = depth_sensor.get_option(option)
            else:
                settings[name] = None
        return settings

    def _depth_visualization_settings(self) -> dict[str, Any]:
        return {
            "visual_preset": self.config.stereo_depth_visual_preset,
            "visual_preset_value": self.colorizer.get_option(rs.option.visual_preset),
            "color_scheme": self.config.stereo_depth_color_scheme,
            "color_scheme_value": self.colorizer.get_option(rs.option.color_scheme),
            "histogram_equalization_enabled": self.colorizer.get_option(
                rs.option.histogram_equalization_enabled
            ),
        }

    def _post_processing_settings(self) -> list[dict[str, Any]]:
        enabled_names = {name for name, _ in self.post_processing_blocks}
        ordered = [
            ("decimation_filter", self.config.enable_post_processing and self.config.enable_decimation_filter),
            ("rotation_filter", self.config.enable_post_processing and self.config.enable_rotation_filter),
            ("hdr_merge", self.config.enable_post_processing and self.config.enable_hdr_merge),
            ("sequence_id_filter", self.config.enable_post_processing and self.config.enable_sequence_id_filter),
            ("threshold_filter", self.config.enable_post_processing and self.config.enable_threshold_filter),
            ("depth_to_disparity", self.config.enable_post_processing and self.config.enable_depth_to_disparity),
            ("spatial_filter", self.config.enable_post_processing and self.config.enable_spatial_filter),
            ("temporal_filter", self.config.enable_post_processing and self.config.enable_temporal_filter),
            ("hole_filling_filter", self.config.enable_post_processing and self.config.enable_hole_filling_filter),
            ("disparity_to_depth", self.config.enable_post_processing and self.config.enable_disparity_to_depth),
        ]
        block_map = dict(self.post_processing_blocks)
        return [
            {
                "name": name,
                "enabled": enabled,
                "applied": name in enabled_names,
                "options": _filter_option_state(block_map[name]) if name in block_map else {},
            }
            for name, enabled in ordered
        ]
