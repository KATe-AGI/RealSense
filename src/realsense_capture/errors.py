class RealSenseCaptureError(RuntimeError):
    """Base error for RealSense capture failures."""


class DeviceNotFoundError(RealSenseCaptureError):
    """Raised when no RealSense device is detected."""


class DeviceBusyError(RealSenseCaptureError):
    """Raised when another process appears to own the camera."""


class StreamConfigurationError(RealSenseCaptureError):
    """Raised when the requested stream configuration cannot be started."""


class FrameCaptureError(RealSenseCaptureError):
    """Raised when a complete aligned frameset cannot be captured."""


class SampleSaveError(RealSenseCaptureError):
    """Raised when a captured sample cannot be saved completely."""
