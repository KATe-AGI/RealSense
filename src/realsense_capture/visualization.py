import cv2
import numpy as np


_PREVIEW_WINDOWS: set[str] = set()
_PREVIEW_STATES: dict[str, dict] = {}


class PointCloudViewState:
    def __init__(self) -> None:
        self.pitch = np.radians(-10.0)
        self.yaw = np.radians(-15.0)
        self.translation = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.distance = 2.0
        self.prev_mouse = (0, 0)
        self.mouse_buttons = {1: False, 2: False}

    def reset(self) -> None:
        self.pitch = 0.0
        self.yaw = 0.0
        self.translation[:] = (0.0, 0.0, -1.0)
        self.distance = 2.0

    @property
    def rotation(self) -> np.ndarray:
        rx, _ = cv2.Rodrigues(np.array([self.pitch, 0.0, 0.0], dtype=np.float32))
        ry, _ = cv2.Rodrigues(np.array([0.0, self.yaw, 0.0], dtype=np.float32))
        return np.dot(ry, rx).astype(np.float32)

    @property
    def pivot(self) -> np.ndarray:
        return self.translation + np.array([0.0, 0.0, self.distance], dtype=np.float32)


def _depth_intrinsics(streams: dict, depth_shape: tuple[int, int]) -> dict:
    stream = streams.get("depth_aligned_to_color", {})
    intrinsics = stream.get("intrinsics")
    if intrinsics:
        return intrinsics

    height, width = depth_shape
    focal = float(max(width, height))
    return {
        "fx": focal,
        "fy": focal,
        "ppx": (width - 1) / 2.0,
        "ppy": (height - 1) / 2.0,
    }


def _view_points(points: np.ndarray, view_state: PointCloudViewState) -> np.ndarray:
    pivot = view_state.pivot
    return np.dot(points - pivot, view_state.rotation) + pivot - view_state.translation


def _project_points(points: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    view_aspect = output_height / output_width
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = points[:, :2] / points[:, 2, np.newaxis]
        projected = projected * (output_width * view_aspect, output_height) + (
            output_width / 2.0,
            output_height / 2.0,
        )
    projected[points[:, 2] < 0.03] = np.nan
    return projected


def _line_3d(
    canvas: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    height, width = canvas.shape[:2]
    projected = _project_points(np.asarray([point_a, point_b], dtype=np.float32), width, height)
    if np.isnan(projected).any():
        return

    p0 = tuple(projected[0].astype(int))
    p1 = tuple(projected[1].astype(int))
    inside, p0, p1 = cv2.clipLine((0, 0, width, height), p0, p1)
    if inside:
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_grid(
    canvas: np.ndarray,
    view_state: PointCloudViewState,
    pos: tuple[float, float, float] = (0.0, 0.5, 1.0),
    size: float = 1.0,
    divisions: int = 10,
    color: tuple[int, int, int] = (128, 128, 128),
) -> None:
    pos_array = np.asarray(pos, dtype=np.float32)
    step = size / float(divisions)
    half = 0.5 * size
    rotation = np.eye(3, dtype=np.float32)
    for index in range(divisions + 1):
        x = -half + index * step
        z = -half + index * step
        _line_3d(
            canvas,
            _view_points((pos_array + np.dot((x, 0, -half), rotation)).reshape(1, 3), view_state)[0],
            _view_points((pos_array + np.dot((x, 0, half), rotation)).reshape(1, 3), view_state)[0],
            color,
        )
        _line_3d(
            canvas,
            _view_points((pos_array + np.dot((-half, 0, z), rotation)).reshape(1, 3), view_state)[0],
            _view_points((pos_array + np.dot((half, 0, z), rotation)).reshape(1, 3), view_state)[0],
            color,
        )


def _draw_axes(
    canvas: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray | None = None,
    size: float = 0.1,
    thickness: int = 1,
) -> None:
    if rotation is None:
        rotation = np.eye(3, dtype=np.float32)
    _line_3d(canvas, origin, origin + np.dot((size, 0, 0), rotation), (255, 0, 0), thickness)
    _line_3d(canvas, origin, origin + np.dot((0, size, 0), rotation), (0, 255, 0), thickness)
    _line_3d(canvas, origin, origin + np.dot((0, 0, size), rotation), (0, 0, 255), thickness)


def _deproject_pixel(intrinsics: dict, pixel: tuple[float, float], depth_m: float) -> np.ndarray:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    x = (float(pixel[0]) - ppx) / fx * depth_m
    y = (float(pixel[1]) - ppy) / fy * depth_m
    return np.asarray([x, y, depth_m], dtype=np.float32)


def _draw_frustum(
    canvas: np.ndarray,
    intrinsics: dict,
    view_state: PointCloudViewState,
    color: tuple[int, int, int] = (64, 64, 64),
) -> None:
    origin = _view_points(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), view_state)[0]
    width = float(intrinsics.get("width", 0))
    height = float(intrinsics.get("height", 0))
    if width <= 0 or height <= 0:
        return

    for depth_m in range(1, 6, 2):
        corners = [
            _deproject_pixel(intrinsics, (0, 0), float(depth_m)),
            _deproject_pixel(intrinsics, (width, 0), float(depth_m)),
            _deproject_pixel(intrinsics, (width, height), float(depth_m)),
            _deproject_pixel(intrinsics, (0, height), float(depth_m)),
        ]
        viewed_corners = _view_points(np.asarray(corners, dtype=np.float32), view_state)
        for corner in viewed_corners:
            _line_3d(canvas, origin, corner, color)
        for index in range(4):
            _line_3d(canvas, viewed_corners[index], viewed_corners[(index + 1) % 4], color)


def create_depth_visualization(depth_image: np.ndarray) -> np.ndarray:
    valid_depth = depth_image[depth_image > 0]
    scaled = np.zeros(depth_image.shape, dtype=np.uint8)

    if valid_depth.size:
        lower = float(np.percentile(valid_depth, 1))
        upper = float(np.percentile(valid_depth, 99))
        if upper <= lower:
            upper = lower + 1.0

        clipped = np.clip(depth_image.astype(np.float32), lower, upper)
        scaled = ((clipped - lower) / (upper - lower) * 255.0).astype(np.uint8)
        scaled[depth_image == 0] = 0

    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colored[depth_image == 0] = 0
    return colored


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width == target_width:
        return image
    target_height = max(1, round(height * target_width / width))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def compose_preview(
    color_image: np.ndarray,
    depth_visualization_image: np.ndarray,
    max_width: int,
    max_height: int,
    pointcloud_image: np.ndarray | None = None,
) -> np.ndarray:
    images = [_ensure_rgb(color_image), _ensure_rgb(depth_visualization_image)]
    if pointcloud_image is not None:
        images.append(_ensure_rgb(pointcloud_image))

    aspects = [image.shape[1] / image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]
    width_limited_by_height = max_height / sum(1 / aspect for aspect in aspects)
    target_width = max(
        1,
        round(min(max_width, *widths, width_limited_by_height)),
    )

    previews = [_resize_to_width(image, target_width) for image in images]
    return np.vstack(previews)


def _resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    if target_width == width and target_height == height:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _stream_aspect(streams: dict, fallback_shape: tuple[int, int]) -> float:
    stream = streams.get("depth_aligned_to_color", {})
    intrinsics = stream.get("intrinsics", {})
    width = int(intrinsics.get("width") or stream.get("width") or fallback_shape[1])
    height = int(intrinsics.get("height") or stream.get("height") or fallback_shape[0])
    if width <= 0 or height <= 0:
        return fallback_shape[1] / fallback_shape[0]
    return width / height


def _pointcloud_preview_size(captured, max_width: int, max_height: int) -> tuple[int, int]:
    aspect = _stream_aspect(captured.streams, captured.depth_image.shape[:2])
    width = max_width
    height = max(1, round(width / aspect))
    if height > max_height:
        height = max_height
        width = max(1, round(height * aspect))
    return width, height


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _preview_state(window_name: str) -> dict:
    if window_name not in _PREVIEW_STATES:
        _PREVIEW_STATES[window_name] = {
            "buttons": {},
            "closed": False,
            "fullscreen": None,
            "minimized": set(),
            "panel_rects": {},
            "pointcloud_view": PointCloudViewState(),
        }
    return _PREVIEW_STATES[window_name]


def _inside_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def _preview_mouse_callback(event: int, x: int, y: int, flags: int, userdata) -> None:
    state = userdata
    view = state["pointcloud_view"]
    if event == cv2.EVENT_LBUTTONUP:
        view.mouse_buttons[1] = False
        for panel_name, panel_buttons in state["buttons"].items():
            for action, rect in panel_buttons.items():
                if not _inside_rect(x, y, rect):
                    continue
                if action == "close":
                    state["closed"] = True
                elif action == "minimize":
                    if panel_name in state["minimized"]:
                        state["minimized"].remove(panel_name)
                    else:
                        state["minimized"].add(panel_name)
                        if state["fullscreen"] == panel_name:
                            state["fullscreen"] = None
                elif action == "fullscreen":
                    state["fullscreen"] = None if state["fullscreen"] == panel_name else panel_name
                    state["minimized"].discard(panel_name)
                return

    pointcloud_rect = state.get("panel_rects", {}).get("pointcloud")
    if pointcloud_rect is None or not _inside_rect(x, y, pointcloud_rect):
        if event == cv2.EVENT_RBUTTONUP:
            view.mouse_buttons[2] = False
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        view.mouse_buttons[1] = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        view.mouse_buttons[2] = True
    elif event == cv2.EVENT_RBUTTONUP:
        view.mouse_buttons[2] = False
    elif event == cv2.EVENT_MOUSEMOVE:
        dx = x - view.prev_mouse[0]
        dy = y - view.prev_mouse[1]
        _, _, width, height = pointcloud_rect
        if view.mouse_buttons[1]:
            view.yaw += float(dx) / max(1, width) * 2.0
            view.pitch -= float(dy) / max(1, height) * 2.0
        elif view.mouse_buttons[2]:
            delta = np.array((dx / max(1, width), dy / max(1, height), 0.0), dtype=np.float32)
            view.translation -= np.dot(view.rotation, delta)
    elif event == cv2.EVENT_MOUSEWHEEL:
        dz = 0.1 if flags > 0 else -0.1
        view.translation[2] += dz
        view.distance -= dz

    view.prev_mouse = (x, y)


def _draw_button(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    danger: bool = False,
) -> None:
    x, y, width, height = rect
    fill = (60, 70, 80) if not danger else (70, 45, 45)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (115, 125, 135), 1)
    cv2.putText(
        canvas,
        label,
        (x + 6, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 235, 240),
        1,
        cv2.LINE_AA,
    )


def _draw_panel(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    image_rgb: np.ndarray,
    minimized: bool,
) -> dict[str, tuple[int, int, int, int]]:
    x, y, width, height = rect
    title_height = 30
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (24, 28, 32), -1)
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + title_height), (42, 48, 54), -1)
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (85, 95, 105), 1)
    cv2.putText(
        canvas,
        title,
        (x + 10, y + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 235, 240),
        1,
        cv2.LINE_AA,
    )

    button_width = 24
    button_height = 22
    button_gap = 4
    close_rect = (x + width - button_width - 6, y + 4, button_width, button_height)
    fullscreen_rect = (
        close_rect[0] - button_width - button_gap,
        y + 4,
        button_width,
        button_height,
    )
    minimize_rect = (
        fullscreen_rect[0] - button_width - button_gap,
        y + 4,
        button_width,
        button_height,
    )
    _draw_button(canvas, minimize_rect, "_")
    _draw_button(canvas, fullscreen_rect, "[]")
    _draw_button(canvas, close_rect, "X", danger=True)

    if not minimized and height > title_height + 2:
        content_x = x + 1
        content_y = y + title_height + 1
        content_width = max(1, width - 2)
        content_height = max(1, height - title_height - 2)
        image = _resize_to_fit(_ensure_rgb(image_rgb), content_width, content_height)
        image_height, image_width = image.shape[:2]
        paste_x = content_x + (content_width - image_width) // 2
        paste_y = content_y + (content_height - image_height) // 2
        canvas[paste_y:paste_y + image_height, paste_x:paste_x + image_width] = image

    return {
        "minimize": minimize_rect,
        "fullscreen": fullscreen_rect,
        "close": close_rect,
    }


def show_preview(captured, window_name: str, max_width: int, max_height: int) -> int:
    state = _preview_state(window_name)
    if window_name in _PREVIEW_WINDOWS and _window_closed(window_name):
        return ord("q")
    if state["closed"]:
        return ord("q")

    left_width = max(1, max_width // 2)
    pane_height = max(1, max_height // 2)
    right_width = max(1, max_width - left_width)
    pointcloud_image = create_pointcloud_visualization(
        captured.depth_image,
        captured.color_image,
        captured.depth_scale,
        captured.streams,
        output_size=_pointcloud_preview_size(captured, right_width, max_height),
        view_state=state["pointcloud_view"],
    )

    images = {
        "rgb": captured.color_image,
        "depth": captured.depth_visualization_image,
        "pointcloud": pointcloud_image,
    }
    titles = {
        "rgb": "RGB",
        "depth": "Depth",
        "pointcloud": "Point Cloud",
    }

    canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    fullscreen_panel = state["fullscreen"]
    if fullscreen_panel:
        panel_rects = {fullscreen_panel: (0, 0, max_width, max_height)}
    else:
        panel_rects = {
            "rgb": (0, 0, left_width, pane_height),
            "depth": (0, pane_height, left_width, max_height - pane_height),
            "pointcloud": (left_width, 0, max_width - left_width, max_height),
        }

    state["panel_rects"] = panel_rects
    state["buttons"] = {}
    for panel_name, rect in panel_rects.items():
        state["buttons"][panel_name] = _draw_panel(
            canvas,
            rect,
            titles[panel_name],
            images[panel_name],
            panel_name in state["minimized"],
        )

    if window_name not in _PREVIEW_WINDOWS:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, max_width, max_height)
        cv2.setMouseCallback(window_name, _preview_mouse_callback, state)
        _PREVIEW_WINDOWS.add(window_name)
    cv2.imshow(window_name, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1) & 0xFF
    if state["closed"] or _window_closed(window_name):
        return ord("q")
    return key


def close_preview_windows() -> None:
    cv2.destroyAllWindows()
    _PREVIEW_WINDOWS.clear()
    _PREVIEW_STATES.clear()


def create_aligned_pointcloud_visualization(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    depth_scale: float,
    streams: dict,
    output_size: tuple[int, int] = (960, 540),
    target_points: int = 60000,
) -> np.ndarray:
    height, width = depth_image.shape[:2]
    output_width, output_height = output_size
    pointcloud = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    stride = max(1, round((height * width / target_points) ** 0.5))
    sampled_depth = depth_image[::stride, ::stride]
    valid_mask = sampled_depth > 0
    if not np.any(valid_mask):
        return pointcloud

    v_grid, u_grid = np.indices(sampled_depth.shape)
    u = (u_grid[valid_mask] * stride).astype(np.float32)
    v = (v_grid[valid_mask] * stride).astype(np.float32)
    z = sampled_depth[valid_mask].astype(np.float32) * float(depth_scale)

    px = np.round(u * (output_width - 1) / max(1, width - 1)).astype(np.int32)
    py = np.round(v * (output_height - 1) / max(1, height - 1)).astype(np.int32)
    inside = (px >= 0) & (px < output_width) & (py >= 0) & (py < output_height)
    if not np.any(inside):
        return pointcloud

    color_height, color_width = color_image.shape[:2]
    color_u = np.clip(
        np.round(u[inside] * (color_width - 1) / max(1, width - 1)).astype(np.int32),
        0,
        color_width - 1,
    )
    color_v = np.clip(
        np.round(v[inside] * (color_height - 1) / max(1, height - 1)).astype(np.int32),
        0,
        color_height - 1,
    )
    colors = color_image[color_v, color_u, :3]

    draw_order = np.argsort(z[inside])[::-1]
    pointcloud[py[inside][draw_order], px[inside][draw_order]] = colors[draw_order]
    return cv2.dilate(pointcloud, np.ones((2, 2), dtype=np.uint8), iterations=1)


def create_pointcloud_visualization(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    depth_scale: float,
    streams: dict,
    output_size: tuple[int, int] = (960, 540),
    target_points: int = 60000,
    view_state: PointCloudViewState | None = None,
) -> np.ndarray:
    if view_state is None:
        view_state = PointCloudViewState()

    height, width = depth_image.shape[:2]
    output_width, output_height = output_size
    pointcloud = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    intrinsics = _depth_intrinsics(streams, (height, width))
    _draw_grid(pointcloud, view_state)
    _draw_frustum(pointcloud, intrinsics, view_state)
    _draw_axes(
        pointcloud,
        _view_points(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), view_state)[0],
        view_state.rotation,
        size=0.1,
        thickness=1,
    )

    stride = max(1, round((height * width / target_points) ** 0.5))
    sampled_depth = depth_image[::stride, ::stride]
    valid_mask = sampled_depth > 0
    if not np.any(valid_mask):
        return pointcloud

    v_grid, u_grid = np.indices(sampled_depth.shape)
    u = (u_grid[valid_mask] * stride).astype(np.float32)
    v = (v_grid[valid_mask] * stride).astype(np.float32)
    z = sampled_depth[valid_mask].astype(np.float32) * float(depth_scale)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    x = (u - ppx) / fx * z
    y = (v - ppy) / fy * z
    vertices = np.column_stack((x, y, z)).astype(np.float32)
    pivot = view_state.pivot
    view_vertices = np.dot(vertices - pivot, view_state.rotation) + pivot - view_state.translation

    z_view = view_vertices[:, 2]
    valid_z = z_view > 0.03
    if not np.any(valid_z):
        return pointcloud

    view_vertices = view_vertices[valid_z]
    u = u[valid_z]
    v = v[valid_z]
    z_view = z_view[valid_z]

    view_aspect = output_height / output_width
    projected = view_vertices[:, :2] / z_view[:, np.newaxis]
    px = (projected[:, 0] * output_width * view_aspect + output_width / 2.0).astype(np.int32)
    py = (projected[:, 1] * output_height + output_height / 2.0).astype(np.int32)
    inside = (px >= 0) & (px < output_width) & (py >= 0) & (py < output_height)
    if not np.any(inside):
        return pointcloud

    color_height, color_width = color_image.shape[:2]
    color_u = np.clip(
        np.round(u[inside] * (color_width - 1) / max(1, width - 1)).astype(np.int32),
        0,
        color_width - 1,
    )
    color_v = np.clip(
        np.round(v[inside] * (color_height - 1) / max(1, height - 1)).astype(np.int32),
        0,
        color_height - 1,
    )
    colors = color_image[color_v, color_u, :3]

    point_layer = np.zeros_like(pointcloud)
    draw_order = np.argsort(z_view[inside])[::-1]
    point_layer[py[inside][draw_order], px[inside][draw_order]] = colors[draw_order]
    point_layer = cv2.dilate(point_layer, np.ones((2, 2), dtype=np.uint8), iterations=1)
    point_mask = point_layer > 0
    pointcloud[point_mask] = point_layer[point_mask]
    if any(view_state.mouse_buttons.values()):
        _draw_axes(
            pointcloud,
            _view_points(view_state.pivot.reshape(1, 3), view_state)[0],
            view_state.rotation,
            thickness=4,
        )
    return pointcloud
