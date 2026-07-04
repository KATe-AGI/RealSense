# pyrealsense2 接口说明文档

本文用于帮助你从“相机开发者”的角度理解 `pyrealsense2` 能提供哪些接口、哪些参数可以配置、哪些能力适合放在当前 RealSense 项目中。

本文基于当前机器安装的 `pyrealsense2` 和已连接的 Intel RealSense D435IF 整理。当前项目边界是相机处理：设备连接、参数配置、帧采集、depth->color 对齐、相机元数据、采集数据保存、索引管理。YOLO、检测、分割、训练集转换等视觉算法逻辑应放在独立视觉算法项目中。

## 1. 典型调用流程

大多数 RealSense Python 程序都围绕 `pipeline` 和 `config` 展开：

```python
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)
try:
    frames = pipeline.wait_for_frames()
    aligned = rs.align(rs.stream.color).process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
finally:
    pipeline.stop()
```

核心对象可以先这样理解：

```text
rs.context   SDK 上下文，用于枚举设备
rs.device    物理设备或回放设备
rs.sensor    具体传感器模块，用于读取 stream profile 和设置 option
rs.pipeline  高层采集管线，最常用
rs.config    流配置，例如分辨率、格式、FPS、录制/回放
rs.frame     图像帧/深度帧/运动帧的基础对象
rs.align     对齐工具，例如 depth -> color
```

## 2. 设备发现与设备信息

如果要在启动采集前检查相机，可以使用 `rs.context()`。

常用接口：

```text
context.query_devices()
context.query_all_sensors()
context.set_devices_changed_callback(...)
device.query_sensors()
device.get_info(...)
device.supports(...)
device.hardware_reset()
```

常用 `rs.camera_info` 字段：

```text
name
serial_number
firmware_version
recommended_firmware_version
product_id
product_line
usb_type_descriptor
physical_port
advanced_mode
camera_locked
connection_type
imu_type
```

示例：

```python
ctx = rs.context()
for dev in ctx.query_devices():
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(name, serial)
```

## 3. Stream 与 Format

当前 SDK 暴露的主要 stream 类型包括：

```text
color        彩色图
depth        深度图
infrared     红外图
accel        加速度计
gyro         陀螺仪
motion       运动流
pose         位姿流，部分设备支持
fisheye      鱼眼流，部分设备支持
confidence   置信度流，部分设备支持
gpio         GPIO 原始流，部分设备支持
```

常见 format：

```text
rgb8            RGB 彩色图
bgr8            BGR 彩色图，适合 OpenCV
rgba8           RGBA 彩色图
bgra8           BGRA 彩色图
yuyv            压缩/打包彩色格式
z16             原始深度，uint16
y8              8-bit 红外
y16             16-bit 红外
motion_xyz32f   IMU 三轴浮点数据
xyz32f          3D 点格式
```

当前项目默认使用：

```text
Color: rs.stream.color, 1920x1080, rs.format.rgb8, 30 FPS
Depth: rs.stream.depth, 1280x720, rs.format.z16, 30 FPS
Align: rs.align(rs.stream.color)
```

深度图 `d2rgb.npy` 默认保存 raw aligned depth 的 `uint16` 矩阵。换算到米：

```python
depth_meters = depth_raw * depth_scale
```

其中 `depth_scale` 来自：

```python
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
```

## 4. D435IF 可选分辨率、格式和 FPS

下面是当前连接的 Intel RealSense D435IF 通过 `sensor.get_stream_profiles()` 枚举到的主要可选项。不同型号、固件、USB 模式下可能不同，实际开发时应以本机枚举结果为准。

### 4.1 Depth 可选项

Depth stream 使用 `rs.stream.depth`，当前主要格式为 `rs.format.z16`。

| 分辨率 | 格式 | FPS |
| --- | --- | --- |
| 1280x720 | z16 | 6, 15, 30 |
| 848x480 | z16 | 6, 15, 30, 60, 90 |
| 848x100 | z16 | 100, 300 |
| 640x480 | z16 | 6, 15, 30, 60, 90 |
| 640x360 | z16 | 6, 15, 30, 60, 90 |
| 480x270 | z16 | 6, 15, 30, 60, 90 |
| 424x240 | z16 | 6, 15, 30, 60, 90 |
| 256x144 | z16 | 90, 300 |

当前项目默认：

```text
Depth: 1280x720, z16, 30 FPS
```

### 4.2 RGB Color 可选项

Color stream 使用 `rs.stream.color`。常用格式包括 `rgb8`、`bgr8`、`rgba8`、`bgra8`、`yuyv`。其中 `rgb8` 适合直接保存为标准 RGB 图片，`bgr8` 更适合直接给 OpenCV 使用。

| 分辨率 | 常用格式 | FPS |
| --- | --- | --- |
| 1920x1080 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30 |
| 1920x1080 | raw16 | 30 |
| 1280x720 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30 |
| 960x540 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30, 60 |
| 848x480 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30, 60 |
| 640x480 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30, 60 |
| 640x360 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30, 60 |
| 424x240 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 15, 30, 60 |
| 320x240 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 30, 60 |
| 320x180 | rgb8, bgr8, rgba8, bgra8, yuyv | 6, 30, 60 |

当前项目默认：

```text
Color: 1920x1080, rgb8, 30 FPS
```

### 4.3 Infrared 可选项

Infrared stream 使用 `rs.stream.infrared`，D435IF 有左右两个红外流，stream index 通常为 `1` 和 `2`。

常用 `y8` 红外：

| 分辨率 | 格式 | FPS |
| --- | --- | --- |
| 1280x720 | y8 | 6, 15, 30 |
| 1280x800 | y8 | 15, 30 |
| 848x480 | y8 | 6, 15, 30, 60, 90 |
| 848x100 | y8 | 100, 300 |
| 640x480 | y8 | 6, 15, 30, 60, 90 |
| 640x360 | y8 | 6, 15, 30, 60, 90 |
| 480x270 | y8 | 6, 15, 30, 60, 90 |
| 424x240 | y8 | 6, 15, 30, 60, 90 |

部分 `y16` 红外：

| 分辨率 | 格式 | FPS |
| --- | --- | --- |
| 1280x800 | y16 | 15, 25 |
| 640x400 | y16 | 15, 25 |

### 4.4 IMU 可选项

Motion Module 暴露加速度计和陀螺仪：

| Stream | 格式 | FPS |
| --- | --- | --- |
| accel | motion_xyz32f | 100, 200, 400 |
| gyro | motion_xyz32f | 200, 400 |

### 4.5 当前代码的参数限制

当前 `capture_current_frame.py` 已经把 color 和 depth 的分辨率/FPS 分开配置：

```text
--color-width
--color-height
--color-fps
--depth-width
--depth-height
--depth-fps
```

当前默认配置：

```text
Color: 1920x1080 @ 30
Depth: 1280x720 @ 30
Post-Processing: 默认关闭，可通过 --enable-post-processing 开启
```

也可以通过参数切换到其他组合，例如：

```text
--color-width 1280 --color-height 720 --color-fps 30 --depth-width 1280 --depth-height 720 --depth-fps 30
--color-width 640 --color-height 480 --color-fps 30 --depth-width 640 --depth-height 480 --depth-fps 30
```

`depth -> color` 对齐后的 raw depth 分辨率会跟 color 图一致。因此如果 color 是 `1920x1080`，默认保存的 `d2rgb.npy` 也是 `1920x1080`。如果运行时添加 `--enable-post-processing`，Viewer 风格后处理会另存为 `d2rgb_filtered.npy`；其中 Decimation Filter 会下采样，因此 filtered 结果通常会变成 `960x540`。

## 5. Pipeline 与 Config

`rs.pipeline` 是最常用的高层采集接口。

常用方法：

```text
pipeline.start()
pipeline.start(config)
pipeline.wait_for_frames(timeout_ms=5000)
pipeline.poll_for_frames()
pipeline.try_wait_for_frames()
pipeline.get_active_profile()
pipeline.stop()
```

`rs.config` 用于选择设备和数据流。

常用方法：

```text
config.enable_stream(...)
config.disable_stream(...)
config.disable_all_streams()
config.enable_device(serial)
config.enable_device_from_file(path)
config.enable_record_to_file(path)
config.can_resolve(pipeline_wrapper)
config.resolve(pipeline_wrapper)
```

工程建议：

```text
需要稳定输出尺寸时，显式调用 enable_stream。
需要指定某台相机时，使用 enable_device(serial)。
需要保存 SDK 原始回放流时，使用 enable_record_to_file。
当前项目主要保存 PNG/NPY/JPG/JSON，不以 SDK 录制文件作为主输出。
```

## 6. Sensor Option 参数控制

相机参数主要通过 `sensor.get_option(...)` 和 `sensor.set_option(...)` 控制。

基本模式：

```python
for sensor in device.query_sensors():
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1.0)
```

常用检查接口：

```text
sensor.get_supported_options()
sensor.supports(option)
sensor.get_option(option)
sensor.set_option(option, value)
sensor.get_option_range(option)
sensor.is_option_read_only(option)
sensor.get_option_description(option)
```

相机开发中常见 option：

```text
enable_auto_exposure       是否启用自动曝光
exposure                   手动曝光值
gain                       增益
enable_auto_white_balance  自动白平衡
white_balance              手动白平衡
brightness                 亮度
contrast                   对比度
saturation                 饱和度
sharpness                  锐度
gamma                      Gamma
backlight_compensation     逆光补偿
power_line_frequency       电源频率
visual_preset              深度预设
laser_power                激光功率
emitter_enabled            是否启用红外发射器
emitter_on_off             发射器开关模式
emitter_always_on          发射器常开
depth_units                深度单位
inter_cam_sync_mode        多相机硬件同步模式
global_time_enabled        全局时间
frames_queue_size          帧队列大小
auto_exposure_priority     自动曝光优先级
auto_exposure_limit        自动曝光上限
auto_gain_limit            自动增益上限
```

使用规则：

```text
自动曝光：enable_auto_exposure = 1.0
手动曝光：先 enable_auto_exposure = 0.0，再设置 exposure/gain
深度激光：使用 laser_power 和 emitter_enabled
多相机同步：使用 inter_cam_sync_mode，并配合硬件同步线
时间一致性：需要和外部系统对时间时，关注 global_time_enabled
```

## 7. 当前 D435IF 支持的关键参数

当前连接设备：

```text
Device: Intel RealSense D435IF
Product line: D400
USB: 3.2
Depth scale: 约 0.001
```

Stereo Module 常用参数：

```text
Exposure: 默认 8500，范围 1..165000
Gain: 默认 16，范围 16..248
Enable Auto Exposure: 默认 1
Visual Preset: 默认 0，范围 0..5
Laser Power: 默认 150，范围 0..360，步长 30
Emitter Enabled: 默认 1，范围 0..2
Depth Units: 默认约 0.001
Inter Cam Sync Mode: 默认 0
Global Time Enabled: 默认 1
Auto Exposure Limit: 范围 1..165000
Auto Gain Limit: 范围 16..248
```

RGB Camera 常用参数：

```text
Backlight Compensation: 默认 0
Brightness: 默认 0，范围 -64..64
Contrast: 默认 50，范围 0..100
Exposure: 默认 156，范围 1..10000
Gain: 默认 64，范围 0..128
Gamma: 默认 300，范围 100..500
Hue: 默认 0，范围 -180..180
Saturation: 默认 64，范围 0..100
Sharpness: 默认 50，范围 0..100
White Balance: 默认 4600，范围 2800..6500
Enable Auto Exposure: 默认 1
Enable Auto White Balance: 默认 1
Power Line Frequency: 默认 3
Auto Exposure Priority: 默认 0
Global Time Enabled: 默认 1
```

Motion Module 常用参数：

```text
Frames Queue Size
Enable Motion Correction
Global Time Enabled
Gyro Sensitivity
```

## 8. Frame、Profile、内参和元数据

拿到 frames 后常用：

```text
frames.get_color_frame()
frames.get_depth_frame()
frame.get_data()
frame.get_frame_number()
frame.get_timestamp()
frame.get_frame_timestamp_domain()
```

转成 numpy：

```python
color = np.asanyarray(color_frame.get_data())
depth = np.asanyarray(depth_frame.get_data())
```

读取 stream profile 和内参：

```python
profile = color_frame.get_profile().as_video_stream_profile()
intrinsics = profile.get_intrinsics()
```

内参字段：

```text
width
height
fx
fy
ppx
ppy
model
coeffs
```

读取 frame metadata 时，必须先判断是否支持：

```python
field = rs.frame_metadata_value.actual_exposure
if frame.supports_frame_metadata(field):
    value = frame.get_frame_metadata(field)
```

常用 metadata：

```text
frame_counter
frame_timestamp
sensor_timestamp
backend_timestamp
actual_exposure
gain_level
actual_fps
auto_exposure
time_of_arrival
temperature
frame_laser_power
input_width
input_height
```

## 9. Depth -> Color 对齐

对齐的作用是把 depth 映射到另一个 stream 的坐标系。

当前项目使用：

```python
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
aligned_depth = aligned_frames.get_depth_frame()
```

对齐后的 raw depth 图像具有 color 的分辨率和内参。当前项目默认把 raw aligned depth 保存为 `d2rgb.npy`，所以默认可以和 `color.png` 做像素级对应。如果添加 `--enable-post-processing`，后处理结果会另存为 `d2rgb_filtered.npy`，尺寸可能小于 `color.png`。

## 10. 点云接口

SDK 提供点云生成接口：

```python
pc = rs.pointcloud()
pc.map_to(color_frame)
points = pc.calculate(depth_frame)
points.export_to_ply("sample_pointcloud.ply", color_frame)
```

相关接口：

```text
rs.pointcloud.calculate(depth_frame)
rs.pointcloud.map_to(color_frame)
rs.points.get_vertices()
rs.points.get_texture_coordinates()
rs.points.export_to_ply(path, color_frame)
rs.save_to_ply
```

当前项目建议：

```text
默认不保存 PLY。
默认以 color.png + raw aligned d2rgb.npy 作为当前采集样本；添加 `--enable-post-processing` 后，会额外保存 `d2rgb_filtered.npy` 作为后处理深度。
需要三维检查、标定或空间验证时，再打开点云导出。
```

## 11. 后处理滤波接口

SDK 提供多个 depth 后处理模块。注意：滤波会改变深度数据。当前项目默认不启用后处理；只有添加 `--enable-post-processing` 时才按 Viewer 默认风格另存 `d2rgb_filtered.npy`。

常用滤波器：

```text
rs.decimation_filter       深度下采样
rs.spatial_filter          空间平滑
rs.temporal_filter         时间平滑
rs.hole_filling_filter     空洞填补
rs.threshold_filter        深度阈值过滤
rs.disparity_transform     depth/disparity 转换
rs.colorizer               深度可视化
```

示例：

```python
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
filtered = spatial.process(depth_frame)
filtered = temporal.process(filtered)
```

项目建议：

```text
默认保存 raw aligned depth。
启用后处理时，raw aligned depth 的 stream 信息记录在 meta.json 中。
启用后处理时，保存的 `d2rgb.npy` 仍是 raw aligned depth，另存的 `d2rgb_filtered.npy` 是后处理结果。
meta.json 中记录每个后处理模块是否启用、是否实际应用、以及 block option。
如果后续需要严格像素级测量，可以增加 raw depth 额外输出。
```

## 12. Record 与 Playback

SDK 支持录制和回放：

```text
config.enable_record_to_file(path)
config.enable_device_from_file(path)
rs.recorder
rs.playback
context.load_device(path)
```

这些接口适合 SDK 调试、问题复现和回放测试。但当前项目需要的是明确的 PNG/NPY/JPG/JSON 数据集，因此不建议把 SDK 录制文件作为主输出。

## 13. IMU 与 Motion

D435IF 暴露 Motion Module。相关 stream/format：

```text
rs.stream.accel
rs.stream.gyro
rs.format.motion_xyz32f
```

示例配置：

```python
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
```

IMU 可用于记录相机运动状态，但当前静态单帧采集暂时不需要。

## 14. 校准与高级接口

部分设备支持高级接口：

```text
device.as_updatable()
device.as_auto_calibrated_device()
device.as_calibration_change_device()
device.as_debug_protocol()
device.as_recorder()
device.as_playback()
rs.rs400_advanced_mode
```

这些接口可能改变设备状态，比如固件、校准、advanced mode。当前项目正常采集时建议只读取校准相关信息，不在普通采集流程里修改校准结果。

## 15. 当前项目适合暴露的参数

已经暴露在 `capture_current_frame.py` 中：

```text
--mode {single,continuous}
--duration-s
--interval-s
--output-dir
--color-width
--color-height
--color-fps
--depth-width
--depth-height
--depth-fps
--warmup-frames
--auto-exposure {on,off,default}
--enable-post-processing
```

连续采集示例：

```powershell
D:\miniconda3\envs\realsense\python.exe capture_current_frame.py --mode continuous --duration-s 10
```

连续采集会打开可视化窗口，左侧为 RGB，右侧为 depth visualization。默认启动后立即保存第一帧，然后每 `1.0s` 保存一帧；可以通过 `--interval-s` 修改。按 `q` 或 `Esc` 可提前退出。

后续适合逐步增加：

```text
--manual-exposure
--gain
--laser-power
--emitter-enabled
--visual-preset
--save-pointcloud
--enable-filter spatial|temporal|hole-filling
--min-depth
--max-depth
--serial-number
```

推荐顺序：

```text
1. 先保持 capture pipeline 稳定。
2. 需要三维检查时增加点云。
3. 场景固定后再增加 manual exposure/gain。
4. 如果任务需要严格原始深度，再增加 raw depth 额外输出。
5. 多相机之前先增加 serial-number 选择。
```

## 16. 当前仓库的职责边界

适合放在本仓库：

```text
RealSense 连接检查
相机 stream 配置
相机 option 配置
单帧/批量采集
depth->color 对齐
depth 保存和后处理记录
相机 metadata 保存
可选点云导出
相机侧校准记录
```

不建议放在本仓库：

```text
YOLO 模型训练
目标检测
图像分割
标注格式转换
数据增强
视觉算法 benchmark
```
