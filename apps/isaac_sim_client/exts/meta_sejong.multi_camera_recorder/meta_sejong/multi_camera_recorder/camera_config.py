"""Hard-coded camera calibration used by the recorder.

Cameras 2/4/6/8 are copied from edge_client/config/cameras.local.yaml.
Cameras 1/3/5/7 are the matching cameras from the same deployment calibration
result (calibration_result_20260814-123941.json), completing the requested
eight calibrated views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Matrix3 = Tuple[Tuple[float, float, float], ...]
Matrix4 = Tuple[Tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    image_size: tuple[int, int]
    camera_matrix: Matrix3
    camera_to_world: Matrix4


CAMERAS = (
    CameraCalibration(
        name="camera_1",
        image_size=(1920, 1080),
        camera_matrix=(
            (1084.319215, 0.0, 1004.401968),
            (0.0, 1084.886231, 562.166553),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (
                -0.371437050010139,
                0.5078045464022022,
                -0.7772831205119703,
                95.1463620791792,
            ),
            (
                0.9278980315954676,
                0.1739533016261343,
                -0.32976583587232383,
                11.902003414315796,
            ),
            (
                -0.03224562654550238,
                -0.8437267142470082,
                -0.535803537688168,
                17.93938459641992,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_2",
        image_size=(1920, 1080),
        camera_matrix=(
            (1062.382134, 0.0, 985.023806),
            (0.0, 1062.649562, 545.594731),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (
                0.09344430594339304,
                0.5835270836663197,
                -0.806699650825367,
                87.96380999510313,
            ),
            (
                0.994734321574476,
                -0.020462332066286712,
                0.10042390510876469,
                4.572397051814354,
            ),
            (
                0.04209311002727585,
                -0.8118358287826712,
                -0.5823665137966749,
                18.331585308263065,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_3",
        image_size=(1920, 1080),
        camera_matrix=(
            (1056.852075, 0.0, 950.555565),
            (0.0, 1055.659453, 561.079672),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (
                0.9942237131563773,
                0.09412606702649745,
                0.05157159467445976,
                85.04657439657854,
            ),
            (
                0.034357754992141984,
                -0.7343380689428767,
                0.6779139696582899,
                0.8953949518379465,
            ),
            (
                0.10168035069482842,
                -0.672226076662244,
                -0.7333302029787597,
                18.654805588769435,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_4",
        image_size=(1920, 1080),
        camera_matrix=(
            (1085.155271, 0.0, 962.543523),
            (0.0, 1085.272475, 510.818334),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (
                0.9578086453127708,
                -0.11757429784049936,
                0.2622568922221206,
                76.69943890692215,
            ),
            (
                -0.28090951599194813,
                -0.5758904740308722,
                0.7677497862100623,
                -7.423195771438736,
            ),
            (
                0.06076350768351047,
                -0.8090280538685997,
                -0.5846207185678977,
                18.72447133657255,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_5",
        image_size=(1920, 1080),
        camera_matrix=(
            (1095.257192, 0.0, 981.644202),
            (0.0, 1096.670312, 512.155081),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (-0.9879207018121395, 0.07184409370003864, -0.1372997264546309, 90.32172238242572),
            (0.1534882156911524, 0.5755276256844142, -0.8032492547543194, 15.938798427371779),
            (0.021311096107462665, -0.8146204295357599, -0.5796028100370759, 17.80443462412862),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_6",
        image_size=(1920, 1080),
        camera_matrix=(
            (949.1181864083335, 0.0, 852.3669302041667),
            (0.0, 949.4937753101852, 465.4189633305556),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (-0.9662214343759505, 0.10190854082924393, -0.2367083582649197, 82.65864132105217),
            (0.25080785782284565, 0.5830376350474685, -0.772762938706836, 7.999023415358104),
            (0.0592587171786456, -0.8060284300550958, -0.5889029165225272, 18.619125319485114),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_7",
        image_size=(1920, 1080),
        camera_matrix=(
            (1075.983623, 0.0, 989.12017),
            (0.0, 1076.859402, 569.653343),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (0.1850763404010905, -0.6373782980355528, 0.7479944625795611, 80.15553943366142),
            (-0.9818973535051698, -0.0887183201190405, 0.1673525306556229, 5.214113088055539),
            (-0.040306032686707675, -0.7654266134320858, -0.6422596667426442, 18.294534321259345),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
    CameraCalibration(
        name="camera_8",
        image_size=(1920, 1080),
        camera_matrix=(
            (1070.814071, 0.0, 954.286513),
            (0.0, 1071.563311, 542.369649),
            (0.0, 0.0, 1.0),
        ),
        camera_to_world=(
            (-0.03169815414408619, -0.6448242050689098, 0.7636732892470067, 72.0069480262324),
            (
                -0.9978862895027608,
                -0.022948732331193673,
                -0.060797012249149206,
                -2.781099359561564,
            ),
            (0.05672870886685138, -0.7639863421756397, -0.6427338547194601, 18.414605928306088),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ),
)


def opencv_camera_to_usd_matrix_rows(
    camera: CameraCalibration,
    meters_per_unit: float,
) -> Matrix4:
    """Return a row-vector USD xform matrix for an OpenCV camera pose.

    OpenCV uses +X right, +Y down, +Z forward. A USD camera uses +X right,
    +Y up, -Z forward. The returned matrix also converts metre translations
    into the current Stage's authored distance unit.
    """
    if meters_per_unit <= 0:
        raise ValueError("meters_per_unit must be positive")

    source = camera.camera_to_world
    # Columns are the USD camera's local axes expressed in world coordinates.
    usd_x = (source[0][0], source[1][0], source[2][0])
    usd_y = (-source[0][1], -source[1][1], -source[2][1])
    usd_z = (-source[0][2], -source[1][2], -source[2][2])
    translation = (
        source[0][3] / meters_per_unit,
        source[1][3] / meters_per_unit,
        source[2][3] / meters_per_unit,
    )
    return (
        (*usd_x, 0.0),
        (*usd_y, 0.0),
        (*usd_z, 0.0),
        (*translation, 1.0),
    )


def usd_camera_intrinsics(
    camera: CameraCalibration,
    focal_length: float = 50.0,
) -> dict[str, float]:
    """Convert OpenCV intrinsics to the equivalent USD camera aperture values."""
    if focal_length <= 0:
        raise ValueError("focal_length must be positive")
    width, height = camera.image_size
    fx = camera.camera_matrix[0][0]
    fy = camera.camera_matrix[1][1]
    cx = camera.camera_matrix[0][2]
    cy = camera.camera_matrix[1][2]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{camera.name} has invalid focal lengths")

    horizontal_aperture = focal_length * width / fx
    vertical_aperture = focal_length * height / fy
    return {
        "focal_length": focal_length,
        "horizontal_aperture": horizontal_aperture,
        "vertical_aperture": vertical_aperture,
        "horizontal_aperture_offset": (width / 2.0 - cx) * focal_length / fx,
        "vertical_aperture_offset": (cy - height / 2.0) * focal_length / fy,
    }
