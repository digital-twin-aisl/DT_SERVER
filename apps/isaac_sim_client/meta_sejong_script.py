"""Receive live people scenes and visualize roots and poses in Isaac Sim.

The default transport is Zenoh. WebSocket support remains available for the
legacy sim_backend path by setting ISAAC_SCENE_TRANSPORT=websocket.
"""

import asyncio
import atexit
import json
import math
import os
import queue
import threading
import traceback

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade


DEFAULT_USD_PATH = "/home/dojan/All/2025_SejongUniv_All.usd"
DEFAULT_ZENOH_ENDPOINT = "127.0.0.1:7447"
DEFAULT_ZENOH_TOPIC = "meta-sejong/scene/v1"

USD_PATH = os.environ.get("ISAAC_USD_PATH", DEFAULT_USD_PATH)
SCENE_TRANSPORT = os.environ.get("ISAAC_SCENE_TRANSPORT", "zenoh").lower()
ZENOH_ENDPOINT = os.environ.get("ISAAC_ZENOH_ENDPOINT", DEFAULT_ZENOH_ENDPOINT)
ZENOH_TOPIC = os.environ.get("ISAAC_ZENOH_TOPIC", DEFAULT_ZENOH_TOPIC)
SIM_BACKEND_WS_URL = os.environ.get(
    "SIM_BACKEND_WS_URL",
    "ws://127.0.0.1:8004/ws/sim",
)
USD_PEOPLE_ROOT = "/World/MetaSejong_People"
USD_LOOKS_ROOT = "/World/MetaSejong_Looks"
LIMBS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 4),
    (4, 5),
    (0, 9),
    (9, 10),
    (10, 11),
    (2, 6),
    (2, 12),
    (6, 7),
    (7, 8),
    (12, 13),
    (13, 14),
)
SUPPORTED_JOINT_FORMAT = "voxelpose_15j_xyz"
FALLBACK_HEIGHT_METERS = 1.7
FALLBACK_RADIUS_METERS = 0.3
JOINT_RADIUS_METERS = 0.04
BONE_RADIUS_METERS = 0.018

if SCENE_TRANSPORT not in {"zenoh", "websocket", "both"}:
    raise ValueError(
        "ISAAC_SCENE_TRANSPORT must be one of: zenoh, websocket, both"
    )


omni.usd.get_context().open_stage(USD_PATH)
print(f"[Meta Sejong] Loading stage from {USD_PATH} ...")


def _id_color(global_id: int) -> Gf.Vec3f:
    palette = (
        (0.90, 0.10, 0.29),
        (0.24, 0.71, 0.29),
        (1.00, 0.88, 0.10),
        (0.00, 0.51, 0.78),
        (0.96, 0.51, 0.19),
        (0.57, 0.12, 0.71),
    )
    return Gf.Vec3f(*palette[global_id % len(palette)])


class MetaSejongDigitalTwin:
    """Update USD people prims from complete scene snapshots."""

    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("USD stage is not ready")
        self.units_per_meter = UsdGeom.GetStageMetersPerUnit(self.stage)
        self._people = {}
        self._materials = {}
        self._render_modes = {}
        self._ensure_root_group()

    def _ensure_root_group(self):
        # Define unconditionally so an existing Scope/typeless prim is also
        # promoted to an editable Xform in the current stage.
        root = UsdGeom.Xform.Define(self.stage, USD_PEOPLE_ROOT)
        root_xform = UsdGeom.Xformable(root)
        op_types = {
            op.GetOpType() for op in root_xform.GetOrderedXformOps()
        }
        if UsdGeom.XformOp.TypeTranslate not in op_types:
            root_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        if UsdGeom.XformOp.TypeOrient not in op_types:
            root_xform.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Quatd(1.0))
        if UsdGeom.XformOp.TypeScale not in op_types:
            root_xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))

        looks_prim = self.stage.GetPrimAtPath(USD_LOOKS_ROOT)
        if not looks_prim.IsValid():
            UsdGeom.Scope.Define(self.stage, USD_LOOKS_ROOT)

    def _position_meters_to_stage_units(self, position_meters) -> Gf.Vec3d:
        """Interpret an incoming position in MetaSejong_People space."""
        scale = 1.0 / self.units_per_meter
        return Gf.Vec3d(
            *(float(value) * scale for value in position_meters)
        )

    def _ensure_person_material(
        self,
        global_id: int,
        color: Gf.Vec3f,
    ) -> UsdShade.Material:
        """Create an RTX-compatible surface material for one person ID."""
        existing = self._materials.get(global_id)
        if existing is not None:
            return existing

        material_path = f"{USD_LOOKS_ROOT}/Person_{global_id}"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(
            self.stage,
            f"{material_path}/Shader",
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).Set(color)
        shader.CreateInput(
            "roughness",
            Sdf.ValueTypeNames.Float,
        ).Set(0.5)
        shader.CreateInput(
            "metallic",
            Sdf.ValueTypeNames.Float,
        ).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(),
            "surface",
        )
        self._materials[global_id] = material
        return material

    def _ensure_person(self, global_id: int):
        """Create one reusable fallback capsule and geometric skeleton."""
        existing = self._people.get(global_id)
        if existing is not None:
            return existing

        person_path = f"{USD_PEOPLE_ROOT}/Person_{global_id}"
        person = UsdGeom.Xform.Define(self.stage, person_path)
        person_xform = UsdGeom.Xformable(person)
        person_xform.ClearXformOpOrder()
        person_translate = person_xform.AddTranslateOp()

        color = _id_color(global_id)
        material = self._ensure_person_material(global_id, color)
        UsdShade.MaterialBindingAPI.Apply(person.GetPrim()).Bind(material)
        fallback = UsdGeom.Capsule.Define(
            self.stage,
            f"{person_path}/RootFallback",
        )
        fallback.GetHeightAttr().Set(
            FALLBACK_HEIGHT_METERS / self.units_per_meter
        )
        fallback.GetRadiusAttr().Set(
            FALLBACK_RADIUS_METERS / self.units_per_meter
        )
        fallback.GetAxisAttr().Set(UsdGeom.Tokens.z)
        fallback.GetDisplayColorAttr().Set([color])

        skeleton = UsdGeom.Xform.Define(self.stage, f"{person_path}/Skeleton")
        UsdGeom.Imageable(skeleton.GetPrim()).MakeInvisible()
        joint_translates = []
        for joint_index in range(15):
            joint = UsdGeom.Sphere.Define(
                self.stage,
                f"{person_path}/Skeleton/Joint_{joint_index:02d}",
            )
            joint.GetRadiusAttr().Set(
                JOINT_RADIUS_METERS / self.units_per_meter
            )
            joint.GetDisplayColorAttr().Set([color])
            joint_xform = UsdGeom.Xformable(joint)
            joint_xform.ClearXformOpOrder()
            joint_translates.append(joint_xform.AddTranslateOp())

        bones = []
        for bone_index in range(len(LIMBS)):
            bone = UsdGeom.Cylinder.Define(
                self.stage,
                f"{person_path}/Skeleton/Bone_{bone_index:02d}",
            )
            bone.GetAxisAttr().Set(UsdGeom.Tokens.z)
            bone.GetRadiusAttr().Set(
                BONE_RADIUS_METERS / self.units_per_meter
            )
            bone.GetDisplayColorAttr().Set([color])
            bone_xform = UsdGeom.Xformable(bone)
            bone_xform.ClearXformOpOrder()
            bones.append(
                {
                    "geometry": bone,
                    "translate": bone_xform.AddTranslateOp(),
                    "orient": bone_xform.AddOrientOp(
                        UsdGeom.XformOp.PrecisionDouble
                    ),
                }
            )

        state = {
            "person_prim": person.GetPrim(),
            "person_translate": person_translate,
            "fallback_prim": fallback.GetPrim(),
            "skeleton_prim": skeleton.GetPrim(),
            "joint_translates": joint_translates,
            "bones": bones,
        }
        for legacy_name in ("Root", "Joints", "Bones"):
            legacy_prim = self.stage.GetPrimAtPath(
                f"{person_path}/{legacy_name}"
            )
            if legacy_prim.IsValid():
                UsdGeom.Imageable(legacy_prim).MakeInvisible()
        self._people[global_id] = state
        return state

    def update_person_root(self, global_id: int, position_meters) -> None:
        """Place a person using a MetaSejong_People-relative transform."""
        state = self._ensure_person(global_id)

        UsdGeom.Imageable(state["person_prim"]).MakeVisible()
        local_position = self._position_meters_to_stage_units(position_meters)
        state["person_translate"].Set(local_position)

    def _show_fallback(self, global_id: int, reason: str) -> None:
        state = self._ensure_person(global_id)
        UsdGeom.Imageable(state["fallback_prim"]).MakeVisible()
        UsdGeom.Imageable(state["skeleton_prim"]).MakeInvisible()
        status = f"root:{reason}"
        if self._render_modes.get(global_id) != status:
            print(
                f"[Meta Sejong] global_id={global_id} "
                f"render=root reason={reason}"
            )
            self._render_modes[global_id] = status

    def update_person_pose(
        self,
        global_id: int,
        root_position_mm,
        pose: dict | None,
    ) -> None:
        """Update joint points and limb curves relative to the person's root."""
        if pose is None:
            self._show_fallback(global_id, "pose_missing")
            return
        if not isinstance(pose, dict):
            self._show_fallback(global_id, "pose_not_object")
            return
        if pose.get("joint_format") != SUPPORTED_JOINT_FORMAT:
            self._show_fallback(
                global_id,
                f"unsupported_joint_format:{pose.get('joint_format')}",
            )
            return
        if not isinstance(pose.get("joints"), list):
            self._show_fallback(global_id, "joints_not_list")
            return

        raw_joints = pose["joints"]
        if len(raw_joints) != 15 or any(
            not isinstance(joint, list)
            or len(joint) != 3
            or not all(math.isfinite(float(value)) for value in joint)
            for joint in raw_joints
        ):
            print(
                f"[Meta Sejong] Ignoring invalid pose for global_id={global_id}"
            )
            self._show_fallback(global_id, "invalid_pose")
            return

        # Incoming roots and joints are expressed in MetaSejong_People space.
        # Person_<id> carries the root translation, so its child geometry only
        # needs the joint-to-root offset in the same coordinate system.
        millimeters_per_stage_unit = 1000.0 * self.units_per_meter
        relative_joints = [
            Gf.Vec3f(
                *(
                    (float(joint[axis]) - float(root_position_mm[axis]))
                    / millimeters_per_stage_unit
                    for axis in range(3)
                )
            )
            for joint in raw_joints
        ]
        state = self._ensure_person(global_id)

        for translate_op, joint_position in zip(
            state["joint_translates"],
            relative_joints,
        ):
            translate_op.Set(Gf.Vec3d(joint_position))

        z_axis = Gf.Vec3d(0.0, 0.0, 1.0)
        for bone, (start_index, end_index) in zip(state["bones"], LIMBS):
            start = Gf.Vec3d(relative_joints[start_index])
            end = Gf.Vec3d(relative_joints[end_index])
            delta = end - start
            length = delta.GetLength()
            if length <= 1e-6:
                UsdGeom.Imageable(bone["geometry"].GetPrim()).MakeInvisible()
                continue

            UsdGeom.Imageable(bone["geometry"].GetPrim()).MakeVisible()
            bone["geometry"].GetHeightAttr().Set(length)
            bone["translate"].Set((start + end) * 0.5)
            bone["orient"].Set(Gf.Rotation(z_axis, delta).GetQuat())

        UsdGeom.Imageable(state["fallback_prim"]).MakeInvisible()
        UsdGeom.Imageable(state["skeleton_prim"]).MakeVisible()
        if self._render_modes.get(global_id) != "skeleton":
            print(
                f"[Meta Sejong] global_id={global_id} "
                f"render=skeleton joints={len(raw_joints)}"
            )
            self._render_modes[global_id] = "skeleton"

    def apply_scene(self, scene: dict) -> None:
        """Apply one schema_version=1 SceneOutput snapshot."""
        active_paths = set()
        for person in scene.get("people", []):
            global_id = int(person["global_id"])
            root_position_mm = person.get("root", {}).get("position")
            if not isinstance(root_position_mm, list) or len(root_position_mm) != 3:
                continue

            # VoxelPose SceneOutput coordinates are millimeters.
            root_position_m = [value / 1000.0 for value in root_position_mm]
            self.update_person_root(global_id, root_position_m)
            self.update_person_pose(
                global_id,
                root_position_mm,
                person.get("pose"),
            )
            active_paths.add(f"{USD_PEOPLE_ROOT}/Person_{global_id}")

        root_prim = self.stage.GetPrimAtPath(USD_PEOPLE_ROOT)
        for child in root_prim.GetChildren():
            if str(child.GetPath()) not in active_paths:
                UsdGeom.Imageable(child).MakeInvisible()

    def apply_legacy_object(self, track_id: int, position: dict) -> None:
        """Apply the legacy WebSocket position contract, whose unit is meters."""
        self.update_person_root(
            int(track_id),
            (
                position.get("x", 0.0),
                position.get("y", 0.0),
                position.get("z", 0.0),
            ),
        )


class LatestScene:
    """A single-slot handoff from a network thread to Isaac's update loop."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self.dropped = 0

    def put(self, scene: dict) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(scene)
        except queue.Full:
            self.dropped += 1

    def take(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class ZenohSceneReceiver:
    def __init__(self, endpoint: str, topic: str, scenes: LatestScene):
        self.endpoint = endpoint
        self.topic = topic
        self.scenes = scenes
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._worker = threading.Thread(
            target=self._run,
            name="isaac-zenoh-scene-subscriber",
            daemon=True,
        )
        self._worker.start()

    def _config(self):
        import zenoh

        endpoint = self.endpoint
        if "/" not in endpoint:
            endpoint = f"tcp/{endpoint}"
        return zenoh.Config.from_json5(
            json.dumps(
                {
                    "mode": "client",
                    "connect": {"endpoints": [endpoint]},
                }
            )
        )

    def _on_sample(self, sample) -> None:
        try:
            scene = json.loads(sample.payload.to_bytes().decode("utf-8"))
            if scene.get("schema_version") != 1:
                print(
                    "[Meta Sejong] Ignoring unsupported scene schema:",
                    scene.get("schema_version"),
                )
                return
            if not isinstance(scene.get("people"), list):
                raise ValueError("scene.people must be a list")
            self.scenes.put(scene)
        except Exception:
            print("[Meta Sejong] Invalid Zenoh scene payload")
            traceback.print_exc()

    def _run(self) -> None:
        try:
            import zenoh

            with zenoh.open(self._config()) as session:
                subscriber = session.declare_subscriber(
                    self.topic,
                    self._on_sample,
                )
                self._ready.set()
                print(
                    "[Meta Sejong] Zenoh subscriber is ready: "
                    f"endpoint={self.endpoint} topic={self.topic}"
                )
                self._stop.wait()
                _ = subscriber
        except Exception as exc:
            self._error = exc
            self._ready.set()
            print(f"[Meta Sejong] Zenoh subscriber failed: {exc}")
            traceback.print_exc()

    def close(self) -> None:
        self._stop.set()
        if self._worker.is_alive() and threading.current_thread() is not self._worker:
            self._worker.join(timeout=5)


async def zenoh_update_task(scenes: LatestScene):
    digital_twin = None
    last_timestamp = None
    while True:
        await omni.kit.app.get_app().next_update_async()
        scene = scenes.take()
        if scene is None:
            continue

        try:
            if digital_twin is None:
                digital_twin = MetaSejongDigitalTwin()
            digital_twin.apply_scene(scene)
            timestamp = scene.get("timestamp")
            if last_timestamp is None:
                print(
                    "[Meta Sejong] First Zenoh scene received: "
                    f"timestamp={timestamp} people={len(scene['people'])}"
                )
            last_timestamp = timestamp
        except Exception:
            print("[Meta Sejong] Failed to apply Zenoh scene")
            traceback.print_exc()


async def websocket_listener_task():
    import websockets

    digital_twin = MetaSejongDigitalTwin()
    print(f"[Meta Sejong] Connecting to WebSocket {SIM_BACKEND_WS_URL} ...")
    try:
        async with websockets.connect(SIM_BACKEND_WS_URL) as websocket:
            print("[Meta Sejong] WebSocket connected")
            while True:
                data = json.loads(await websocket.recv())
                if data.get("action") != "UpdateTransforms":
                    continue
                for obj in data.get("objects", []):
                    track_id = obj.get("track_id")
                    position = obj.get("position")
                    if track_id is not None and position is not None:
                        digital_twin.apply_legacy_object(track_id, position)
    except websockets.exceptions.ConnectionClosed:
        print("[Meta Sejong] WebSocket connection closed")
    except Exception:
        print("[Meta Sejong] WebSocket listener failed")
        traceback.print_exc()


_zenoh_receiver = None
if SCENE_TRANSPORT in {"zenoh", "both"}:
    _latest_scenes = LatestScene()
    _zenoh_receiver = ZenohSceneReceiver(
        ZENOH_ENDPOINT,
        ZENOH_TOPIC,
        _latest_scenes,
    )
    atexit.register(_zenoh_receiver.close)
    asyncio.ensure_future(zenoh_update_task(_latest_scenes))

if SCENE_TRANSPORT in {"websocket", "both"}:
    asyncio.ensure_future(websocket_listener_task())
