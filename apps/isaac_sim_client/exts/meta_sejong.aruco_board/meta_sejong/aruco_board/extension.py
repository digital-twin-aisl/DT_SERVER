from __future__ import annotations

from pathlib import Path
import re

import omni.ext
import omni.ui as ui
import omni.usd
from pxr import Gf, Usd, UsdGeom

from .generator import DICTIONARY_NAMES, append_marker_to_tree_usd


class ArucoBoardExtension(omni.ext.IExt):
    WINDOW_TITLE = "ArUco Marker Tree Generator"

    def on_startup(self, ext_id: str) -> None:
        self._window = ui.Window(self.WINDOW_TITLE, width=580, height=390)
        self._marker_id_model = ui.SimpleIntModel(0)
        self._marker_length_model = ui.SimpleFloatModel(100.0)
        default_path = self._default_output_directory() / "aruco_marker_tree.usd"
        self._tree_path_model = ui.SimpleStringModel(str(default_path))
        self._add_to_stage_model = ui.SimpleBoolModel(True)
        self._spawn_in_front_model = ui.SimpleBoolModel(True)
        self._camera_distance_model = ui.SimpleFloatModel(1.0)
        self._status_label = None
        self._build_ui()

    @staticmethod
    def _default_output_directory() -> Path:
        # extension.py -> aruco_board -> meta_sejong -> extension root -> exts
        # -> isaac_sim_client
        return Path(__file__).resolve().parents[4] / "aruco_boards"

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Add ArUco marker boards to one reusable USD tree", height=24)
                with ui.HStack(height=28):
                    ui.Label("Dictionary", width=150)
                    self._dictionary_combo = ui.ComboBox(0, *DICTIONARY_NAMES)
                with ui.HStack(height=28):
                    ui.Label("Marker ID", width=150)
                    ui.IntField(model=self._marker_id_model)
                with ui.HStack(height=28):
                    ui.Label("Marker length (mm)", width=150)
                    ui.FloatField(model=self._marker_length_model)
                with ui.HStack(height=28):
                    ui.Label("Marker tree USD", width=150)
                    ui.StringField(model=self._tree_path_model)
                with ui.HStack(height=28):
                    ui.Label("Camera placement", width=150)
                    ui.CheckBox(model=self._spawn_in_front_model, width=22)
                    ui.Label("In front", width=70)
                    ui.Label("Distance (m)", width=88)
                    ui.FloatField(model=self._camera_distance_model, width=90)
                with ui.HStack(height=24):
                    ui.Spacer(width=150)
                    ui.CheckBox(model=self._add_to_stage_model, width=22)
                    ui.Label("Load/update tree in current Stage")
                with ui.HStack(height=36, spacing=8):
                    ui.Button(
                        "Add marker to tree", clicked_fn=self._on_add_marker
                    )
                    ui.Button("Load tree", clicked_fn=self._on_load_tree)
                self._status_label = ui.Label(
                    "Ready (each tree USD is one marker collection)",
                    word_wrap=True,
                    height=52,
                )

    def _on_add_marker(self) -> None:
        try:
            dictionary_index = (
                self._dictionary_combo.model.get_item_value_model().as_int
            )
            position_m = None
            orientation_wxyz = None
            if self._spawn_in_front_model.as_bool:
                distance_m = self._camera_distance_model.as_float
                if distance_m <= 0:
                    raise ValueError("Camera distance must be greater than 0 m.")
                position_m, orientation_wxyz = self._marker_pose_in_front_of_camera(
                    distance_m
                )
            result = append_marker_to_tree_usd(
                dictionary_name=DICTIONARY_NAMES[dictionary_index],
                marker_id=self._marker_id_model.as_int,
                marker_length_mm=self._marker_length_model.as_float,
                tree_path=self._tree_path_model.as_string,
                position_m=position_m,
                orientation_wxyz=orientation_wxyz,
            )
            if self._add_to_stage_model.as_bool:
                try:
                    selected_path, placement = self._load_tree_into_current_stage(
                        result.tree_path,
                        result.marker_prim_path,
                    )
                    self._status_label.text = (
                        f"Tree has {result.marker_count} marker(s): {selected_path} "
                        f"({placement})"
                    )
                except Exception as stage_exc:
                    self._status_label.text = (
                        f"Tree updated, but Stage load failed: {stage_exc}"
                    )
                    print(f"[ArUco Tree] Stage load failed: {stage_exc}")
            else:
                self._status_label.text = (
                    f"Tree updated: {result.tree_path} "
                    f"({result.marker_count} markers)"
                )
            print(
                f"[ArUco Tree] Added {result.marker_prim_path} to {result.tree_path}"
            )
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            print(f"[ArUco Tree] Error: {exc}")

    def _on_load_tree(self) -> None:
        try:
            tree_path = Path(self._tree_path_model.as_string).expanduser().resolve()
            if tree_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
                tree_path = tree_path.with_suffix(".usd")
            if not tree_path.exists():
                raise FileNotFoundError(f"Marker tree does not exist: {tree_path}")
            selected_path, placement = self._load_tree_into_current_stage(
                tree_path
            )
            self._status_label.text = f"Loaded: {selected_path} ({placement})"
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            print(f"[ArUco Tree] Error: {exc}")

    def _load_tree_into_current_stage(self, tree_path, marker_prim_path=None):
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("No USD Stage is currently open.")

        if not stage.GetPrimAtPath("/World").IsValid():
            UsdGeom.Xform.Define(stage, "/World")
        tree_group = UsdGeom.Xform.Define(
            stage, "/World/ArUcoMarkerTrees"
        ).GetPrim()

        tree_path = Path(tree_path).resolve()
        tree_asset = str(tree_path)
        instance = None
        for child in tree_group.GetChildren():
            if child.GetCustomDataByKey("aruco:treeAsset") == tree_asset:
                instance = UsdGeom.Xform(child)
                break

        is_new_instance = instance is None
        if is_new_instance:
            safe_tree_name = re.sub(r"[^A-Za-z0-9_]", "_", tree_path.stem)
            base_path = f"/World/ArUcoMarkerTrees/{safe_tree_name}"
            instance_path = base_path
            suffix = 2
            while stage.GetPrimAtPath(instance_path).IsValid():
                instance_path = f"{base_path}_{suffix}"
                suffix += 1
            instance = UsdGeom.Xform.Define(stage, instance_path)
            instance.GetPrim().SetCustomDataByKey("aruco:treeAsset", tree_asset)
        else:
            instance_path = instance.GetPath().pathString

        # Keep scene placement on a wrapper and the referenced tree on a child.
        # This prevents camera-preview transforms from overriding transforms
        # authored in the tree asset itself. Migrate instances made by v0.1 too.
        instance.GetPrim().GetReferences().ClearReferences()
        tree_reference_path = f"{instance_path}/Tree"
        tree_reference = UsdGeom.Xform.Define(stage, tree_reference_path)
        if is_new_instance or not tree_reference.GetPrim().HasAuthoredReferences():
            tree_reference.GetPrim().GetReferences().AddReference(tree_asset)

        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        unit_scale = 1.0 / meters_per_unit
        # The wrapper remains at the origin. Every marker's pose lives in the tree
        # asset itself, so loading always restores those authored child transforms.
        instance.ClearXformOpOrder()
        if abs(unit_scale - 1.0) > 1.0e-9:
            instance.AddScaleOp().Set(Gf.Vec3f(unit_scale, unit_scale, unit_scale))

        selected_path = instance_path
        if marker_prim_path:
            marker_relative_path = marker_prim_path.removeprefix("/ArUcoMarkerTree")
            selected_path = f"{tree_reference_path}{marker_relative_path}"
        context.get_selection().set_selected_prim_paths([selected_path], True)

        return selected_path, "marker transforms restored from tree"

    def _marker_pose_in_front_of_camera(self, distance_m: float):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD Stage is currently open.")
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        position, orientation = self._pose_in_front_of_camera(
            stage, distance_m / meters_per_unit
        )
        position_m = tuple(float(value) * meters_per_unit for value in position)
        imaginary = orientation.GetImaginary()
        orientation_wxyz = (
            float(orientation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
        return position_m, orientation_wxyz

    @staticmethod
    def _pose_in_front_of_camera(stage, distance_in_stage_units: float):
        from omni.kit.viewport.utility import get_active_viewport_camera_path

        camera_path = get_active_viewport_camera_path()
        if not camera_path:
            raise RuntimeError("No active Viewport camera was found.")
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"Active camera Prim is unavailable: {camera_path}")

        camera_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        camera_position = camera_world.ExtractTranslation()
        camera_forward = camera_world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        camera_forward.Normalize()
        position = camera_position + camera_forward * distance_in_stage_units
        orientation = camera_world.ExtractRotationQuat()
        return position, orientation

    def on_shutdown(self) -> None:
        self._status_label = None
        if self._window:
            self._window.visible = False
        self._window = None
