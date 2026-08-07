from __future__ import annotations

from pathlib import Path
import re

import omni.ext
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

from .generator import (
    DICTIONARY_NAMES,
    append_marker_to_tree_usd,
    delete_marker_from_tree_usd,
    save_marker_transform_to_tree_usd,
    update_marker_in_tree_usd,
)


class ArucoBoardExtension(omni.ext.IExt):
    WINDOW_TITLE = "ArUco Marker Tree Generator"

    def on_startup(self, ext_id: str) -> None:
        self._window = ui.Window(self.WINDOW_TITLE, width=580, height=470)
        self._marker_id_model = ui.SimpleIntModel(0)
        self._marker_length_model = ui.SimpleFloatModel(100.0)
        default_path = self._default_output_directory() / "aruco_marker_tree.usd"
        self._tree_path_model = ui.SimpleStringModel(str(default_path))
        self._add_to_stage_model = ui.SimpleBoolModel(True)
        self._spawn_in_front_model = ui.SimpleBoolModel(True)
        self._camera_distance_model = ui.SimpleFloatModel(1.0)
        self._loaded_tree_path = None
        self._loaded_tree_reference_path = None
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
                    ui.Label("Preview tree in current Stage session")
                with ui.HStack(height=36, spacing=8):
                    ui.Button("Add marker to tree", clicked_fn=self._on_add_marker)
                    ui.Button("Load tree", clicked_fn=self._on_load_tree)
                with ui.HStack(height=36, spacing=8):
                    ui.Button(
                        "Save selected pose",
                        clicked_fn=self._on_save_selected_pose,
                    )
                    ui.Button(
                        "Apply fields + pose",
                        clicked_fn=self._on_update_selected_marker,
                    )
                    ui.Button(
                        "Delete selected",
                        clicked_fn=self._on_delete_selected_marker,
                    )
                self._status_label = ui.Label(
                    "Ready (all marker edits are saved only in the tree USD)",
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
                    f"Tree updated: {result.tree_path} ({result.marker_count} markers)"
                )
            print(f"[ArUco Tree] Added {result.marker_prim_path} to {result.tree_path}")
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
            selected_path, placement = self._load_tree_into_current_stage(tree_path)
            self._status_label.text = f"Loaded: {selected_path} ({placement})"
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            print(f"[ArUco Tree] Error: {exc}")

    def _on_save_selected_pose(self) -> None:
        try:
            stage, selected_path, marker_prim_path = self._selected_tree_marker()
            marker = stage.GetPrimAtPath(selected_path)
            local_transform = UsdGeom.Xformable(marker).GetLocalTransformation(
                Usd.TimeCode.Default()
            )
            save_marker_transform_to_tree_usd(
                self._loaded_tree_path,
                marker_prim_path,
                local_transform,
            )
            self._remove_session_prim_spec(stage, selected_path)
            self._reload_tree_layer(stage)
            new_selected_path = self._composed_marker_path(marker_prim_path)
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [new_selected_path], True
            )
            self._status_label.text = f"Saved pose to tree: {marker_prim_path}"
            print(f"[ArUco Tree] Saved pose for {marker_prim_path}")
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            print(f"[ArUco Tree] Error: {exc}")

    def _on_update_selected_marker(self) -> None:
        try:
            stage, selected_path, marker_prim_path = self._selected_tree_marker()
            marker = stage.GetPrimAtPath(selected_path)
            local_transform = UsdGeom.Xformable(marker).GetLocalTransformation(
                Usd.TimeCode.Default()
            )
            dictionary_index = (
                self._dictionary_combo.model.get_item_value_model().as_int
            )
            result = update_marker_in_tree_usd(
                tree_path=self._loaded_tree_path,
                marker_prim_path=marker_prim_path,
                dictionary_name=DICTIONARY_NAMES[dictionary_index],
                marker_id=self._marker_id_model.as_int,
                marker_length_mm=self._marker_length_model.as_float,
                local_transform=local_transform,
            )
            self._remove_session_prim_spec(stage, selected_path)
            self._reload_tree_layer(stage)
            new_selected_path = self._composed_marker_path(result.marker_prim_path)
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [new_selected_path], True
            )
            self._status_label.text = (
                f"Updated {result.marker_prim_path} in {result.tree_path}"
            )
            print(f"[ArUco Tree] Updated {result.marker_prim_path}")
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            print(f"[ArUco Tree] Error: {exc}")

    def _on_delete_selected_marker(self) -> None:
        try:
            stage, selected_path, marker_prim_path = self._selected_tree_marker()
            marker_count = delete_marker_from_tree_usd(
                self._loaded_tree_path,
                marker_prim_path,
            )
            self._remove_session_prim_spec(stage, selected_path)
            self._reload_tree_layer(stage)
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [self._loaded_tree_reference_path], True
            )
            self._status_label.text = (
                f"Deleted {marker_prim_path} from tree ({marker_count} markers remain)"
            )
            print(f"[ArUco Tree] Deleted {marker_prim_path}")
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            print(f"[ArUco Tree] Error: {exc}")

    def _load_tree_into_current_stage(self, tree_path, marker_prim_path=None):
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("No USD Stage is currently open.")

        tree_path = Path(tree_path).resolve()
        tree_asset = str(tree_path)
        tree_stage = Usd.Stage.Open(tree_asset)
        tree_root = tree_stage.GetDefaultPrim() if tree_stage else None
        if (
            tree_root is None
            or not tree_root.IsValid()
            or tree_root.GetPath().pathString != "/ArUcoMarkerTree"
        ):
            raise ValueError(f"USD is not a valid ArUco marker tree: {tree_path}")

        session_layer = stage.GetSessionLayer()
        tree_reference_path = self._find_existing_tree_reference(stage, tree_asset)
        if tree_reference_path is None:
            # The preview reference and all gizmo edits live in the anonymous
            # session layer. The campus root layer is never selected or saved.
            with Usd.EditContext(stage, Usd.EditTarget(session_layer)):
                if not stage.GetPrimAtPath("/World").IsValid():
                    UsdGeom.Xform.Define(stage, "/World")
                safe_tree_name = re.sub(r"[^A-Za-z0-9_]", "_", tree_path.stem)
                instance_path = f"/World/ArUcoMarkerTreeEditor/{safe_tree_name}"
                instance = UsdGeom.Xform.Define(stage, instance_path)
                instance.GetPrim().SetCustomDataByKey("aruco:treeAsset", tree_asset)
                tree_reference_path = f"{instance_path}/Tree"
                tree_reference = UsdGeom.Xform.Define(stage, tree_reference_path)
                references = tree_reference.GetPrim().GetReferences()
                references.ClearReferences()
                references.AddReference(tree_asset)

                meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
                unit_scale = 1.0 / meters_per_unit
                instance.ClearXformOpOrder()
                if abs(unit_scale - 1.0) > 1.0e-9:
                    instance.AddScaleOp().Set(
                        Gf.Vec3f(unit_scale, unit_scale, unit_scale)
                    )

        self._loaded_tree_path = tree_path
        self._loaded_tree_reference_path = tree_reference_path
        # Keep subsequent viewport gizmo changes out of the immutable campus file.
        stage.SetEditTarget(Usd.EditTarget(session_layer))
        self._reload_tree_layer(stage)

        selected_path = tree_reference_path
        if marker_prim_path:
            selected_path = self._composed_marker_path(marker_prim_path)
        markers = stage.GetPrimAtPath(f"{tree_reference_path}/Markers")
        marker_count = len(list(markers.GetChildren())) if markers.IsValid() else 0
        context.get_selection().set_selected_prim_paths([selected_path], True)

        return selected_path, f"{marker_count} marker(s), session edit target"

    @staticmethod
    def _find_existing_tree_reference(stage, tree_asset: str):
        for prim in stage.Traverse():
            if prim.GetCustomDataByKey("aruco:treeAsset") != tree_asset:
                continue
            tree_child = stage.GetPrimAtPath(prim.GetPath().AppendChild("Tree"))
            if tree_child.IsValid():
                return tree_child.GetPath().pathString
            if prim.HasAuthoredReferences():
                return prim.GetPath().pathString
        return None

    def _selected_tree_marker(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD Stage is currently open.")
        if self._loaded_tree_path is None or self._loaded_tree_reference_path is None:
            raise RuntimeError("Load a marker tree before editing a marker.")

        selected_paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not selected_paths:
            raise RuntimeError("Select a marker in the loaded tree first.")
        selected = Sdf.Path(selected_paths[0])
        marker_root = Sdf.Path(f"{self._loaded_tree_reference_path}/Markers")
        if not selected.HasPrefix(marker_root) or selected == marker_root:
            raise ValueError("The selection is not a marker in the loaded tree.")
        relative_path = selected.pathString[len(marker_root.pathString) :].lstrip("/")
        marker_name = relative_path.split("/", 1)[0]
        marker_path = marker_root.AppendChild(marker_name)
        marker = stage.GetPrimAtPath(marker_path)
        if not marker.IsValid():
            raise ValueError(f"Selected marker is unavailable: {marker_path}")
        tree_marker_path = "/ArUcoMarkerTree/Markers/" + marker_path.name
        return stage, marker_path.pathString, tree_marker_path

    def _composed_marker_path(self, marker_prim_path: str) -> str:
        suffix = marker_prim_path.removeprefix("/ArUcoMarkerTree")
        return f"{self._loaded_tree_reference_path}{suffix}"

    @staticmethod
    def _remove_session_prim_spec(stage, prim_path: str) -> None:
        session_layer = stage.GetSessionLayer()
        if session_layer.GetPrimAtPath(prim_path) is None:
            return
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Remove(prim_path))
        if not session_layer.Apply(edit):
            raise RuntimeError(f"Could not clear session edit for {prim_path}")

    def _reload_tree_layer(self, stage) -> None:
        if self._loaded_tree_path is None:
            return
        target = Path(self._loaded_tree_path).resolve()
        for layer in stage.GetUsedLayers():
            if layer.realPath and Path(layer.realPath).resolve() == target:
                layer.Reload()
                break

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
