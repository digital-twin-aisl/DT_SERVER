# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Playback-only USD geometry in an owned anonymous session sublayer."""

import math
from uuid import uuid4

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

LIMBS = ((0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (0, 9), (9, 10),
         (10, 11), (2, 6), (2, 12), (6, 7), (7, 8), (12, 13), (13, 14))
COLORS = ((0.90, 0.10, 0.29), (0.24, 0.71, 0.29), (1.00, 0.88, 0.10),
          (0.00, 0.51, 0.78), (0.96, 0.51, 0.19), (0.57, 0.12, 0.71))


class SceneRenderer:
    def __init__(self, stage):
        self.stage = stage
        self.units = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(self.units) or self.units <= 0:
            raise ValueError("Stage metersPerUnit must be positive")
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise ValueError("Open a Z-up USD stage matching the recorded world coordinates")
        # A top-level root avoids inheriting transforms on the building's /World.
        self.root_path = f"/MetaSejong_Playback_{uuid4().hex[:12]}"
        self.layer = Sdf.Layer.CreateAnonymous("meta_sejong_scene_playback.usda")
        self.people = {}
        self.stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        try:
            with Usd.EditContext(stage, self.layer):
                UsdGeom.Xform.Define(stage, self.root_path)
        except Exception:
            self.close()
            raise

    def _person(self, identity):
        if identity in self.people:
            return self.people[identity]
        path = f"{self.root_path}/Person_{identity}"
        person = UsdGeom.Xform.Define(self.stage, path)
        translate = person.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        color = Gf.Vec3f(*COLORS[identity % len(COLORS)])
        material = UsdShade.Material.Define(self.stage, path + "/Material")
        shader = UsdShade.Shader.Define(self.stage, path + "/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(person.GetPrim()).Bind(material)

        capsule = UsdGeom.Capsule.Define(self.stage, path + "/Root")
        capsule.CreateHeightAttr(1.7 / self.units)
        capsule.CreateRadiusAttr(0.3 / self.units)
        capsule.CreateAxisAttr(UsdGeom.Tokens.z)
        capsule.CreateDisplayColorAttr([color])
        skeleton = UsdGeom.Xform.Define(self.stage, path + "/Skeleton")
        joints = []
        for index in range(15):
            joint = UsdGeom.Sphere.Define(self.stage, path + f"/Skeleton/Joint_{index:02d}")
            joint.CreateRadiusAttr(0.04 / self.units)
            joint.CreateDisplayColorAttr([color])
            joints.append(joint.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble))
        bones = UsdGeom.BasisCurves.Define(self.stage, path + "/Skeleton/Bones")
        bones.CreateTypeAttr(UsdGeom.Tokens.linear)
        bones.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        bones.CreateCurveVertexCountsAttr([2] * len(LIMBS))
        bones.CreateWidthsAttr([0.036 / self.units])
        bones.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        bones.CreateDisplayColorAttr([color])
        state = (person, translate, capsule, skeleton, joints, bones)
        self.people[identity] = state
        return state

    def apply(self, scene):
        active = set()
        with Usd.EditContext(self.stage, self.layer):
            for item in scene["people"]:
                identity = item["global_id"]
                active.add(identity)
                person, translate, capsule, skeleton, joints, bones = self._person(identity)
                UsdGeom.Imageable(person.GetPrim()).MakeVisible()
                root = item["root"]["position"]
                scale = 1000.0 * self.units
                translate.Set(Gf.Vec3d(*(value / scale for value in root)))
                pose = item.get("pose")
                if pose is None:
                    UsdGeom.Imageable(capsule.GetPrim()).MakeVisible()
                    UsdGeom.Imageable(skeleton.GetPrim()).MakeInvisible()
                else:
                    points = [Gf.Vec3d(*((joint[k] - root[k]) / scale for k in range(3)))
                              for joint in pose["joints"]]
                    for op, point in zip(joints, points):
                        op.Set(point)
                    bones.CreatePointsAttr([Gf.Vec3f(points[k]) for pair in LIMBS for k in pair])
                    UsdGeom.Imageable(capsule.GetPrim()).MakeInvisible()
                    UsdGeom.Imageable(skeleton.GetPrim()).MakeVisible()
            for identity, state in self.people.items():
                if identity not in active:
                    UsdGeom.Imageable(state[0].GetPrim()).MakeInvisible()

    def close(self):
        if self.layer is not None:
            paths = self.stage.GetSessionLayer().subLayerPaths
            if self.layer.identifier in paths:
                paths.remove(self.layer.identifier)
            self.layer = None
        self.people.clear()
