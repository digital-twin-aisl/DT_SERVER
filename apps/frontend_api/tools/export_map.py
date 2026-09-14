"""Export a static USD stage to GLB without Isaac/RTX; keep world registration.

Supports mesh instances, transforms, visibility, normals, UVs, material subsets,
UsdPreviewSurface and base-color textures from the bundled Unreal MDL files.
Procedural MDL/RTX shading is approximated and reported in map.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


class Exporter:
    def __init__(self, source, output, texture_size=1024):
        self.source = Path(source).resolve()
        self.output = Path(output)
        self.stage = Usd.Stage.Open(str(self.source))
        if not self.stage or UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
            raise ValueError('Expected an existing Z-up USD stage')
        errors = self.stage.GetCompositionErrors()
        if errors:
            raise ValueError(f'USD has unresolved composition errors: {errors}')
        self.scale = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.cache = UsdGeom.XformCache()
        self.texture_size = texture_size
        self.binary = bytearray()
        self.doc = {'asset': {'version': '2.0', 'generator': 'Meta Sejong USD exporter'},
                    'scene': 0, 'scenes': [{'nodes': [0]}],
                    'nodes': [{'name': 'USD Z-up to glTF Y-up',
                               'rotation': [-math.sqrt(.5), 0, 0, math.sqrt(.5)], 'children': []}],
                    'meshes': [], 'materials': [], 'textures': [], 'images': [],
                    'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
                    'bufferViews': [], 'accessors': []}
        self.materials = {}
        self.textures = {}
        self.geometries = {}
        self.warnings = set()
        self.bounds_min = np.full(3, np.inf)
        self.bounds_max = -self.bounds_min.copy()
        self.triangles = 0

    def blob(self, data):
        self.binary.extend(b'\0' * (-len(self.binary) % 4))
        index = len(self.doc['bufferViews'])
        self.doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(self.binary), 'byteLength': len(data)})
        self.binary.extend(data)
        return index

    def accessor(self, values, kind='VEC3'):
        values = np.asarray(values, dtype='<f4')
        index = len(self.doc['accessors'])
        self.doc['accessors'].append({'bufferView': self.blob(values.tobytes()), 'componentType': 5126,
                                     'count': len(values), 'type': kind,
                                     'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist()})
        return index

    def texture(self, path):
        path = Path(path)
        key = str(path)
        if key in self.textures:
            return self.textures[key]
        try:
            with Image.open(path) as im:
                im = im.convert('RGBA' if 'A' in im.getbands() else 'RGB')
                im.thumbnail((self.texture_size, self.texture_size))
                data = io.BytesIO()
                im.save(data, format='PNG' if im.mode == 'RGBA' else 'JPEG', quality=85)
                mime = 'image/png' if im.mode == 'RGBA' else 'image/jpeg'
        except (OSError, ValueError):
            self.warnings.add(f'Unsupported or missing texture: {path.name}')
            return None
        index = len(self.doc['textures'])
        self.doc['images'].append({'bufferView': self.blob(data.getvalue()), 'mimeType': mime})
        self.doc['textures'].append({'sampler': 0, 'source': index})
        self.textures[key] = index
        return index

    def material(self, prim):
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        key = str(mat.GetPath()) if mat else '__default__'
        if key in self.materials:
            return self.materials[key]
        color, roughness, metallic, tex = [.65, .68, .70, 1.0], .8, 0., None
        if mat:
            for child in Usd.PrimRange(mat.GetPrim()):
                shader = UsdShade.Shader(child)
                if not shader:
                    continue
                if shader.GetIdAttr().Get() == 'UsdPreviewSurface':
                    for name, default in [('diffuseColor', None), ('roughness', None), ('metallic', None)]:
                        value = shader.GetInput(name).Get() if shader.GetInput(name) else default
                        if value is not None:
                            if name == 'diffuseColor': color[:3] = list(value)
                            elif name == 'roughness': roughness = float(value)
                            else: metallic = float(value)
                    source = shader.GetInput('diffuseColor').GetConnectedSource() if shader.GetInput('diffuseColor') else None
                    if source:
                        asset = UsdShade.Shader(source[0]).GetInput('file').Get()
                        if asset: tex = self.texture(asset.resolvedPath or asset.path)
                asset = child.GetAttribute('info:mdl:sourceAsset').Get()
                if asset:
                    mdl = Path(asset.resolvedPath or self.source.parent / asset.path)
                    if not mdl.is_file():
                        self.warnings.add(f'Missing MDL: {mdl.name}')
                        continue
                    code = mdl.read_text(encoding='utf-8-sig')
                    # This is a base-color approximation, not an MDL interpreter.
                    self.warnings.add('MDL materials approximated as base-color PBR; procedural/RTX effects omitted')
                    matches = re.findall(r'texture_2d\("([^"]+)"\s*,\s*::tex::gamma_srgb', code)
                    if matches:
                        tex = self.texture(mdl.parent / matches[0])
                        if tex is not None: color = [1, 1, 1, 1]
                    match = re.search(r'BaseColor_mdl\s*=\s*float3\(([\d.eE+\-, ]+)\)', code)
                    if match:
                        rgb = [float(x) for x in match[1].split(',')]
                        color[:3] = rgb * 3 if len(rgb) == 1 else rgb
                    for name in ['Roughness', 'Metallic']:
                        match = re.search(name + r'_mdl\s*=\s*([\d.]+);', code)
                        if match:
                            if name == 'Roughness': roughness = float(match[1])
                            else: metallic = float(match[1])
        pbr = {'baseColorFactor': [max(0., min(1., x)) for x in color],
               'roughnessFactor': max(0., min(1., roughness)), 'metallicFactor': max(0., min(1., metallic))}
        if tex is not None: pbr['baseColorTexture'] = {'index': tex}
        value = {'pbrMetallicRoughness': pbr, 'doubleSided': True}
        signature = json.dumps(value, sort_keys=True)
        if signature not in self.materials:
            self.materials[signature] = len(self.doc['materials'])
            self.doc['materials'].append(dict(value, name=mat.GetPrim().GetName() if mat else 'Default'))
        self.materials[key] = self.materials[signature]
        return self.materials[key]

    @staticmethod
    def corners(values, interpolation, point_indices, face_indices, corner_indices):
        values = np.asarray(values)
        if interpolation == 'faceVarying': return values[corner_indices]
        if interpolation in ('vertex', 'varying'): return values[point_indices]
        if interpolation == 'uniform': return values[face_indices]
        return np.repeat(values[:1], len(point_indices), axis=0)

    def mesh(self, prim):
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if not points.size: return
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not len(counts): return
        if np.all(counts == 3):
            corners = np.arange(len(indices)).reshape(-1, 3)
            faces = np.arange(len(counts))
        else:
            corners, faces, offset = [], [], 0
            for face, count in enumerate(counts):
                for k in range(1, count - 1):
                    corners.append([offset, offset + k, offset + k + 1]); faces.append(face)
                offset += count
            corners, faces = np.asarray(corners), np.asarray(faces)
        holes = np.asarray(mesh.GetHoleIndicesAttr().Get() or [], dtype=np.int64)
        keep = ~np.isin(faces, holes)
        corners, faces = corners[keep], faces[keep]
        if not len(faces): return
        if mesh.GetOrientationAttr().Get() == 'leftHanded': corners = corners[:, [0, 2, 1]]
        corner_indices = corners.reshape(-1)
        point_indices = indices[corner_indices]
        face_indices = np.repeat(faces, 3)
        attrs = {'POSITION': (points[point_indices] * self.scale).astype('<f4')}
        normals = mesh.GetNormalsAttr().Get()
        if normals:
            attrs['NORMAL'] = self.corners(normals, mesh.GetNormalsInterpolation(), point_indices, face_indices, corner_indices).astype('<f4')
        else:
            tri = attrs['POSITION'].reshape(-1, 3, 3)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            attrs['NORMAL'] = np.repeat(n, 3, axis=0).astype('<f4')
        uv = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance('st')
        if uv and uv.HasValue():
            coords = self.corners(uv.ComputeFlattened(), uv.GetInterpolation(), point_indices, face_indices, corner_indices).astype('<f4')
            coords[:, 1] = 1 - coords[:, 1]
            attrs['TEXCOORD_0'] = coords
        materials = np.full(len(faces), self.material(prim), dtype=np.int32)
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            materials[np.isin(faces, np.asarray(subset.GetIndicesAttr().Get()))] = self.material(subset.GetPrim())
        digest = hashlib.sha256(materials.tobytes())
        for name, values in attrs.items(): digest.update(name.encode()); digest.update(values.tobytes())
        key = digest.hexdigest()
        if key not in self.geometries:
            primitives = []
            for material in np.unique(materials):
                mask = np.repeat(materials == material, 3)
                primitives.append({'attributes': {name: self.accessor(values[mask], 'VEC2' if name == 'TEXCOORD_0' else 'VEC3') for name, values in attrs.items()}, 'material': int(material)})
            self.geometries[key] = len(self.doc['meshes'])
            self.doc['meshes'].append({'primitives': primitives})
        matrix = np.array(self.cache.GetLocalToWorldTransform(prim))
        world = (points @ matrix[:3, :3] + matrix[3, :3]) * self.scale
        self.bounds_min = np.minimum(self.bounds_min, world.min(axis=0))
        self.bounds_max = np.maximum(self.bounds_max, world.max(axis=0))
        matrix[3, :3] *= self.scale
        node = {'name': prim.GetName(), 'mesh': self.geometries[key], 'matrix': matrix.reshape(-1).tolist(), 'extras': {'usd_path': str(prim.GetPath())}}
        self.doc['nodes'][0]['children'].append(len(self.doc['nodes']))
        self.doc['nodes'].append(node)
        self.triangles += len(faces)

    def deployment(self, path):
        manifest_path = Path(path).resolve()
        manifest = json.loads(manifest_path.read_text())
        calibration_path = manifest_path.parent / manifest['calibration_result']
        calibration = json.loads(calibration_path.read_text())
        cameras = [{k: camera[k] for k in ('camera_id', 'position_m', 'camera_to_world')} for camera in calibration['cameras']]
        ground_path = manifest_path.parent / manifest['scene']['ground_cache']
        with np.load(ground_path, allow_pickle=False) as cache:
            triangles = cache['triangles_mm'] / 1000
        def height(xy):
            a = triangles[:, 0, :2]; b = triangles[:, 1, :2] - a; c = triangles[:, 2, :2] - a
            delta = np.asarray(xy) - a
            det = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
            valid = abs(det) > 1e-10
            safe = np.where(valid, det, 1)
            u = (delta[:, 0] * c[:, 1] - delta[:, 1] * c[:, 0]) / safe
            v = (b[:, 0] * delta[:, 1] - b[:, 1] * delta[:, 0]) / safe
            inside = valid & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
            z = triangles[:, 0, 2] + u * (triangles[:, 1, 2] - triangles[:, 0, 2]) + v * (triangles[:, 2, 2] - triangles[:, 0, 2])
            if not inside.any():
                return float(np.median(triangles[:, :, 2]))
            return float(z[inside].max())
        edges = []
        for edge in manifest['edges']:
            if not edge.get('enabled', True): continue
            polygon = edge.get('workspace', {}).get('polygon_xy_m', [])
            edges.append({'id': edge['id'], 'camera_ids': edge['camera_ids'],
                          'polygon_m': [[*xy, height(xy) + .08] for xy in polygon]})
        points = [p for edge in edges for p in edge['polygon_m']]
        focus = np.mean(points, axis=0).tolist() if points else np.mean([c['position_m'] for c in cameras], axis=0).tolist()
        return {'id': manifest['scene']['id'], 'edges': edges, 'cameras': cameras, 'focus_m': focus}

    def run(self, deployment=None):
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            if str(prim.GetPath()).startswith('/World/MetaSejong_People'): continue
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible':
                if UsdGeom.Imageable(prim).ComputePurpose() == 'guide': continue
                self.mesh(prim)
        if not self.doc['meshes']: raise ValueError('No visible mesh geometry in stage')
        people = self.stage.GetPrimAtPath('/World/MetaSejong_People')
        parent = self.stage.GetPrimAtPath('/World')
        matrix = np.array(self.cache.GetLocalToWorldTransform(people or parent)) if people or parent else np.eye(4)
        matrix[3, :3] *= self.scale
        self.output.mkdir(parents=True, exist_ok=True)
        self.binary.extend(b'\0' * (-len(self.binary) % 4))
        self.doc['buffers'] = [{'byteLength': len(self.binary)}]
        for key in ['images', 'textures', 'materials']:
            if not self.doc[key]: del self.doc[key]
        raw = json.dumps(self.doc, separators=(',', ':'), allow_nan=False).encode()
        raw += b' ' * (-len(raw) % 4)
        glb = self.output / 'map.glb'
        with glb.open('wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, 28 + len(raw) + len(self.binary)))
            f.write(struct.pack('<II', len(raw), 0x4E4F534A)); f.write(raw)
            f.write(struct.pack('<II', len(self.binary), 0x004E4942)); f.write(self.binary)
        metadata = {'schema_version': 1, 'source': self.source.name, 'source_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                    'source_meters_per_unit': self.scale, 'coordinate_system': 'USD world, Z-up',
                    'people_matrix': matrix.reshape(-1).tolist(), 'bounds_m': [self.bounds_min.tolist(), self.bounds_max.tolist()],
                    'mesh_nodes': len(self.doc['nodes']) - 1, 'unique_meshes': len(self.doc['meshes']),
                    'triangles': self.triangles, 'bytes': glb.stat().st_size, 'warnings': sorted(self.warnings)}
        if deployment: metadata['deployment'] = self.deployment(deployment)
        (self.output / 'map.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(metadata, indent=2))
        return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usd', required=True)
    parser.add_argument('--output', default='apps/frontend_api/assets')
    parser.add_argument('--deployment', help='Existing edge AOI/calibration manifest')
    parser.add_argument('--texture-size', type=int, default=1024)
    args = parser.parse_args()
    if args.texture_size < 1: parser.error('--texture-size must be positive')
    Exporter(args.usd, args.output, args.texture_size).run(args.deployment)
