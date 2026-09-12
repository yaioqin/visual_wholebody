#!/usr/bin/env python3
"""Render a URDF to a PNG without Isaac Gym, OpenGL, or a display server.

The renderer intentionally has only two runtime dependencies: NumPy and
OpenCV.  It resolves mesh paths relative to the URDF, evaluates the URDF joint
tree by joint name, and draws four orthographic views with a CPU painter's
algorithm.  OBJ and binary/ascii STL visual meshes are supported.

Example:
    python render_urdf.py robot.urdf -o robot.png \
        --joint shoulder=1.2 --joint elbow=-0.7
"""

from __future__ import annotations

import argparse
import ast
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in a wrong env
    raise SystemExit(
        "render_urdf.py needs numpy and opencv-python. "
        "Run it with the project's b2z1 conda environment."
    ) from exc


Array = np.ndarray


@dataclass
class Visual:
    link: str
    transform: Array
    triangles: Array
    color: Array


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: Array
    axis: Array


def _vec(text: str | None, default: Sequence[float]) -> Array:
    if not text:
        return np.asarray(default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != len(default):
        raise ValueError(f"expected {len(default)} values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: Array) -> Array:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _transform(xyz: Array | None = None, rpy: Array | None = None) -> Array:
    result = np.eye(4, dtype=np.float64)
    if rpy is not None:
        result[:3, :3] = _rpy_matrix(rpy)
    if xyz is not None:
        result[:3, 3] = xyz
    return result


def _origin(element: ET.Element | None) -> Array:
    if element is None:
        return np.eye(4, dtype=np.float64)
    return _transform(
        _vec(element.get("xyz"), (0.0, 0.0, 0.0)),
        _vec(element.get("rpy"), (0.0, 0.0, 0.0)),
    )


def _axis_angle(axis: Array, angle: float) -> Array:
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    return result


def _load_obj(path: Path) -> Array:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                indices = []
                for token in tokens:
                    raw = int(token.split("/", 1)[0])
                    indices.append(raw - 1 if raw > 0 else len(vertices) + raw)
                # URDF meshes in this repository contain n-gons.  A fan is
                # sufficient for their planar CAD export and keeps this loader
                # independent of a geometry package.
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"OBJ has no renderable faces: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float32)
    return vertex_array[np.asarray(faces, dtype=np.int64)]


def _load_stl(path: Path) -> Array:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * count == len(data):
            dtype = np.dtype(
                [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
            )
            records = np.frombuffer(data, dtype=dtype, count=count, offset=84)
            return records["vertices"].copy()

    vertices: List[Tuple[float, float, float]] = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in parts[1:4]))
    if not vertices or len(vertices) % 3:
        raise ValueError(f"STL has no renderable triangles: {path}")
    return np.asarray(vertices, dtype=np.float32).reshape((-1, 3, 3))


def _box_triangles(size: Array) -> Array:
    x, y, z = size / 2.0
    vertices = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices[faces]


def _cylinder_triangles(radius: float, length: float, segments: int = 32) -> Array:
    vertices: List[Tuple[float, float, float]] = []
    for z in (-length / 2.0, length / 2.0):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    vertices.extend([(0.0, 0.0, -length / 2.0), (0.0, 0.0, length / 2.0)])
    faces: List[Tuple[int, int, int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(
            [
                (index, nxt, segments + nxt),
                (index, segments + nxt, segments + index),
                (2 * segments, nxt, index),
                (2 * segments + 1, segments + index, segments + nxt),
            ]
        )
    return np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]


def _sphere_triangles(radius: float, rings: int = 12, segments: int = 24) -> Array:
    vertices: List[Tuple[float, float, float]] = []
    for ring in range(rings + 1):
        latitude = math.pi * ring / rings
        for segment in range(segments):
            longitude = 2.0 * math.pi * segment / segments
            vertices.append(
                (
                    radius * math.sin(latitude) * math.cos(longitude),
                    radius * math.sin(latitude) * math.sin(longitude),
                    radius * math.cos(latitude),
                )
            )
    faces: List[Tuple[int, int, int]] = []
    for ring in range(rings):
        for segment in range(segments):
            nxt = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + nxt
            c = (ring + 1) * segments + segment
            d = (ring + 1) * segments + nxt
            faces.extend([(a, c, d), (a, d, b)])
    return np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]


def _load_geometry(geometry: ET.Element, urdf_dir: Path, cache: Dict[Path, Array]) -> Array:
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("mesh element has no filename")
        if filename.startswith("package://"):
            raise ValueError(
                f"package URI is ambiguous without a ROS package path: {filename}"
            )
        path = (urdf_dir / filename).resolve()
        if path not in cache:
            suffix = path.suffix.lower()
            if suffix == ".obj":
                cache[path] = _load_obj(path)
            elif suffix == ".stl":
                cache[path] = _load_stl(path)
            else:
                raise ValueError(f"unsupported visual mesh format: {path}")
        triangles = cache[path]
        scale = _vec(mesh.get("scale"), (1.0, 1.0, 1.0)).astype(np.float32)
        return triangles * scale

    box = geometry.find("box")
    if box is not None:
        return _box_triangles(_vec(box.get("size"), (1.0, 1.0, 1.0)))
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return _cylinder_triangles(float(cylinder.get("radius")), float(cylinder.get("length")))
    sphere = geometry.find("sphere")
    if sphere is not None:
        return _sphere_triangles(float(sphere.get("radius")))
    raise ValueError("visual geometry has no supported mesh or primitive")


def _parse_materials(root: ET.Element) -> Dict[str, Array]:
    materials: Dict[str, Array] = {}
    for material in root.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None:
            materials[material.get("name")] = _vec(
                color.get("rgba"), (0.72, 0.74, 0.78, 1.0)
            )[:3]
    return materials


def _visual_color(visual: ET.Element, materials: Dict[str, Array]) -> Array:
    material = visual.find("material")
    if material is None:
        return np.array((0.72, 0.74, 0.78), dtype=np.float64)
    inline = material.find("color")
    if inline is not None:
        return _vec(inline.get("rgba"), (0.72, 0.74, 0.78, 1.0))[:3]
    return materials.get(
        material.get("name", ""), np.array((0.72, 0.74, 0.78), dtype=np.float64)
    )


def _parse_urdf(path: Path) -> Tuple[str, List[Visual], List[Joint], str]:
    root = ET.parse(path).getroot()
    materials = _parse_materials(root)
    mesh_cache: Dict[Path, Array] = {}
    visuals: List[Visual] = []
    links = []
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            raise ValueError("link without a name")
        links.append(name)
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            visuals.append(
                Visual(
                    link=name,
                    transform=_origin(visual.find("origin")),
                    triangles=_load_geometry(geometry, path.parent, mesh_cache),
                    color=_visual_color(visual, materials),
                )
            )

    joints: List[Joint] = []
    child_links = set()
    for element in root.findall("joint"):
        name = element.get("name")
        kind = element.get("type", "fixed")
        parent_element, child_element = element.find("parent"), element.find("child")
        if not name or parent_element is None or child_element is None:
            raise ValueError("joint is missing its name, parent, or child")
        parent, child = parent_element.get("link"), child_element.get("link")
        if not parent or not child:
            raise ValueError(f"joint {name!r} has an empty parent or child")
        axis_element = element.find("axis")
        axis = _vec(axis_element.get("xyz") if axis_element is not None else None, (1, 0, 0))
        joints.append(Joint(name, kind, parent, child, _origin(element.find("origin")), axis))
        child_links.add(child)

    roots = [link for link in links if link not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")
    return root.get("name", path.stem), visuals, joints, roots[0]


def _forward_kinematics(joints: List[Joint], root: str, values: Dict[str, float]) -> Dict[str, Array]:
    transforms: Dict[str, Array] = {root: np.eye(4, dtype=np.float64)}
    pending = list(joints)
    while pending:
        next_pending = []
        made_progress = False
        for joint in pending:
            if joint.parent not in transforms:
                next_pending.append(joint)
                continue
            motion = np.eye(4, dtype=np.float64)
            value = values.get(joint.name, 0.0)
            if joint.kind in ("revolute", "continuous"):
                motion = _axis_angle(joint.axis, value)
            elif joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * value
            elif joint.kind != "fixed":
                raise ValueError(f"unsupported joint type {joint.kind!r} on {joint.name}")
            transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
            made_progress = True
        if not made_progress and next_pending:
            names = ", ".join(joint.name for joint in next_pending)
            raise ValueError(f"joint tree is disconnected or cyclic near: {names}")
        pending = next_pending
    return transforms


def _transform_triangles(triangles: Array, transform: Array) -> Array:
    flat = triangles.reshape((-1, 3)).astype(np.float64, copy=False)
    result = flat @ transform[:3, :3].T + transform[:3, 3]
    return result.reshape((-1, 3, 3)).astype(np.float32)


def _assemble_model(
    visuals: List[Visual], link_transforms: Dict[str, Array], max_triangles: int
) -> Tuple[Array, Array, int]:
    transformed, colors = [], []
    for visual in visuals:
        triangles = _transform_triangles(
            visual.triangles, link_transforms[visual.link] @ visual.transform
        )
        transformed.append(triangles)
        colors.append(np.repeat(visual.color[None, :], len(triangles), axis=0))
    all_triangles = np.concatenate(transformed, axis=0)
    all_colors = np.concatenate(colors, axis=0)
    original_count = len(all_triangles)
    if max_triangles > 0 and original_count > max_triangles:
        selection = np.linspace(0, original_count - 1, max_triangles, dtype=np.int64)
        all_triangles = all_triangles[selection]
        all_colors = all_colors[selection]
    return all_triangles, all_colors, original_count


def _normalize(vector: Array) -> Array:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _camera_basis(direction: Sequence[float]) -> Tuple[Array, Array, Array]:
    # direction points from the model toward the camera.
    toward_camera = _normalize(np.asarray(direction, dtype=np.float64))
    world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(toward_camera, world_up))) > 0.95:
        world_up = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    right = _normalize(np.cross(world_up, toward_camera))
    up = _normalize(np.cross(toward_camera, right))
    return right, up, toward_camera


def _project(points: Array, basis: Tuple[Array, Array, Array], center: Array) -> Array:
    relative = points - center
    return np.stack([relative @ axis for axis in basis], axis=-1)


def _draw_axis_widget(image: Array, basis: Tuple[Array, Array, Array]) -> None:
    height, width = image.shape[:2]
    origin = np.array((62, height - 58), dtype=np.float64)
    axes = (
        (np.array((1.0, 0.0, 0.0)), (80, 95, 255), "X"),
        (np.array((0.0, 1.0, 0.0)), (95, 220, 95), "Y"),
        (np.array((0.0, 0.0, 1.0)), (255, 150, 70), "Z"),
    )
    right, up, _ = basis
    for axis, color, label in axes:
        delta = np.array((np.dot(axis, right), -np.dot(axis, up))) * 34.0
        endpoint = origin + delta
        cv2.arrowedLine(
            image,
            tuple(np.rint(origin).astype(int)),
            tuple(np.rint(endpoint).astype(int)),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        label_at = endpoint + _normalize(delta) * 10.0 if np.linalg.norm(delta) > 1 else endpoint
        cv2.putText(
            image,
            label,
            tuple(np.rint(label_at).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_ground_grid(
    image: Array,
    basis: Tuple[Array, Array, Array],
    center: Array,
    world_to_pixel: float,
    z: float,
    extent: float,
) -> None:
    height, width = image.shape[:2]
    grid_values = np.arange(-extent, extent + 1e-9, 0.2)
    for value in grid_values:
        major = abs(value - round(value)) < 1e-6
        color = (66, 73, 82) if major else (48, 55, 64)
        thickness = 1
        lines = (
            np.array([[-extent, value, z], [extent, value, z]], dtype=np.float64),
            np.array([[value, -extent, z], [value, extent, z]], dtype=np.float64),
        )
        for line in lines:
            projected = _project(line, basis, center)
            pixels = np.empty((2, 2), dtype=np.int32)
            pixels[:, 0] = np.rint(width / 2 + projected[:, 0] * world_to_pixel)
            pixels[:, 1] = np.rint(height / 2 - projected[:, 1] * world_to_pixel)
            cv2.line(image, tuple(pixels[0]), tuple(pixels[1]), color, thickness, cv2.LINE_AA)


def _render_view(
    triangles: Array,
    colors: Array,
    direction: Sequence[float],
    title: str,
    size: Tuple[int, int],
    center: Array,
    model_span: float,
    ground_z: float,
) -> Array:
    width, height = size
    image = np.full((height, width, 3), (31, 36, 43), dtype=np.uint8)
    basis = _camera_basis(direction)
    projected = _project(triangles, basis, center)
    scale = 0.78 * min(width, height) / max(model_span, 1e-6)
    _draw_ground_grid(image, basis, center, scale, ground_z, max(1.0, model_span))

    pixels = np.empty((len(projected), 3, 2), dtype=np.int32)
    pixels[:, :, 0] = np.rint(width / 2 + projected[:, :, 0] * scale)
    pixels[:, :, 1] = np.rint(height / 2 - projected[:, :, 1] * scale)

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = normal_lengths > 1e-11
    normals[valid] /= normal_lengths[valid, None]
    light = _normalize(np.array((-0.35, -0.45, 1.0), dtype=np.float64))
    diffuse = np.abs(normals @ light)
    shade = np.clip(0.38 + 0.72 * diffuse, 0.0, 1.0)
    rgb = np.clip(colors * shade[:, None] * 255.0, 0, 255).astype(np.uint8)
    bgr = rgb[:, ::-1]

    # The third projected coordinate grows toward the camera.  Draw the most
    # distant triangles first, then paint nearer surfaces over them.
    depth = projected[:, :, 2].mean(axis=1)
    order = np.argsort(depth)
    for index in order:
        if not valid[index]:
            continue
        polygon = pixels[index]
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].max() < 0
            or polygon[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(image, polygon, tuple(int(value) for value in bgr[index]), cv2.LINE_AA)

    cv2.putText(
        image,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (232, 236, 241),
        2,
        cv2.LINE_AA,
    )
    _draw_axis_widget(image, basis)
    return image


def render_collage(
    triangles: Array,
    colors: Array,
    output: Path,
    robot_name: str,
    original_count: int,
    width: int,
    pose_label: str,
) -> None:
    points = triangles.reshape((-1, 3))
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    span = float(np.max(bounds_max - bounds_min))
    panel_width, panel_height = width // 2, int(width * 0.39)
    views = (
        ((1.45, -1.6, 1.05), "Isometric"),
        ((1.0, 0.0, 0.12), "Front (+X)"),
        ((0.0, -1.0, 0.10), "Right side (-Y)"),
        ((0.001, 0.0, 1.0), "Top (+Z)"),
    )
    panels = [
        _render_view(
            triangles,
            colors,
            direction,
            title,
            (panel_width, panel_height),
            center,
            span,
            float(bounds_min[2]),
        )
        for direction, title in views
    ]
    gap, header = 8, 92
    collage = np.full(
        (header + 2 * panel_height + gap, 2 * panel_width + gap, 3),
        (20, 24, 30),
        dtype=np.uint8,
    )
    collage[header : header + panel_height, :panel_width] = panels[0]
    collage[header : header + panel_height, panel_width + gap :] = panels[1]
    collage[header + panel_height + gap :, :panel_width] = panels[2]
    collage[header + panel_height + gap :, panel_width + gap :] = panels[3]
    cv2.putText(
        collage,
        robot_name,
        (28, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (241, 244, 247),
        2,
        cv2.LINE_AA,
    )
    subtitle = (
        f"URDF visual geometry | {pose_label} | "
        f"{len(triangles):,}/{original_count:,} triangles rendered"
    )
    cv2.putText(
        collage,
        subtitle,
        (28, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (154, 166, 179),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), collage):
        raise OSError(f"failed to write {output}")


def _joint_argument(value: str) -> Tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("joint values must use NAME=RADIANS")
    name, raw_angle = value.split("=", 1)
    try:
        return name, float(raw_angle)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid angle in {value!r}") from exc


def _joint_values_from_config(path: Path) -> Dict[str, float]:
    """Read default_joint_angles from a Python config without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "default_joint_angles" for target in targets):
            candidates.append(node.value)
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one default_joint_angles assignment in {path}, found {len(candidates)}"
        )
    raw_values = ast.literal_eval(candidates[0])
    if not isinstance(raw_values, dict):
        raise ValueError(f"default_joint_angles is not a dictionary in {path}")
    return {str(name): float(value) for name, value in raw_values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="URDF file to render")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output PNG")
    parser.add_argument(
        "--joint-config",
        type=Path,
        help="Python config containing a literal default_joint_angles dictionary",
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        type=_joint_argument,
        metavar="NAME=RADIANS",
        help="joint position; repeat for multiple joints",
    )
    parser.add_argument(
        "--max-triangles",
        type=int,
        default=0,
        help="uniform draft-preview face budget; 0 keeps every triangle (default: 0)",
    )
    parser.add_argument("--width", type=int, default=1800, help="collage width in pixels")
    args = parser.parse_args()

    urdf = args.urdf.expanduser().resolve()
    if not urdf.is_file():
        parser.error(f"URDF not found: {urdf}")
    if args.width < 800:
        parser.error("--width must be at least 800")
    try:
        values = (
            _joint_values_from_config(args.joint_config.expanduser().resolve())
            if args.joint_config
            else {}
        )
    except (OSError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    values.update(dict(args.joint))
    robot_name, visuals, joints, root = _parse_urdf(urdf)
    known_joints = {joint.name for joint in joints}
    unknown = sorted(set(values) - known_joints)
    if unknown:
        parser.error(f"unknown joint(s): {', '.join(unknown)}")
    transforms = _forward_kinematics(joints, root, values)
    triangles, colors, original_count = _assemble_model(
        visuals, transforms, args.max_triangles
    )
    if args.joint_config:
        pose_label = f"configured pose ({args.joint_config.stem})"
    elif args.joint:
        pose_label = "custom joint pose"
    else:
        pose_label = "zero joint pose"
    render_collage(
        triangles,
        colors,
        args.output.resolve(),
        robot_name,
        original_count,
        args.width,
        pose_label,
    )
    print(
        f"Rendered {len(triangles):,}/{original_count:,} visual triangles "
        f"from {len(visuals)} visual elements to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
