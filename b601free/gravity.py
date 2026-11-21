"""Gravity torque g(q) from a URDF, pure numpy.

Port of Seeed's `lerobot_robot_seeed_b601/gravity.py` (plugin v1.0.5). The maths and the joint
numbering are unchanged. Only the yaml/project-root URDF lookup was dropped: pass a URDF path
(default: the plugin's bundled `00-arm-rs_asm-v3.urdf`, copied into b601free/assets/).

For each movable joint i (world axis a_i, pivot p_i) with downstream links subtree(i):
    g_i(q) = -a_i . sum_{j in subtree(i)} (c_j - p_i) x (m_j * g_vec)
Returns N*m, usable directly as the MIT-mode tau feed-forward. q is in radians, in the motor frame
(the plugin feeds raw `state.pos`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EARTH_GRAVITY: tuple[float, float, float] = (0.0, 0.0, -9.81)
DEFAULT_URDF = str(Path(__file__).resolve().parent / "assets" / "00-arm-rs_asm-v3.urdf")
_MODEL_CACHE: dict[str, "PurePyModel"] = {}


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rpy_to_R(roll, pitch, yaw):
    """URDF rpy -> rotation matrix, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def _axis_angle_R(axis, angle):
    """Rodrigues formula."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    x, y, z = a
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


@dataclass
class _LinkInfo:
    name: str
    mass: float
    com: np.ndarray  # centre of mass in the link frame (3,)


@dataclass
class _JointInfo:
    name: str
    jtype: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    parent: str
    child: str
    axis: np.ndarray
    q_idx: int = -1  # generalised-coordinate index of a movable joint; -1 for fixed


@dataclass
class PurePyModel:
    nq: int
    nv: int
    links: dict[str, _LinkInfo] = field(default_factory=dict)
    joints: list[_JointInfo] = field(default_factory=list)
    root: str = ""
    children: dict[str, list[int]] = field(default_factory=dict)


def _parse_floats(text, n):
    vals = [float(v) for v in text.replace(",", " ").split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {len(vals)}: {text!r}")
    return np.array(vals, dtype=float)


def _parse_urdf(path: str) -> PurePyModel:
    root = ET.parse(path).getroot()
    links: dict[str, _LinkInfo] = {}
    for link_el in root.findall("link"):
        name = link_el.get("name", "")
        mass, com = 0.0, np.zeros(3)
        inertial = link_el.find("inertial")
        if inertial is not None:
            m_el = inertial.find("mass")
            if m_el is not None:
                mass = float(m_el.get("value", "0"))
            o_el = inertial.find("origin")
            if o_el is not None and o_el.get("xyz"):
                com = _parse_floats(o_el.get("xyz", "0 0 0"), 3)
        links[name] = _LinkInfo(name=name, mass=mass, com=com)

    joints: list[_JointInfo] = []
    for j_el in root.findall("joint"):
        o_el = j_el.find("origin")
        xyz = _parse_floats(o_el.get("xyz", "0 0 0"), 3) if o_el is not None else np.zeros(3)
        rpy = _parse_floats(o_el.get("rpy", "0 0 0"), 3) if o_el is not None else np.zeros(3)
        parent = j_el.find("parent").get("link", "")
        child = j_el.find("child").get("link", "")
        a_el = j_el.find("axis")
        axis = _parse_floats(a_el.get("xyz", "0 0 1"), 3) if a_el is not None else np.array([0, 0, 1.0])
        nrm = float(np.linalg.norm(axis))
        axis = axis / nrm if nrm > 1e-12 else np.array([0, 0, 1.0])
        joints.append(_JointInfo(name=j_el.get("name", ""), jtype=j_el.get("type", "fixed"), origin_xyz=xyz,
                                 origin_rpy=rpy, parent=parent, child=child, axis=axis))

    child_links = {j.child for j in joints}
    root_candidates = [name for name in links if name not in child_links]
    if not root_candidates:
        raise ValueError("URDF has no root link")
    root_link = root_candidates[0]

    # base -> tip DFS order; movable joints are numbered in this order
    children_traverse: dict[str, list[int]] = {name: [] for name in links}
    for i, j in enumerate(joints):
        children_traverse.setdefault(j.parent, []).append(i)
    ordered: list[_JointInfo] = []

    def _dfs(link: str) -> None:
        for ji in children_traverse.get(link, []):
            j = joints[ji]
            ordered.append(j)
            _dfs(j.child)

    _dfs(root_link)
    movable = [j for j in ordered if j.jtype in ("revolute", "prismatic", "continuous")]
    for idx, j in enumerate(movable):
        j.q_idx = idx
    children: dict[str, list[int]] = {name: [] for name in links}
    for idx, j in enumerate(ordered):
        children[j.parent].append(idx)
    return PurePyModel(nq=len(movable), nv=len(movable), links=links, joints=ordered, root=root_link,
                       children=children)


def _build_fast_cache(model: PurePyModel) -> dict:
    names = list(model.links.keys())
    idx = {n: i for i, n in enumerate(names)}
    N, J = len(names), len(model.joints)
    mass = np.array([model.links[n].mass for n in names], dtype=float)
    com = np.array([model.links[n].com for n in names], dtype=float)
    parent_link, child_link = np.zeros(J, dtype=int), np.zeros(J, dtype=int)
    R_origin, origin_xyz, axis = np.zeros((J, 3, 3)), np.zeros((J, 3)), np.zeros((J, 3))
    q_idx = np.full(J, -1, dtype=int)
    is_rev, is_pris = np.zeros(J, dtype=bool), np.zeros(J, dtype=bool)
    for j, ji in enumerate(model.joints):
        parent_link[j], child_link[j] = idx[ji.parent], idx[ji.child]
        R_origin[j] = _rpy_to_R(*ji.origin_rpy)
        origin_xyz[j], axis[j], q_idx[j] = ji.origin_xyz, ji.axis, ji.q_idx
        is_rev[j] = ji.jtype in ("revolute", "continuous")
        is_pris[j] = ji.jtype == "prismatic"
    children_of_link: dict[str, list[int]] = {n: [] for n in names}
    for ji in model.joints:
        children_of_link[ji.parent].append(idx[ji.child])

    def _subtree_idx(r: int) -> np.ndarray:
        out, stack = [], [r]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(children_of_link[names[x]])
        return np.array(out, dtype=int)

    sub_idx = [_subtree_idx(int(child_link[j])) for j in range(J)]
    return {"N": N, "J": J, "nq": int((q_idx >= 0).sum()), "mass": mass, "com": com,
            "parent_link": parent_link, "child_link": child_link, "R_origin": R_origin,
            "origin_xyz": origin_xyz, "axis": axis, "q_idx": q_idx, "is_rev": is_rev, "is_pris": is_pris,
            "sub_idx": sub_idx, "root_idx": idx[model.root]}


def _get_fast_cache(model: PurePyModel) -> dict:
    fc = getattr(model, "_fast", None)
    if fc is None:
        fc = _build_fast_cache(model)
        object.__setattr__(model, "_fast", fc)
    return fc


def _pad_q(model: PurePyModel, q) -> np.ndarray:
    if q is None:
        return np.zeros(model.nq)
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] == model.nq:
        return q
    padded = np.zeros(model.nq)
    padded[: min(q.shape[0], model.nq)] = q[: min(q.shape[0], model.nq)]
    return padded


def load_dynamics_model(urdf_path: Optional[str] = None) -> PurePyModel:
    """Parse (cached per path) and warm up the vectorised fast path."""
    path = str(urdf_path or DEFAULT_URDF)
    cached = _MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    model = _parse_urdf(path)
    compute_generalized_gravity(model=model, q=np.zeros(model.nq))
    _MODEL_CACHE[path] = model
    return model


def get_default_gravity() -> np.ndarray:
    return np.array(EARTH_GRAVITY)


def compute_generalized_gravity(model: Optional[PurePyModel] = None, q: Optional[np.ndarray] = None,
                                data: Optional[object] = None, gravity: Optional[np.ndarray] = None) -> np.ndarray:
    """Generalised gravity vector g(q), shape (nq,), N*m."""
    if model is None:
        model = load_dynamics_model()
    q = _pad_q(model, q)
    g_vec = np.asarray(gravity, dtype=float) if gravity is not None else get_default_gravity()
    fc = _get_fast_cache(model)
    N, J = fc["N"], fc["J"]
    parent_link, child_link, R_origin = fc["parent_link"], fc["child_link"], fc["R_origin"]
    origin_xyz, axis, q_idx, is_rev, is_pris = fc["origin_xyz"], fc["axis"], fc["q_idx"], fc["is_rev"], fc["is_pris"]

    R_world = np.zeros((N, 3, 3))
    R_world[fc["root_idx"]] = np.eye(3)
    t_world = np.zeros((N, 3))
    for j in range(J):
        p = parent_link[j]
        Rp, tp = R_world[p], t_world[p]
        R_fixed = Rp @ R_origin[j]
        t_fixed = tp + Rp @ origin_xyz[j]
        qi = q_idx[j]
        if is_rev[j]:
            R_child, t_child = R_fixed @ _axis_angle_R(axis[j], q[qi]), t_fixed
        elif is_pris[j]:
            R_child, t_child = R_fixed, t_fixed + R_fixed @ (axis[j] * q[qi])
        else:
            R_child, t_child = R_fixed, t_fixed
        cl = child_link[j]
        R_world[cl], t_world[cl] = R_child, t_child

    com_world = np.einsum("nij,nj->ni", R_world, fc["com"]) + t_world
    mass, sub_idx = fc["mass"], fc["sub_idx"]
    tau = np.zeros(fc["nq"])
    for j in range(J):
        qi = q_idx[j]
        if qi < 0:
            continue
        p = parent_link[j]
        Rp, tp = R_world[p], t_world[p]
        axis_world = (Rp @ R_origin[j]) @ axis[j]
        p_i = tp + Rp @ origin_xyz[j]
        s = sub_idx[j]
        if is_rev[j]:
            moment = np.cross(com_world[s] - p_i, mass[s][:, None] * g_vec).sum(axis=0)
            tau[qi] = -float(axis_world @ moment)
        else:
            tau[qi] = -float(axis_world @ (mass[s][:, None] * g_vec).sum(axis=0))
    return tau


# ---------------------------------------------------------------------- identification (NEW, not in the plugin)
def _link_frames(model: PurePyModel, q: np.ndarray):
    """World rotation/translation of every link frame at q (same kinematics as compute_generalized_gravity)."""
    fc = _get_fast_cache(model)
    N, J = fc["N"], fc["J"]
    R_world, t_world = np.zeros((N, 3, 3)), np.zeros((N, 3))
    R_world[fc["root_idx"]] = np.eye(3)
    for j in range(J):
        p = fc["parent_link"][j]
        R_fixed = R_world[p] @ fc["R_origin"][j]
        t_fixed = t_world[p] + R_world[p] @ fc["origin_xyz"][j]
        qi = fc["q_idx"][j]
        if fc["is_rev"][j]:
            R_c, t_c = R_fixed @ _axis_angle_R(fc["axis"][j], q[qi]), t_fixed
        elif fc["is_pris"][j]:
            R_c, t_c = R_fixed, t_fixed + R_fixed @ (fc["axis"][j] * q[qi])
        else:
            R_c, t_c = R_fixed, t_fixed
        R_world[fc["child_link"][j]], t_world[fc["child_link"][j]] = R_c, t_c
    return R_world, t_world


def gravity_regressor(model: PurePyModel, q, gravity: Optional[np.ndarray] = None) -> np.ndarray:
    """Y(q), shape (nq, 4N): g(q) = Y(q) @ theta, theta = per link [m, m*com_x, m*com_y, m*com_z] (link frame).
    Gravity torque is linear in these, so they can be fitted by least squares from measured holding torques."""
    q = _pad_q(model, q)
    g_vec = np.asarray(gravity, dtype=float) if gravity is not None else get_default_gravity()
    fc = _get_fast_cache(model)
    R_world, t_world = _link_frames(model, q)
    Y = np.zeros((fc["nq"], 4 * fc["N"]))
    for j in range(fc["J"]):
        qi = fc["q_idx"][j]
        if qi < 0:
            continue
        p = fc["parent_link"][j]
        axis_world = (R_world[p] @ fc["R_origin"][j]) @ fc["axis"][j]
        if fc["is_pris"][j]:
            for l in fc["sub_idx"][j]:
                Y[qi, 4 * l] = -axis_world @ g_vec
            continue
        p_i = t_world[p] + R_world[p] @ fc["origin_xyz"][j]
        for l in fc["sub_idx"][j]:
            Y[qi, 4 * l] = -axis_world @ np.cross(t_world[l] - p_i, g_vec)
            for k in range(3):
                Y[qi, 4 * l + 1 + k] = -axis_world @ np.cross(R_world[l][:, k], g_vec)
    return Y


def link_params(model: PurePyModel) -> np.ndarray:
    fc = _get_fast_cache(model)
    return np.concatenate([np.r_[m, m * c] for m, c in zip(fc["mass"], fc["com"])])


def apply_link_params(model: PurePyModel, theta: np.ndarray) -> None:
    """Write fitted [m, m*com] back into the model (in place; affects every user of this cached model)."""
    fc = _get_fast_cache(model)
    th = np.asarray(theta, dtype=float).reshape(-1, 4)
    for l, (m, hx, hy, hz) in enumerate(th):
        m = max(float(m), 1e-4)
        fc["mass"][l] = m
        fc["com"][l] = np.array([hx, hy, hz]) / m


def link_names(model: PurePyModel) -> list[str]:
    return list(model.links.keys())
