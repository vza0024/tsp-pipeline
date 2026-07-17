"""
sim/renderer.py — Renderer using objects from objects.zip (tests_v11).

All object creation and animation code taken directly from:
  test1_spheres.py, test2_tiles.py, test3_cylinders.py, test4_books.py
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

from sim.display_utils import (
    make_static,
    render_equivalence,
    render_ratio_equivalence,
)

LOGGER = logging.getLogger(__name__)
SIM_HOST = os.getenv("COPPELIASIM_HOST", "localhost")
SIM_PORT = int(os.getenv("COPPELIASIM_ZMQ_PORT", "23000"))

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    RemoteAPIClient = None  # type: ignore[assignment]

# ── CONSTANTS (from objects.zip test files) ───────────────────────────────────

TABLE_Z = 0.05
LABEL_C = [0.0, 0.0, 0.0]

# Sphere constants
SPHERE_R = 0.035

# Tile constants
TILE = 0.08
GAP = 0.005
TILE_H = 0.01

# Cylinder constants
CYL_R = 0.03
CYL_H = 0.1

# Book constants
BOOK_W = 0.06
BOOK_D = 0.08
BOOK_H = 0.015
COVER_H = 0.002
PAGES_H = 0.011
SPINE_W = 0.004
PAGES_COLOR = [0.95, 0.93, 0.88]

ALL_COVERS = [
    [0.15, 0.25, 0.70],
    [0.75, 0.10, 0.10],
    [0.10, 0.50, 0.15],
    [0.45, 0.10, 0.55],
    [0.50, 0.30, 0.10],
    [0.10, 0.45, 0.50],
    [0.85, 0.45, 0.05],
    [0.60, 0.10, 0.30],
    [0.30, 0.30, 0.60],
]

RAINBOW = [
    [0.20, 0.60, 1.00],
    [1.00, 0.50, 0.10],
    [0.20, 0.85, 0.35],
    [0.80, 0.20, 0.80],
    [0.90, 0.75, 0.10],
    [0.10, 0.80, 0.70],
    [0.90, 0.30, 0.30],
]


# ── SIM SETUP ────────────────────────────────────────────────────────────────


def _connect() -> Any:
    """Connect to CoppeliaSim and start the simulation if necessary."""
    if RemoteAPIClient is None:
        raise RuntimeError(
            "CoppeliaSim client is not installed. Install coppeliasim-zmqremoteapi-client."
        )

    client = RemoteAPIClient(host=SIM_HOST, port=SIM_PORT)
    sim = client.require("sim")
    try:
        if sim.getSimulationState() != 17:  # 17 is the running state.
            sim.startSimulation()
            time.sleep(2.0)
    except Exception:
        LOGGER.debug("Could not query simulation state; attempting to start it", exc_info=True)
        sim.startSimulation()
        time.sleep(2.0)
    return sim


def _clear_scene(sim: Any) -> None:
    """Remove shape objects left by the previous question."""
    try:
        handles = sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0)
    except Exception:
        LOGGER.warning("Unable to enumerate CoppeliaSim scene objects", exc_info=True)
        return

    for handle in handles:
        try:
            sim.removeObjects([handle])
        except Exception:
            LOGGER.debug("Unable to remove CoppeliaSim object %s", handle, exc_info=True)
    time.sleep(0.5)


# ── OBJECT CREATORS (from objects.zip) ────────────────────────────────────────


def _sphere(sim, name, pos, color):
    """From test1_spheres.py"""
    h = sim.createPrimitiveShape(
        sim.primitiveshape_spheroid, [SPHERE_R * 2, SPHERE_R * 2, SPHERE_R * 2], 0
    )
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, -1, pos)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(sim, h)
    return h


def _half_sphere(sim, name, pos, color):
    """From test1_spheres.py"""
    h = sim.createPrimitiveShape(
        sim.primitiveshape_spheroid, [SPHERE_R * 2, SPHERE_R * 2, SPHERE_R], 0
    )
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, -1, pos)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(sim, h)
    return h


def _tile(sim, name, pos, color):
    """From test2_tiles.py"""
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [TILE, TILE, TILE_H], 0)
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, -1, pos)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(sim, h)
    return h


def _cyl(sim, name, pos, color, r=None, h_=None):
    """From test3_cylinders.py"""
    r = r or CYL_R
    h_ = h_ or CYL_H
    handle = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [r * 2, r * 2, h_], 0)
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, -1, pos)
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(sim, handle)
    return handle


def _build_book(sim, name, pos, cover_color, z_offset=0):
    """From test4_books.py — build_book()"""
    x, y = pos[0], pos[1]
    base_z = TABLE_Z + z_offset
    parts = {}
    all_p = []

    bot = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [BOOK_W, BOOK_D, COVER_H], 0)
    sim.setObjectAlias(bot, f"{name}_bot")
    sim.setObjectPosition(bot, -1, [x, y, base_z + COVER_H * 0.5])
    sim.setShapeColor(bot, None, sim.colorcomponent_ambient_diffuse, cover_color)
    make_static(sim, bot)
    parts["bot"] = bot
    all_p.append(bot)

    pg = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid, [BOOK_W - 0.004, BOOK_D - 0.004, PAGES_H], 0
    )
    sim.setObjectAlias(pg, f"{name}_pg")
    sim.setObjectPosition(pg, -1, [x, y, base_z + COVER_H + PAGES_H * 0.5])
    sim.setShapeColor(pg, None, sim.colorcomponent_ambient_diffuse, PAGES_COLOR)
    make_static(sim, pg)
    parts["pages"] = pg
    all_p.append(pg)

    top = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [BOOK_W, BOOK_D, COVER_H], 0)
    sim.setObjectAlias(top, f"{name}_top")
    top_z = base_z + COVER_H + PAGES_H + COVER_H * 0.5
    sim.setObjectPosition(top, -1, [x, y, top_z])
    sim.setShapeColor(top, None, sim.colorcomponent_ambient_diffuse, cover_color)
    make_static(sim, top)
    parts["top"] = top
    all_p.append(top)

    sp = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [SPINE_W, BOOK_D, BOOK_H], 0)
    sim.setObjectAlias(sp, f"{name}_sp")
    sim.setObjectPosition(sp, -1, [x - BOOK_W * 0.5 + SPINE_W * 0.5, y, base_z + BOOK_H * 0.5])
    sim.setShapeColor(
        sp,
        None,
        sim.colorcomponent_ambient_diffuse,
        [max(0, c - 0.1) for c in cover_color],
    )
    make_static(sim, sp)
    parts["spine"] = sp
    all_p.append(sp)

    ts = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid, [BOOK_W * 0.6, BOOK_D * 0.15, 0.001], 0
    )
    sim.setObjectAlias(ts, f"{name}_title")
    sim.setObjectPosition(ts, -1, [x, y, top_z + COVER_H * 0.5 + 0.001])
    sim.setShapeColor(ts, None, sim.colorcomponent_ambient_diffuse, [0.85, 0.75, 0.3])
    make_static(sim, ts)
    parts["title"] = ts
    all_p.append(ts)

    parts["_all"] = all_p
    parts["_pos"] = [x, y]
    parts["_color"] = cover_color
    return parts


# ── PLACEMENT ─────────────────────────────────────────────────────────────────


def _pick_color(i, numerator, group_a, group_b, col_a, col_b, col_n, action=""):
    """All objects placed as neutral. Action in Step 3 does the coloring."""
    return col_n


def _place_objects(
    sim,
    obj,
    action,
    numerator,
    denominator,
    col_a,
    col_b,
    col_n,
    group_a,
    all_h,
    group_b=0,
):
    handles = []

    if obj == "sphere":
        for i in range(denominator):
            x = (i - (denominator - 1) / 2.0) * 0.09
            if action == "grouping" and group_a > 0:
                rgb = col_a if i < group_a else col_b
            elif action == "grouping":
                g = math.gcd(numerator, denominator) if numerator > 0 else 1
                grp = i // g if g > 0 else 0
                rgb = RAINBOW[grp % len(RAINBOW)]
            else:
                rgb = _pick_color(i, numerator, group_a, group_b, col_a, col_b, col_n, action)
            h = _sphere(sim, f"s{i}", [x, 0, TABLE_Z + SPHERE_R], rgb)
            handles.append(h)
            all_h.append(h)

    elif obj == "tile":
        step = TILE + GAP
        if denominator <= 10:
            for i in range(denominator):
                x = (i - (denominator - 1) / 2.0) * step * 1.1
                rgb = _pick_color(i, numerator, group_a, group_b, col_a, col_b, col_n, action)
                h = _tile(sim, f"t{i}", [x, 0, TABLE_Z], rgb)
                handles.append(h)
                all_h.append(h)
        else:
            cols = (denominator + 1) // 2
            idx = 0
            for row in range(2):
                count = cols if row == 0 else denominator - cols
                for col in range(count):
                    x = (col - (count - 1) / 2.0) * step * 1.1
                    y = (row - 0.5) * step * 1.3
                    rgb = _pick_color(idx, numerator, group_a, group_b, col_a, col_b, col_n, action)
                    h = _tile(sim, f"t{idx}", [x, y, TABLE_Z], rgb)
                    handles.append(h)
                    all_h.append(h)
                    idx += 1

    elif obj == "cylinder":
        # From test3_cylinders.py placement pattern
        gap = 0.1
        if denominator <= 12:
            for i in range(denominator):
                x = (i - (denominator - 1) / 2.0) * gap
                rgb = _pick_color(i, numerator, group_a, group_b, col_a, col_b, col_n, action)
                h = _cyl(sim, f"c{i}", [x, 0, TABLE_Z + CYL_H / 2], rgb)
                handles.append(h)
                all_h.append(h)
        else:
            row0 = (denominator + 1) // 2
            row1 = denominator - row0
            idx = 0
            for row, count in enumerate([row0, row1]):
                for i in range(count):
                    x = (i - count / 2 + 0.5) * gap * 0.9
                    y = (row - 0.5) * 0.08
                    rgb = _pick_color(idx, numerator, group_a, group_b, col_a, col_b, col_n, action)
                    h = _cyl(sim, f"c{idx}", [x, y, TABLE_Z + CYL_H / 2], rgb)
                    handles.append(h)
                    all_h.append(h)
                    idx += 1

    elif obj == "book":
        gap = 0.11
        for i in range(denominator):
            x = (i - (denominator - 1) / 2.0) * gap * 0.85
            rgb = _pick_color(i, numerator, group_a, group_b, col_a, col_b, col_n, action)
            b = _build_book(sim, f"b{i}", [x, 0], rgb)
            handles.append(b)
            all_h += b["_all"]

    return handles


# ── ACTIONS (from objects.zip test files) ─────────────────────────────────────


def _run_action(
    sim,
    obj,
    action,
    numerator,
    denominator,
    handles,
    col_a,
    col_b,
    col_n,
    all_h,
    group_a=0,
):

    # ── SPHERE ACTIONS (from test1_spheres.py) ────────────────────────────

    if action == "bounce":
        # From test1_spheres.py ACTION A: BOUNCE
        bounce_height = 0.08
        for i in range(numerator):
            sim.setShapeColor(handles[i], None, sim.colorcomponent_ambient_diffuse, col_a)
            pos = sim.getObjectPosition(handles[i], -1)
            for step in range(25):
                t = step / 25
                if t < 0.4:
                    h = bounce_height * (t / 0.4)
                elif t < 0.6:
                    h = bounce_height * (1.0 + 0.2 * math.sin((t - 0.4) / 0.2 * math.pi))
                else:
                    h = bounce_height * (
                        1.0 + 0.1 * math.sin((t - 0.6) / 0.4 * math.pi * 2) * (1 - t)
                    )
                sim.setObjectPosition(handles[i], -1, [pos[0], pos[1], pos[2] + h])
                time.sleep(0.015)
            sim.setObjectPosition(handles[i], -1, [pos[0], pos[1], pos[2] + bounce_height])
            time.sleep(0.15)

    elif action == "grouping":
        # For ratio questions (group_a > 0): slide into 2 semantic groups (A vs B)
        # For fraction questions: slide into GCD-based equal groups
        if group_a > 0:
            # Ratio grouping: 2 groups matching group_a and group_b
            group_colors = [col_a, col_b]
            box_colors = [[min(1, c + 0.3) for c in gc] for gc in group_colors]
            groups = [(0, group_a), (group_a, denominator)]

            for gi, (start, end) in enumerate(groups):
                count = end - start
                gx = (gi - 0.5) * 0.4
                for j, idx in enumerate(range(start, end)):
                    if idx < len(handles):
                        jx = gx + (j - (count - 1) / 2.0) * 0.085
                        pos = sim.getObjectPosition(handles[idx], -1)
                        sim.setShapeColor(
                            handles[idx],
                            None,
                            sim.colorcomponent_ambient_diffuse,
                            group_colors[gi],
                        )
                        for s in range(15):
                            t = (s + 1) / 15
                            new_x = pos[0] + (jx - pos[0]) * t
                            sim.setObjectPosition(handles[idx], -1, [new_x, pos[1], pos[2]])
                            time.sleep(0.01)

            # Add group boxes
            for gi, (start, end) in enumerate(groups):
                count = end - start
                gx = (gi - 0.5) * 0.4
                box_w = max(0.1, count * 0.085 + 0.04)
                box = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [box_w, 0.09, 0.004], 0)
                sim.setObjectPosition(box, -1, [gx, 0, TABLE_Z])
                sim.setShapeColor(box, None, sim.colorcomponent_ambient_diffuse, box_colors[gi])
                make_static(sim, box)
                all_h.append(box)
                time.sleep(0.2)
        else:
            # Fraction grouping: GCD-based equal groups
            # Color numerator/g groups with col_a, rest with col_n
            # e.g., 12/20: gcd=4, 5 groups of 4. First 3 groups = col_a (12 items), last 2 = col_n
            g = math.gcd(numerator, denominator) if numerator > 0 else 1
            num_groups = denominator // g if g > 0 else denominator
            active_groups = numerator // g if g > 0 else 0

            for grp in range(num_groups):
                is_active = grp < active_groups
                grp_color = col_a if is_active else col_n
                gx = (grp - (num_groups - 1) / 2.0) * 0.4
                start = grp * g
                for j in range(g):
                    idx = start + j
                    if idx < len(handles):
                        jx = gx + (j - (g - 1) / 2.0) * 0.085
                        pos = sim.getObjectPosition(handles[idx], -1)
                        sim.setShapeColor(
                            handles[idx],
                            None,
                            sim.colorcomponent_ambient_diffuse,
                            grp_color,
                        )
                        for s in range(15):
                            t = (s + 1) / 15
                            new_x = pos[0] + (jx - pos[0]) * t
                            sim.setObjectPosition(handles[idx], -1, [new_x, pos[1], pos[2]])
                            time.sleep(0.01)

            # Add group boxes — active groups get col_a box, others get col_n
            for grp in range(num_groups):
                is_active = grp < active_groups
                box_color = [min(1, c + 0.3) for c in (col_a if is_active else col_n)]
                gx = (grp - (num_groups - 1) / 2.0) * 0.4
                box_w = max(0.1, g * 0.085 + 0.04)
                box = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [box_w, 0.09, 0.004], 0)
                sim.setObjectPosition(box, -1, [gx, 0, TABLE_Z])
                sim.setShapeColor(box, None, sim.colorcomponent_ambient_diffuse, box_color)
                make_static(sim, box)
                all_h.append(box)
                time.sleep(0.2)

    elif action == "split":
        # From test1_spheres.py ACTION C: SPLIT
        for i in range(numerator, denominator):
            sim.setShapeColor(handles[i], None, sim.colorcomponent_ambient_diffuse, col_n)
        for idx in range(numerator):
            pos = sim.getObjectPosition(handles[idx], -1)
            sim.removeObjects([handles[idx]])
            h1 = _half_sphere(
                sim,
                f"h{idx}_t",
                [pos[0], pos[1] - 0.02, pos[2] + SPHERE_R * 0.3],
                col_a,
            )
            h2 = _half_sphere(
                sim,
                f"h{idx}_b",
                [pos[0], pos[1] + 0.02, pos[2] - SPHERE_R * 0.3],
                col_a,
            )
            all_h += [h1, h2]
            for step in range(10):
                t = step / 10
                offset = 0.02 * t
                sim.setObjectPosition(
                    h1, -1, [pos[0], pos[1] - 0.02 - offset, pos[2] + SPHERE_R * 0.3]
                )
                sim.setObjectPosition(
                    h2, -1, [pos[0], pos[1] + 0.02 + offset, pos[2] - SPHERE_R * 0.3]
                )
                time.sleep(0.03)
            time.sleep(0.2)

    elif action == "push_apart":
        # From test1_spheres.py ACTION D: PUSH APART
        for i in range(numerator, denominator):
            sim.setShapeColor(handles[i], None, sim.colorcomponent_ambient_diffuse, col_b)
        for i in range(numerator):
            sim.setShapeColor(handles[i], None, sim.colorcomponent_ambient_diffuse, col_a)
            pos = sim.getObjectPosition(handles[i], -1)
            for step in range(25):
                t = step / 25
                sim.setObjectPosition(handles[i], -1, [pos[0], pos[1] - 0.15 * t, pos[2]])
                time.sleep(0.02)
            time.sleep(0.15)

    # ── TILE ACTIONS (from test2_tiles.py) ────────────────────────────────

    elif action == "fill":
        # From test2_tiles.py ACTION A: FILL
        for i in range(numerator):
            sim.setShapeColor(handles[i], None, sim.colorcomponent_ambient_diffuse, col_a)
            time.sleep(0.2)

    elif action == "stamp":
        # From test2_tiles.py ACTION B: STAMP
        for idx in range(numerator):
            sim.setShapeColor(handles[idx], None, sim.colorcomponent_ambient_diffuse, col_a)
            tp = sim.getObjectPosition(handles[idx], -1)
            for angle in [0.785, -0.785]:
                bar = sim.createPrimitiveShape(
                    sim.primitiveshape_cuboid, [TILE * 0.5, 0.01, 0.01], 0
                )
                sim.setObjectPosition(bar, -1, [tp[0], tp[1], TABLE_Z + TILE_H / 2 + 0.007])
                sim.setObjectOrientation(bar, -1, [0, 0, angle])
                sim.setShapeColor(bar, None, sim.colorcomponent_ambient_diffuse, col_a)
                make_static(sim, bar)
                all_h.append(bar)
            time.sleep(0.15)

    elif action == "rotate":
        # From test2_tiles.py ACTION C: ROTATE
        for idx in range(numerator):
            sim.setShapeColor(handles[idx], None, sim.colorcomponent_ambient_diffuse, col_a)
            pos = sim.getObjectPosition(handles[idx], -1)
            for s in range(15):
                t_frac = (s + 1) / 15
                angle = (math.pi / 4) * t_frac
                lift = 0.012 * math.sin(t_frac * math.pi)
                sim.setObjectOrientation(handles[idx], -1, [0, 0, angle])
                sim.setObjectPosition(handles[idx], -1, [pos[0], pos[1], pos[2] + lift])
                time.sleep(0.02)
            time.sleep(0.1)

    elif action == "grow":
        # From test2_tiles.py ACTION D: GROW
        for idx in range(numerator):
            pos = sim.getObjectPosition(handles[idx], -1)
            orig_tile = handles[idx]
            for s in range(10):
                t_frac = (s + 1) / 10
                new_size = TILE * (1.0 + 0.5 * t_frac)
                new_h = TILE_H * (1.0 + 0.3 * t_frac)
                sim.removeObjects([orig_tile])
                orig_tile = sim.createPrimitiveShape(
                    sim.primitiveshape_cuboid, [new_size, new_size, new_h], 0
                )
                sim.setObjectAlias(orig_tile, f"grow_{idx}_{s}")
                sim.setObjectPosition(orig_tile, -1, [pos[0], pos[1], TABLE_Z])
                sim.setShapeColor(orig_tile, None, sim.colorcomponent_ambient_diffuse, col_a)
                make_static(sim, orig_tile)
                time.sleep(0.03)
            handles[idx] = orig_tile
            all_h.append(orig_tile)
            time.sleep(0.1)

    # ── CYLINDER ACTIONS (from test3_cylinders.py) ────────────────────────

    elif action == "lid_drop":
        # From test3_cylinders.py ACTION A: LID DROP
        for i in range(numerator):
            sim.setShapeColor(handles[i], None, sim.colorcomponent_ambient_diffuse, col_a)
            cp = sim.getObjectPosition(handles[i], -1)
            target_z = TABLE_Z + CYL_H + 0.008
            lid = _cyl(
                sim,
                f"lid{i}",
                [cp[0], cp[1], target_z + 0.1],
                col_a,
                r=CYL_R + 0.005,
                h_=0.015,
            )
            all_h.append(lid)
            for s in range(15):
                t = (s + 1) / 15
                z = (target_z + 0.1) - 0.1 * (t * t)
                sim.setObjectPosition(lid, -1, [cp[0], cp[1], z])
                time.sleep(0.02)
            time.sleep(0.15)

    elif action == "spin":
        # From test3_cylinders.py ACTION B: SPIN
        for idx in range(numerator):
            sim.setShapeColor(handles[idx], None, sim.colorcomponent_ambient_diffuse, col_a)
            for s in range(30):
                angle = (s / 30) * math.pi * 4
                sim.setObjectOrientation(handles[idx], -1, [0, 0, angle])
                time.sleep(0.02)
            time.sleep(0.15)

    elif action == "knock_over":
        # From test3_cylinders.py ACTION C: KNOCK OVER
        for idx in range(numerator):
            pos = sim.getObjectPosition(handles[idx], -1)
            sim.setShapeColor(handles[idx], None, sim.colorcomponent_ambient_diffuse, col_a)
            for s in range(20):
                t = s / 20
                angle = (math.pi / 2) * t
                new_z = TABLE_Z + CYL_H / 2 * math.cos(angle) + CYL_R * math.sin(angle)
                new_y = pos[1] + CYL_H / 2 * math.sin(angle) - CYL_R * math.cos(angle) + CYL_R
                sim.setObjectOrientation(handles[idx], -1, [angle, 0, 0])
                sim.setObjectPosition(handles[idx], -1, [pos[0], new_y, new_z])
                time.sleep(0.02)
            time.sleep(0.15)

    elif action == "shrink":
        # From test3_cylinders.py ACTION D: SHRINK
        for idx in range(numerator):
            target = handles[idx]
            pos = sim.getObjectPosition(target, -1)
            for s in range(12):
                t = (s + 1) / 12
                scale = 1.0 - 0.5 * t
                new_r = CYL_R * scale
                new_h = CYL_H * scale
                new_z = TABLE_Z + new_h / 2
                sim.removeObjects([target])
                target = _cyl(
                    sim,
                    f"shrink_{idx}_{s}",
                    [pos[0], pos[1], new_z],
                    col_a,
                    r=new_r,
                    h_=new_h,
                )
                all_h.append(target)
                time.sleep(0.04)
            time.sleep(0.15)

    # ── BOOK ACTIONS (from test4_books.py) ────────────────────────────────

    elif action == "open_book":
        # From test4_books.py open_book()
        for idx in range(numerator):
            book = handles[idx]
            for pn in ["top", "bot", "spine"]:
                sim.setShapeColor(book[pn], None, sim.colorcomponent_ambient_diffuse, col_a)
            top = book["top"]
            title = book["title"]
            x, y = book["_pos"]
            top_z = sim.getObjectPosition(top, -1)[2]
            spine_x = x - BOOK_W * 0.5
            for step in range(25):
                t = (step + 1) / 25
                angle = -(math.pi * 0.35) * t
                dx = BOOK_W * 0.5
                new_x = spine_x + dx * math.cos(angle)
                new_z = top_z + dx * math.sin(-angle)
                sim.setObjectPosition(top, -1, [new_x, y, new_z])
                sim.setObjectOrientation(top, -1, [0, angle, 0])
                sim.setObjectPosition(title, -1, [new_x, y, new_z + 0.002])
                sim.setObjectOrientation(title, -1, [0, angle, 0])
                time.sleep(0.03)
            time.sleep(0.3)

    elif action == "stack_books":
        # Color active books, then create stacks
        for idx in range(numerator):
            for pn in ["top", "bot", "spine"]:
                sim.setShapeColor(handles[idx][pn], None, sim.colorcomponent_ambient_diffuse, col_a)
            handles[idx]["_color"] = col_a
            bpos = handles[idx]["_pos"]
            bcolor = col_a
            for layer in range(3):
                z_off = BOOK_H * (layer + 1) + 0.002 * (layer + 1)
                stk = _build_book(
                    sim,
                    f"stk{idx}_L{layer}",
                    [bpos[0], bpos[1]],
                    bcolor,
                    z_offset=z_off,
                )
                all_h += stk["_all"]
        time.sleep(0.3)

    elif action == "change_cover":
        # From test4_books.py ACTION C: CHANGE COVER COLOR
        for idx in range(numerator):
            for pn in ["top", "bot", "spine"]:
                sim.setShapeColor(handles[idx][pn], None, sim.colorcomponent_ambient_diffuse, col_a)
            time.sleep(0.3)

    elif action == "stand_up":
        # Books rotate upright — fast animation
        for idx in range(numerator):
            book = handles[idx]
            x, y = book["_pos"]
            spine_y = y - BOOK_D * 0.5
            for pn in ["top", "bot", "spine"]:
                sim.setShapeColor(book[pn], None, sim.colorcomponent_ambient_diffuse, col_a)
            for s in range(15):
                t = (s + 1) / 15
                te = t * t * (3 - 2 * t)
                angle = (math.pi / 2) * te
                for p in book["_all"]:
                    try:
                        orig = sim.getObjectPosition(p, -1)
                        dy = orig[1] - spine_y
                        dz = orig[2] - TABLE_Z
                        r = math.sqrt(dy * dy + dz * dz) or 0.01
                        na = math.atan2(dz, dy) + angle
                        sim.setObjectPosition(
                            p,
                            -1,
                            [x, spine_y + r * math.cos(na), TABLE_Z + r * math.sin(na)],
                        )
                        sim.setObjectOrientation(p, -1, [angle, 0, 0])
                    except Exception:
                        LOGGER.debug("Unable to animate one book part", exc_info=True)
                time.sleep(0.02)
            time.sleep(0.08)


# ── PUBLIC API ────────────────────────────────────────────────────────────────


def run(
    obj,
    action,
    numerator,
    denominator,
    set_step,
    col_a,
    col_b,
    col_n,
    group_a=0,
    group_b=0,
    lhs_num=None,
    lhs_den=None,
    rhs_num=None,
    rhs_den=None,
    cancel_check=None,
    qtype=None,
    wait_for_advance=None,
):
    if RemoteAPIClient is None:
        LOGGER.warning("CoppeliaSim client is not installed; skipping visualization")
        return

    def cancelled():
        return cancel_check and cancel_check()

    def wait_step(n):
        if wait_for_advance:
            return wait_for_advance(n)
        return True

    sim = _connect()
    all_h = []
    label_h = []

    # Step 1: wait for student, then narration, CoppeliaSim empty
    if not wait_step(1) or cancelled():
        return
    set_step(1)

    # Step 2: wait for student, then narration, clear old objects, place new ones
    if not wait_step(2) or cancelled():
        return
    set_step(2)
    time.sleep(1)
    _clear_scene(sim)  # clear previous question's objects right before placing new ones
    handles = _place_objects(
        sim,
        obj,
        action,
        numerator,
        denominator,
        col_a,
        col_b,
        col_n,
        group_a,
        all_h,
        group_b=group_b,
    )

    # Step 3: wait for student, then narration, then action
    if not wait_step(3) or cancelled():
        return
    set_step(3)
    time.sleep(1)

    _run_action(
        sim,
        obj,
        action,
        numerator,
        denominator,
        handles,
        col_a,
        col_b,
        col_n,
        all_h,
        group_a=group_a,
    )

    # Show label after action ONLY if it makes sense (lhs != rhs)
    is_ratio = qtype in ("ratio_simplify", "ratio_equiv", "scale_up", "part_whole")
    if (
        lhs_num
        and lhs_den
        and rhs_num
        and rhs_den
        and not (lhs_num == rhs_num and lhs_den == rhs_den)
    ):
        try:
            if is_ratio:
                label_h = render_ratio_equivalence(
                    sim,
                    lhs_num,
                    lhs_den,
                    rhs_num,
                    rhs_den,
                    [0, 0, TABLE_Z + 0.15],
                    scale=0.02,
                    color=LABEL_C,
                )
            else:
                label_h = render_equivalence(
                    sim,
                    lhs_num,
                    lhs_den,
                    rhs_num,
                    rhs_den,
                    [0, 0, TABLE_Z + 0.15],
                    scale=0.02,
                    color=LABEL_C,
                )
            all_h += label_h
        except Exception:
            LOGGER.debug("Unable to render the final equation label", exc_info=True)
