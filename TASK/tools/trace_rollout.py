#!/usr/bin/env python3
"""Instrumented single-seed rollout for the SAR champion (or any submission zip).

Runs the episode in-process (no Docker), exactly like `swarm video --backend local`,
but instead of rendering video it logs, every tick, what the agent decided and what
the ground truth was. Output: one CSV per seed + a one-line summary JSON.

Usage (from the repo root, with miner_env active):
    python TASK/tools/trace_rollout.py --model TASK/champion/submission.zip \
        --seed 1000029 --type 4 --out traces/

Columns that matter for the analysis:
    route        agent route string ('king', 'king+tour', 'king+rgb', '+av')
    tour_on      the lawn-mower overlay is active (it rewrites the clue offset)
    wp_dist      distance from drone to the current tour waypoint (m)
    v_horiz      horizontal distance drone -> victim (ground truth, m)
    v_visible    victim inside the depth-camera frustum, <=30 m and line of sight
    cmd_speed    speed the agent commanded this tick (action[3], 0..1)
    dwell        continuous seconds the confirm predicate has held
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validator.scripts.generate_video import (  # noqa: E402
    _ensure_local_ansible_temp,
    _extract_zip,
    _link_workspace,
    _load_agent,
    build_task,
)

DEPTH_RANGE_M = 30.0
FAMILY = "cf_search_and_rescue"


def _victim_visible(p, cli, env, victim_c, aabb, victim_uids):
    """Exact camera geometry, mirrored from MovingDroneAviary._drone_camera_view():
    camera at pos + fwd*CAMERA_EYE_FWD_M + up*CAMERA_EYE_UP_M, looking along body forward,
    square image, vertical FOV env._fov. Visible = some victim point is inside the frustum,
    within depth range, and the first body a ray from the camera hits is the victim.
    Returns (visible, in_frustum_any, dist)."""
    from swarm.constants import CAMERA_EYE_FWD_M, CAMERA_EYE_UP_M

    pos = np.asarray(env.pos[0], float)
    rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
    fwd, up = rot @ np.array([1.0, 0, 0]), rot @ np.array([0, 0, 1.0])
    right = np.cross(fwd, up)
    cam = pos + fwd * CAMERA_EYE_FWD_M + up * CAMERA_EYE_UP_M
    tan_half = math.tan(math.radians(float(getattr(env, "_fov", 90.0))) / 2.0)

    lo, hi = np.asarray(aabb[0], float), np.asarray(aabb[1], float)
    c = np.asarray(victim_c, float)
    targets = [c, np.array([c[0], c[1], hi[2]]),
               np.array([lo[0], c[1], c[2]]), np.array([hi[0], c[1], c[2]]),
               np.array([c[0], lo[1], c[2]]), np.array([c[0], hi[1], c[2]])]
    dist = float(np.linalg.norm(c - cam))
    in_frustum_any = False
    for tgt in targets:
        v = tgt - cam
        z = float(np.dot(v, fwd))
        if z <= 0.05 or float(np.linalg.norm(v)) > DEPTH_RANGE_M:
            continue
        if abs(float(np.dot(v, right))) / z > tan_half or abs(float(np.dot(v, up))) / z > tan_half:
            continue
        in_frustum_any = True
        hit = p.rayTest(cam.tolist(), tgt.tolist(), physicsClientId=cli)[0]
        if hit[0] in victim_uids:
            return True, True, dist
    return False, in_frustum_any, dist


def trace(model: Path, seed: int, ctype: int, out_dir: Path) -> dict:
    import pybullet as p
    from gym_pybullet_drones.utils.enums import ActionType

    from swarm.constants import SIM_DT, SPEED_LIMIT
    from swarm.utils.env_factory import make_env

    work = (out_dir / f".work_{seed}").resolve()
    extracted = _extract_zip(model.resolve(), work)
    _link_workspace(extracted)
    agent = _load_agent(extracted)

    task = build_task(seed, ctype, family_id=FAMILY)
    _ensure_local_ansible_temp()
    env = make_env(task, gui=False)
    obs, _ = env.reset(seed=task.map_seed)
    cli = getattr(env, "CLIENT", 0)
    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()

    rows, t, info = [], 0.0, {}
    first_visible = None
    closest = (math.inf, None)
    tour_start = None

    while t < task.horizon:
        try:
            raw = agent.act(obs)
        except Exception:
            raw = None
        act = np.zeros(lo.shape, np.float32) if raw is None else np.asarray(raw, np.float32).flatten()
        act = np.clip(act, lo, hi)
        if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
            v = act[:3]
            act[:3] = v * min(1.0, float(SPEED_LIMIT) / max(float(np.linalg.norm(v)), 1e-6))
            act = np.clip(act, lo, hi)

        obs, _, term, trunc, info = env.step(act[None, :])
        t += float(SIM_DT)

        st = env._getDroneStateVector(0)
        pos, rpy, vel = st[0:3], st[7:10], st[10:13]
        vc = np.asarray(env.sar_world.victim_centre, float)
        vtop = float(env.sar_world.victim_aabb[1][2])
        horiz = float(np.linalg.norm(pos[0:2] - vc[0:2]))
        vaabb = env.sar_world.victim_aabb
        vis, infr, vdist = _victim_visible(p, cli, env, vc, vaabb, set(env.sar_world.victim_uids))
        depr = math.degrees(math.atan2(float(pos[2]) - float(vc[2]), max(horiz, 1e-6)))

        tour_on = bool(getattr(agent, "_on", False))
        wp_dist = float("nan")
        w = getattr(agent, "_w", None)
        if tour_on and w is not None and len(w):
            wp = w[min(int(getattr(agent, "_i", 0)), len(w) - 1)]
            wp_dist = float(np.linalg.norm(np.asarray(wp) - pos[0:2]))
        if tour_on and tour_start is None:
            tour_start = t
        if vis and first_visible is None:
            first_visible = t
        if horiz < closest[0]:
            closest = (horiz, t)

        rows.append({
            "t": round(t, 3),
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "speed": float(np.linalg.norm(vel)),
            "v_horiz": horiz,
            "v_height": float(pos[2]) - vtop,
            "v_dist": vdist,
            "v_visible": int(vis),
            "v_in_frustum": int(infr),
            "v_depr_deg": round(depr, 1),
            "route": str(getattr(agent, "route", "")),
            "tour_on": int(tour_on),
            "wp_idx": int(getattr(agent, "_i", -1)) if tour_on else -1,
            "wp_dist": wp_dist,
            "cmd_speed": float(act[3]) if act.shape[0] > 3 else float("nan"),
            "rgb_req": float(act[5]) if act.shape[0] > 5 else 0.0,
            "dwell": float(getattr(env, "_sar_dwell_time", 0.0)),
        })
        if term or trunc:
            break

    success = bool(info.get("success", False))
    reason = str(getattr(env, "_failure_reason", ""))
    visible_ticks = sum(r["v_visible"] for r in rows)
    vis_with_tour = sum(1 for r in rows if r["v_visible"] and r["tour_on"])
    max_dwell = float(info.get("sar_max_dwell", max((r["dwell"] for r in rows), default=0.0)))

    # Failure taxonomy (only meaningful for unsuccessful episodes)
    if success:
        label = "SUCCESS"
    elif reason in ("OBSTACLE_COLLISION", "TILT", "NO_TOUCH_SPHERE"):
        label = f"CRASH:{reason}"
    elif first_visible is None:
        label = "NEVER_SEEN"            # coverage problem: victim never in view
    elif max_dwell > 0.0:
        label = "HOVER_UNSTABLE"        # reached the confirm window but could not hold 2 s
    elif closest[0] <= 6.0:
        label = "CLOSE_BUT_LEFT"        # got within 6 m and moved away
    else:
        label = "SEEN_NOT_APPROACHED"   # in view at some point, never closed in

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"trace_{seed}_t{ctype}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = {
        "seed": seed, "type": ctype, "success": success, "failure_reason": reason,
        "label": label, "sim_time": round(t, 2),
        "tour_start_s": tour_start, "first_visible_s": first_visible,
        "visible_s": round(visible_ticks * float(SIM_DT), 2),
        "visible_while_tour_s": round(vis_with_tour * float(SIM_DT), 2),
        "in_frustum_s": round(sum(r["v_in_frustum"] for r in rows) * float(SIM_DT), 2),
        "fov_deg": round(float(getattr(env, "_fov", 90.0)), 1),
        "closest_horiz_m": round(closest[0], 2), "closest_at_s": closest[1],
        "max_dwell_s": round(max_dwell, 2),
        "csv": str(csv_path),
    }
    with open(out_dir / f"summary_{seed}_t{ctype}.json", "w") as f:
        json.dump(summary, f, indent=2)
    env.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--type", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("traces"))
    a = ap.parse_args()
    print(json.dumps(trace(a.model, a.seed, a.type, a.out), indent=2))


if __name__ == "__main__":
    main()
