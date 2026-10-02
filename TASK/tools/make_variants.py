#!/usr/bin/env python3
"""Build ablation variants of the champion: each one flips ONE subsystem.

    python TASK/tools/make_variants.py                 # builds every variant + packages it
    python TASK/tools/make_variants.py --seeds 40      # also writes village_subset_seeds.json

Every variant is the champion's drone_agent.py with a single, named edit. The weights
(policy.onnx, typenet.bin, victim_rgb_m.onnx) are never touched. Output:
    variants/<name>/            unpacked source
    Submission/<name>.zip       packaged, ready for `swarm benchmark --model`
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

CHAMPION = Path("TASK/champion/submission.zip")

# name -> (what it tests, [(exact old text, new text), ...])
VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "no_overlays": (
        "Pure learned policy everywhere: classifier, tour, avoidance, escape and RGB all off. "
        "Measures what the whole hand-written layer contributes per environment.",
        [("ON=os.environ.get('KT_ON','1')=='1'", "ON=os.environ.get('KT_ON','0')=='1'")],
    ),
    "village_no_tour": (
        "Village without the lawn-mower tour (the tour rewrites the clue the policy sees).",
        [("os.environ.get('KT_TYPES','forest,village,city,mountain')",
          "os.environ.get('KT_TYPES','forest,city,mountain')")],
    ),
    "village_escape": (
        "Village gets the escape subsystem that mountain/forest already have.",
        [("return bool(c.settled('village') or c.settled('warehouse'))",
          "return bool(c.settled('warehouse'))"),
         ("(_c2.is_mountain or _c2.settled('forest'))",
          "(_c2.is_mountain or _c2.settled('forest') or _c2.settled('village'))")],
    ),
    "village_avoid": (
        "Village gets the obstacle-avoidance subsystem that forest/city already have.",
        [("AV_TYPES=('forest','city')", "AV_TYPES=('forest','city','village')")],
    ),
}


# --- "fly lower" variants -------------------------------------------------------------
# One-sided altitude ceiling: in village, while the tour is steering, if the drone is more
# than TARGET metres above whatever is below it (state altitude_norm * 20), add a downward
# velocity component. It never pushes the drone up and never touches the final descent.
LOW_CONSTANTS = """
# PATCH low-cruise: one-sided altitude ceiling while the tour is active
LOW_TYPES=('village',)
LOW_TARGET_M={target}   # ceiling above whatever is below the drone (m)
LOW_GAIN=0.5            # m/s of descent per metre above the ceiling
LOW_VZ_MAX=1.0          # max added descent speed (m/s)
LOW_FRONT_M=4.0         # skip the ceiling if anything in the central view is closer (m)
"""

LOW_FUNC = """
def _low_cap(out,agl_m,target,gain,vz_max,climb_ok=0.1):
 \"\"\"Return a copy of `out` whose vertical velocity is at most -gain*(agl-target).
 Velocity = 3 m/s * speed * unit(dir). Only ever lowers vz; no-op below the ceiling,
 and no-op when the policy itself commands a climb (it is usually clearing an obstacle).\"\"\"
 if agl_m is None or agl_m>=19.9 or agl_m<=target:
  return out
 o=np.array(out,np.float32,copy=True)
 d=o[0:3].astype(np.float64); n=float(np.linalg.norm(d))
 v=(d/n)*3.0*float(o[3]) if n>1e-6 else np.zeros(3)
 if v[2]>climb_ok:
  return out
 vz_des=-min(vz_max,gain*(agl_m-target))
 if v[2]<=vz_des:
  return out
 v[2]=vz_des
 sp=float(np.linalg.norm(v))
 if sp<1e-6:
  return out
 o[0:3]=(v/sp).astype(np.float32); o[3]=np.float32(min(1.0,sp/3.0))
 return o
"""

LOW_CALL = """  if self._on and self._stype() in LOW_TYPES and out.shape[0]>=4:
   _dm=np.asarray(observation['depth'],np.float32).reshape(256,256)*29.5+0.5
   if float(_dm[96:160,96:160].min())>=LOW_FRONT_M:
    _raw=np.asarray(observation['state'],np.float32).reshape(-1)
    out=_low_cap(out,float(_raw[-3])*20.0,LOW_TARGET_M,LOW_GAIN,LOW_VZ_MAX)
  return out
"""


def low_variant(target: float, avoid: bool = False) -> tuple[str, list[tuple[str, str]]]:
    av_new = "AV_TYPES=('forest','city','village')\n" if avoid else "AV_TYPES=('forest','city')\n"
    anchor_ret = "   if self._rp.want_rgb(self.tick):\n    out[5]=np.float32(1.0)\n  return out\n"
    return (
        f"Village tour flies with a {target:g} m ceiling above whatever is below the drone"
        + (" + obstacle avoidance in village." if avoid else "."),
        [
            ("AV_TYPES=('forest','city')\n",
             av_new + LOW_CONSTANTS.format(target=target)),
            ("class _RgbPrimary:\n", LOW_FUNC.lstrip("\n") + "\nclass _RgbPrimary:\n"),
            (anchor_ret, anchor_ret.replace("  return out\n", "") + LOW_CALL),
        ],
    )


def build(name: str, edits: list[tuple[str, str]], package: bool) -> None:
    src = Path("variants") / name
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    with zipfile.ZipFile(CHAMPION) as z:
        z.extractall(src)
    # `swarm model package` writes its own contract; a second copy triggers the duplicate warning
    (src / "swarm_policy_contract.json").unlink(missing_ok=True)

    agent = src / "drone_agent.py"
    code = agent.read_text()
    for old, new in edits:
        if code.count(old) != 1:
            raise SystemExit(f"[{name}] expected exactly one match for: {old!r}")
        code = code.replace(old, new)
    agent.write_text(code)
    compile(code, str(agent), "exec")  # fail early on a syntax error

    if package:
        out = Path("Submission") / f"{name}.zip"
        subprocess.run(
            ["swarm", "model", "package", "--source", str(src), "--output", str(out),
             "--family-id", "cf_search_and_rescue", "--overwrite"],
            check=True,
        )
    print(f"[ok] {name}")


def write_subset(n: int, group: str = "type4_village") -> None:
    d = json.load(open("TASK/practice_seeds.json"))
    # other groups keep one filler seed: the loader rejects empty groups
    d["type_seeds"] = {k: (v[:n] if k == group else v[:1]) for k, v in d["type_seeds"].items()}
    out = Path(f"{group.split('_', 1)[1]}_subset_seeds.json")
    json.dump(d, open(out, "w"), indent=2, sort_keys=True)
    print(f"[ok] {out}: {n} seeds of {group} + 1 filler per other group")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--no-package", action="store_true")
    ap.add_argument("--seeds", type=int, default=0, help="write a village subset of this size")
    ap.add_argument("--low", nargs="*", type=float, default=[],
                    help="also build village_low2_<m> variants, e.g. --low 5 6")
    ap.add_argument("--low-avoid", nargs="*", type=float, default=[],
                    help="also build village_low2_<m>_avoid variants (ceiling + avoidance)")
    a = ap.parse_args()
    for name, (why, edits) in VARIANTS.items():
        if a.only and name not in a.only:
            continue
        print(f"\n{name}: {why}")
        build(name, edits, package=not a.no_package)
    for m in a.low:
        name = f"village_low2_{m:g}"
        if a.only and name not in a.only:
            continue
        why, edits = low_variant(m)
        print(f"\n{name}: {why}")
        build(name, edits, package=not a.no_package)
    for m in a.low_avoid:
        name = f"village_low2_{m:g}_avoid"
        if a.only and name not in a.only:
            continue
        why, edits = low_variant(m, avoid=True)
        print(f"\n{name}: {why}")
        build(name, edits, package=not a.no_package)
    if a.seeds:
        write_subset(a.seeds)


if __name__ == "__main__":
    main()
