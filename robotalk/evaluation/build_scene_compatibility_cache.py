"""Aggregate per-scene compatibility audits into the shared sampling cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from robotalk.scene_sampling import SCENE_POLICY_VERSION


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--require-scene-count", type=int)
    parser.add_argument("--allow-zero-compatible", action="store_true")
    parser.add_argument("--allow-task-errors", action="store_true")
    args=parser.parse_args()
    files=sorted(args.input_dir.glob("layout_*_style_*_seed_*.json")); tasks={}; task_errors=[]; scenes=[]
    for path in files:
        payload=json.loads(path.read_text()); scene=payload["scene"]; scenes.append(scene); task_errors.extend({"scene":scene,**row} for row in payload.get("task_errors",[]))
        for row in payload["records"]:
            task=tasks.setdefault(row["task"],{"configurations":{}})
            config=task["configurations"].setdefault(row["physical_configuration_signature"],{"physical_configuration":row["physical_configuration"],"representative_run_index":row["representative_run_index"],"compatible_scenes":[],"incompatible_scenes":[]})
            target="compatible_scenes" if row["compatible"] else "incompatible_scenes"
            entry=dict(scene)
            if not row["compatible"]: entry["error"]=row["error"]
            config[target].append(entry)
    counts=[]; zero=[]
    for task,row in tasks.items():
        for sig,config in row["configurations"].items():
            count=len(config["compatible_scenes"]); counts.append(count)
            if count==0: zero.append({"task":task,"physical_configuration_signature":sig})
    hist=Counter(counts)
    if args.require_scene_count is not None and len(files) != args.require_scene_count:
        raise ValueError(
            f"found {len(files)} scene audits, expected {args.require_scene_count}"
        )
    if zero and not args.allow_zero_compatible:
        raise ValueError(f"{len(zero)} physical configurations have no compatible scene")
    if task_errors and not args.allow_task_errors:
        raise ValueError(f"scene audit contains {len(task_errors)} task initialization errors")
    payload={"schema_version":1,"scene_policy_version":SCENE_POLICY_VERSION,"source_files":[str(p.resolve()) for p in files],"candidate_scenes":scenes,"summary":{"scene_count":len(files),"task_count":len(tasks),"physical_configuration_count":len(counts),"zero_compatible_configurations":zero,"compatible_scene_count_histogram":dict(sorted(hist.items())),"task_initialization_errors":task_errors},"tasks":tasks}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload["summary"],indent=2))


if __name__=="__main__": main()
