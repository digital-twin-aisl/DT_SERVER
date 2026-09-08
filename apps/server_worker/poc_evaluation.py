"""Compare held LOD decisions against future ground-truth trajectories.

The oracle is the same spatial urgency kernel as the online engine, evaluated
on actual future positions/velocities instead of constant-velocity predictions.
Inferred root CSVs can be used only with an explicit 'estimated' label and must
not be reported as a ground-truth oracle.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from apps.server_worker.priority_engine import PriorityConfig, PriorityEngine


class Trajectories:
    def __init__(self, records, max_gap=0.2):
        if not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("trajectory gap must be finite and positive")
        self.max_gap = max_gap
        grouped = {}
        for row in records:
            key = int(row["global_id"])
            grouped.setdefault(key, []).append(
                (
                    float(row["timestamp_s"]),
                    float(row["x_mm"]) / 1000.0,
                    float(row["y_mm"]) / 1000.0,
                )
            )
        self.tracks = {}
        for key, rows in grouped.items():
            array = np.asarray(sorted(rows), dtype=np.float64)
            if not np.isfinite(array).all() or (np.diff(array[:, 0]) <= 0).any():
                raise ValueError(
                    f"trajectory {key} requires finite, unique increasing timestamps"
                )
            if len(array) < 2:
                raise ValueError(f"trajectory {key} requires at least two samples")
            self.tracks[key] = array
        if not self.tracks:
            raise ValueError("trajectory file is empty")

    def sample(self, key, timestamp):
        array = self.tracks[key]
        if timestamp < array[0, 0] - 1e-9 or timestamp > array[-1, 0] + 1e-9:
            return None
        right = int(np.searchsorted(array[:, 0], timestamp, side="right"))
        right = min(max(1, right), len(array) - 1)
        before, after = array[right - 1], array[right]
        dt = after[0] - before[0]
        if dt > self.max_gap + 1e-9:
            return None
        velocity = (after[1:] - before[1:]) / dt
        position = before[1:] + velocity * (timestamp - before[0])
        return position, velocity

    def scores(self, timestamp, engine, unit_dt):
        scores = {}
        incomplete = []
        for key in self.tracks:
            if self.sample(key, timestamp) is None:
                if self.tracks[key][0, 0] <= timestamp <= self.tracks[key][-1, 0]:
                    incomplete.append(key)
                continue
            total = 0.0
            for step in range(engine.config.prediction_steps + 1):
                sample = self.sample(key, timestamp + step * unit_dt)
                if sample is None:
                    incomplete.append(key)
                    break
                position, velocity = sample
                total += (
                    engine.instant_urgency(position, velocity)
                    * engine.config.delta**step
                    * unit_dt
                )
            else:
                scores[key] = total
        return scores, incomplete


def runtime_summary(values):
    if not values:
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("invalid runtime measurement")
    return {
        "samples": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def compare_decisions(
    rank_records,
    zone_records,
    trajectories,
    engine,
    *,
    interval,
    identity_map=None,
    warmup_batches=10,
):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("comparison interval must be finite and positive")
    if not rank_records or len(rank_records) != len(zone_records):
        raise ValueError(
            "Rank/Zone runs must contain the same nonzero number of batches"
        )
    identity_map = identity_map or {}
    rows = []
    previous = -float("inf")
    for rank, zone in zip(rank_records, zone_records):
        timestamp = float(rank["timestamp"])
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(float(zone["timestamp"]))
            or timestamp <= previous
            or abs(timestamp - float(zone["timestamp"])) > 1e-6
        ):
            raise ValueError(
                "Rank/Zone runs must have identical increasing input timestamps"
            )
        previous = timestamp
        if rank["policy"] != "rank" or zone["policy"] != "zone":
            raise ValueError("expected rank and zone decision files, in that order")
        if rank.get("observations") != zone.get("observations"):
            raise ValueError(
                "Rank/Zone must replay identical observations and identities"
            )
        if not rank.get("input_sha256") or rank["input_sha256"] != zone.get(
            "input_sha256"
        ):
            raise ValueError("Rank/Zone must replay identical recorded packets")
        for record in (rank, zone):
            config = record["configuration"]
            if (
                config["lambda_aoi"] != 0
                or abs(config["priority_interval"] - interval) > 1e-9
            ):
                raise ValueError(
                    "PoC evaluation requires AoI=0 and the same decision interval"
                )
            if not np.array_equal(
                np.asarray(config["hazards_metres"]), np.asarray(engine.config.hazards)
            ):
                raise ValueError("oracle and online hazard points must match")
        scores, incomplete = trajectories.scores(timestamp, engine, interval)
        if incomplete or not scores:
            rows.append(
                {
                    "timestamp_s": timestamp,
                    "evaluated": False,
                    "reason": "incomplete_future_trajectory",
                    "incomplete_ids": incomplete,
                }
            )
            continue
        total = sum(scores.values())
        row = {
            "timestamp_s": timestamp,
            "evaluated": True,
            "oracle_total": total,
            "oracle_people": len(scores),
        }
        for policy, record in (("rank", rank), ("zone", zone)):
            selected = {
                identity_map.get(int(key), int(key)) for key in record["selected_ids"]
            }
            missing = selected - set(scores)
            if missing:
                raise ValueError(
                    f"selected IDs are not present in reference trajectories: {sorted(missing)}; provide an identity map"
                )
            covered = sum(scores[key] for key in selected)
            row[f"{policy}_missed_hazard"] = max(0.0, total - covered)
            row[f"{policy}_selected_count"] = len(selected)
            row[f"{policy}_coverage"] = covered / total if total > 0 else None
        rows.append(row)
    evaluated = [row for row in rows if row["evaluated"]]
    summary = {
        "total_batches": len(rows),
        "evaluated_batches": len(evaluated),
        "excluded_batches": len(rows) - len(evaluated),
        "metric": "sum(all oracle hazard) - sum(selected oracle hazard)",
        "future_horizon_s": engine.config.prediction_steps * interval,
        "runtime_warmup_batches": warmup_batches,
    }
    for policy, records in (("rank", rank_records), ("zone", zone_records)):
        summary[policy] = {
            "mean_missed_hazard": (
                float(np.mean([row[f"{policy}_missed_hazard"] for row in evaluated]))
                if evaluated
                else None
            ),
            "mean_selected_count": (
                float(np.mean([row[f"{policy}_selected_count"] for row in evaluated]))
                if evaluated
                else None
            ),
            "pose_runtime": runtime_summary(
                [record["pose_runtime_ms"] for record in records[warmup_batches:]]
            ),
            "pipeline_runtime": runtime_summary(
                [
                    record["pipeline_runtime_ms"]
                    for record in records[warmup_batches:]
                    if "pipeline_runtime_ms" in record
                ]
            ),
            "pose_gpu": runtime_summary(
                [
                    record["pose_gpu_ms"]
                    for record in records[warmup_batches:]
                    if record.get("pose_gpu_ms") is not None
                ]
            ),
        }
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-decisions", required=True)
    parser.add_argument("--zone-decisions", required=True)
    parser.add_argument(
        "--trajectories", required=True, help="CSV: timestamp_s,global_id,x_mm,y_mm"
    )
    parser.add_argument(
        "--trajectory-kind", required=True, choices=("ground-truth", "estimated")
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument(
        "--ids-aligned",
        action="store_true",
        help="Explicitly confirm reference IDs equal replay IDs",
    )
    identity.add_argument(
        "--identity-map", help="JSON mapping replay IDs to reference IDs"
    )
    parser.add_argument("--max-trajectory-gap", type=float, default=0.2)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.max_trajectory_gap <= 0 or args.warmup_batches < 0:
        parser.error("invalid trajectory gap or warmup count")

    def read_jsonl(path):
        return [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]

    rank, zone = read_jsonl(args.rank_decisions), read_jsonl(args.zone_decisions)
    if not rank:
        raise ValueError("rank decision file is empty")
    config = rank[0]["configuration"]
    interval = float(config["priority_interval"])
    if interval <= 0:
        raise ValueError(
            "oracle evaluation requires a positive fixed priority interval"
        )
    engine = PriorityEngine(
        PriorityConfig(
            hazards=tuple(tuple(point) for point in config["hazards_metres"]),
            lambda_aoi=0,
            initial_unit_time_s=interval,
        )
    )
    with Path(args.trajectories).open(newline="") as stream:
        trajectories = Trajectories(csv.DictReader(stream), args.max_trajectory_gap)
    mapping = (
        {
            int(key): int(value)
            for key, value in json.loads(Path(args.identity_map).read_text()).items()
        }
        if args.identity_map
        else {}
    )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("identity map must be one-to-one")
    summary, rows = compare_decisions(
        rank,
        zone,
        trajectories,
        engine,
        interval=interval,
        identity_map=mapping,
        warmup_batches=args.warmup_batches,
    )
    summary["trajectory_kind"] = args.trajectory_kind
    summary["oracle_kind"] = (
        "ground_truth_future"
        if args.trajectory_kind == "ground-truth"
        else "estimated_trajectory_proxy"
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (output / "per_frame.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
