# Server inference telemetry

`inference.py` creates one telemetry directory per process under
`apps/server_worker/data/metrics` by default. The writer is asynchronous and
bounded; if storage cannot keep up, inference continues and the number of lost
telemetry records is reported.

```bash
python apps/server_worker/inference.py \
  --metrics-run-label poc-hall-load \
  --metrics-tag people=20 \
  --metrics-tag scenario=occlusion
```

Use `--metrics-dir PATH` to move the output, `--metrics-resource-interval 0.5`
to change the resource sample period, or `--no-metrics` to disable collection.

Each run directory contains:

- `run.json`: command/configuration, Edge and camera topology, model paths,
  Git revision, host, CUDA, PyTorch, and GPU information.
- `events.jsonl`: one `batch` record per synchronized batch plus startup,
  input-wait, and shutdown events.
- `summary.json`: run duration, success/failure counts, average/min/max batch
  time and people count, drops, and the number of observed global IDs.

Batch records contain:

- stage times for receive/prepare, clustering, root association, LOD policy,
  pose inference, scene construction, serialization, and transport enqueue;
- CUDA-event-based pose GPU time, total processing capacity in FPS, sync
  spread, and source-to-output age when Edge timestamps use Unix time;
- Re-ID input/confirmed observation counts, mapped IDs, new/retained/exited
  IDs, churn, multi-Edge observations, clustering track state, and root
  association ratio;
- people/root/LOD/pose counts by Edge and camera, tensor shapes, estimated
  input bytes, and scene JSON bytes;
- input/output/telemetry drop totals and deltas, synchronization discards,
  decode/prepare failures, queue depths, exact Zenoh payload byte counters,
  process/system CPU and memory, process I/O, host non-loopback network rates,
  and available CUDA/NVML counters.

No images or Re-ID feature vectors are written. Per-observation Edge, camera,
frame, person index, bounding box, and assigned global ID are retained so a
labeled sequence can be matched offline. Re-ID churn and association fields
alone are operational continuity indicators, not accuracy scores. Compute
IDF1, HOTA, ID switches, mAP, or Rank-1 against ground truth during offline
evaluation.

`timings_ms.processing_total` is the real instrumented wall time.
`pipeline_stages_total` sums the non-overlapping pipeline stages, while
`instrumentation_and_unattributed` makes collection overhead and other Python
gaps visible. `pose_gpu` is already contained inside `pose_inference`, so do
not add it again when summing stages.

For load testing, keep the dataset, TensorRT setting, Edge/camera count, and
warm-up period fixed. Run separate labeled trials for each people count, then
compare `timings_ms.processing_total`, `timings_ms.pose_gpu`,
`rates.processing_capacity_fps`, drop deltas, queue growth, and resource
samples. A sustainable load should not show continuously growing queues or
drops even if a short-run FPS average looks acceptable.

The JSONL file can be loaded directly with pandas:

```python
import pandas as pd

events = pd.read_json("events.jsonl", lines=True)
batches = events[events["event"] == "batch"]
processing_ms = batches["timings_ms"].map(lambda value: value["processing_total"])
print(processing_ms.quantile([0.5, 0.95, 0.99]))
```
