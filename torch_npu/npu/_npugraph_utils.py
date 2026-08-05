"""Align NPUGraph trace events for clearer Chrome Trace visualization.

The utility merges virtual streams, rebuilds per-stream timestamps, aligns
cross-stream record/wait pairs, adds flow events, and validates the result.
The generated ``*.aligned.json`` is derived data; the raw dump is unchanged.
"""

import json
import re
from pathlib import Path as _Path

import torch


log = torch._logging.getArtifactLogger("torch_npu.npugraph", "cudagraphs")


# Virtual stream preprocessing.


def _stream_id_to_tid(stream_id):
    """Convert a numeric stream id to a trace tid."""
    return f"stream{stream_id}"


def _tid_to_stream_id(tid):
    """Convert a trace tid to a numeric stream id, or return None."""
    if tid.startswith("stream"):
        try:
            return int(tid[len("stream") :])
        except ValueError:
            return None
    return None


def _merge_stream_active(events, gap=0.0):
    """Merge streams activated by STREAM_ACTIVE events in place.

    Returns the modified events and the ``(source_tid, target_tid)`` pairs
    that were merged.
    """
    tid_events = {}
    for ev in events:
        tid_events.setdefault(ev.get("tid"), []).append(ev)
    for lst in tid_events.values():
        lst.sort(key=lambda e: e.get("ts", 0))

    stream_active = [
        ev
        for ev in events
        if ((ev.get("args") or {}).get("Task Type") == "STREAM_ACTIVE" or ev.get("name") == "STREAM_ACTIVE")
        and "Active Stream Id" in (ev.get("args") or {})
    ]
    stream_active.sort(key=lambda e: e.get("ts", 0))

    moved = []
    for ev in stream_active:
        args = ev.get("args") or {}  # pylint: disable=redefined-outer-name
        active_id = args.get("Active Stream Id")
        if active_id is None:
            continue
        source_tid = _stream_id_to_tid(active_id)
        target_tid = ev.get("tid")
        if not target_tid or source_tid == target_tid:
            continue
        if source_tid not in tid_events:
            continue
        target_evs = tid_events.get(target_tid, [])
        source_evs = tid_events.get(source_tid, [])
        if not source_evs:
            continue
        target_end = max(e.get("ts", 0) + e.get("dur", 0) for e in target_evs) if target_evs else 0
        source_min = min(e.get("ts", 0) for e in source_evs)
        offset = (target_end + gap) - source_min
        target_stream_id = _tid_to_stream_id(target_tid)
        for e in source_evs:
            e["ts"] = e.get("ts", 0) + offset
            e["tid"] = target_tid
            if target_stream_id is not None:
                args_e = e.get("args")
                if isinstance(args_e, dict) and "Stream Id" in args_e:
                    args_e["Stream Id"] = target_stream_id
        tid_events[target_tid] = target_evs + source_evs
        tid_events.pop(source_tid, None)
        moved.append((source_tid, target_tid))

    return events, moved


# Control-event prefixes are excluded from name cleanup.
CONTROL_PREFIXES = (
    "EVENT_RECORD_",
    "EVENT_WAIT_",
    "MEM_WRITE_VALUE_",
    "MEM_WAIT_VALUE_",
    "EVENT_RESET_",
)


def _parse_control(name: str):
    """Return the control-event kind and handle, or ``(None, None)``."""
    if name.startswith("EVENT_RECORD_"):
        return "EVENT_RECORD", name.split("_")[-1]
    if name.startswith("EVENT_WAIT_"):
        return "EVENT_WAIT", name.split("_")[-1]
    if name.startswith("MEM_WRITE_VALUE_"):
        return "MEM_WRITE_VALUE", name.split("_")[-1]
    if name.startswith("MEM_WAIT_VALUE_"):
        return "MEM_WAIT_VALUE", name.split("_")[-1]
    return None, None


def _clean_name(name: str) -> str:
    """Remove hash-like underscore-separated components from an operator name."""
    parts = name.split("_")
    new_parts = []
    for part in parts:
        if re.fullmatch(r"[a-z0-9]+", part) and sum(c.isdigit() for c in part) > 5:
            continue
        new_parts.append(part)
    return "_".join(new_parts) if new_parts else name


def _align_trace(events):
    """Align events and return aligned events plus pair diagnostics."""
    min_gap = 0.2  # Fixed visualization gap within a stream.
    eps = 1e-9  # Floating-point tolerance.

    # Normalize reset duration.
    for ev in events:
        name = ev.get("name", "")
        if name.startswith("EVENT_RESET_"):
            ev["dur"] = 0.2

    # Group events by tid and preserve their original order.
    tid_indices = {}
    for idx, ev in enumerate(events):
        tid = ev.get("tid")
        tid_indices.setdefault(tid, []).append(idx)
    for tid, indices in tid_indices.items():
        indices.sort(key=lambda i: (events[i].get("ts", 0), i))

    tid_pos = {tid: {idx: pos for pos, idx in enumerate(indices)} for tid, indices in tid_indices.items()}

    # Rebuild timestamps from zero with a fixed gap.
    for tid, indices in tid_indices.items():
        prev_end = 0.0
        for pos, idx in enumerate(indices):
            ev = events[idx]
            gap = 0.0 if pos == 0 else min_gap
            ev["ts"] = prev_end + gap
            prev_end = ev["ts"] + ev["dur"]

    # Pair control events by handle.
    pair_map = {}
    for idx, ev in enumerate(events):
        kind, cid = _parse_control(ev.get("name", ""))
        if not kind:
            continue
        slot = pair_map.setdefault(cid, {})
        if kind in ("EVENT_RECORD", "MEM_WRITE_VALUE"):
            slot["record"] = idx
        elif kind in ("EVENT_WAIT", "MEM_WAIT_VALUE"):
            slot["wait"] = idx

    # Separate same-stream pairs from cross-stream pairs.
    same_tid_pairs = []
    cross_tid_pairs = []
    for cid, slot in pair_map.items():
        if "record" in slot and "wait" in slot:
            r_idx = slot["record"]
            w_idx = slot["wait"]
            if events[r_idx]["tid"] == events[w_idx]["tid"]:
                same_tid_pairs.append((events[r_idx]["name"], events[w_idx]["name"]))
            else:
                cross_tid_pairs.append((cid, r_idx, w_idx))

    # Iteratively align cross-stream pairs.
    max_passes = 200
    for _ in range(max_passes):
        changed = False
        for cid, r_idx, w_idx in cross_tid_pairs:
            record = events[r_idx]
            wait = events[w_idx]
            record_end = record["ts"] + record["dur"]
            if wait["ts"] >= record_end - eps:
                continue  # The dependency is already satisfied.
            needed_dur = record_end - wait["ts"]
            delta = needed_dur - wait["dur"]
            if abs(delta) > eps:
                wait["dur"] = needed_dur
                # Shift tasks after the wait by the same delta.
                tid = wait["tid"]
                start_pos = tid_pos[tid][w_idx]
                indices = tid_indices[tid]
                for idx in indices[start_pos + 1 :]:
                    events[idx]["ts"] += delta
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("Alignment did not converge within pass limit.")

    # Record pairs that already satisfy the dependency.
    skipped_pairs = []
    for cid, r_idx, w_idx in cross_tid_pairs:
        record = events[r_idx]
        wait = events[w_idx]
        if wait["ts"] >= record["ts"] + record["dur"] - eps:
            skipped_pairs.append((record["name"], wait["name"]))

    # Add Chrome Trace flow start/end and instant marker events.
    flow_events = []
    for cid, slot in pair_map.items():
        if "record" in slot and "wait" in slot:
            r = events[slot["record"]]
            w = events[slot["wait"]]
            try:
                flow_id = int(cid)
            except ValueError:
                flow_id = abs(hash(cid)) % 1000000000000
            flow_events.append(
                {
                    "name": f"PAIR_{cid}",
                    "cat": "event_record->wait",
                    "ph": "s",
                    "pid": r.get("pid"),
                    "tid": r.get("tid"),
                    "ts": r["ts"] + r["dur"],
                    "id": flow_id,
                    "bp": "e",
                }
            )
            flow_events.append(
                {
                    "name": f"PAIR_{cid}",
                    "cat": "event_record->wait",
                    "ph": "f",
                    "pid": w.get("pid"),
                    "tid": w.get("tid"),
                    "ts": w["ts"],
                    "id": flow_id,
                    "bp": "e",
                }
            )
            flow_events.append(
                {
                    "name": f"PAIR_{cid}_START",
                    "cat": "event_record->wait",
                    "ph": "i",
                    "s": "t",
                    "pid": r.get("pid"),
                    "tid": r.get("tid"),
                    "ts": r["ts"] + r["dur"],
                }
            )
            flow_events.append(
                {
                    "name": f"PAIR_{cid}_END",
                    "cat": "event_record->wait",
                    "ph": "i",
                    "s": "t",
                    "pid": w.get("pid"),
                    "tid": w.get("tid"),
                    "ts": w["ts"],
                }
            )

    # Clean non-control operator names.
    for ev in events:
        name = ev.get("name")
        if isinstance(name, str):
            if name.startswith(CONTROL_PREFIXES):
                continue
            ev["name"] = _clean_name(name)

    events.extend(flow_events)

    return events, same_tid_pairs, skipped_pairs


def _validate_alignment(events, skipped_pairs=None):
    import math

    if skipped_pairs is None:
        skipped_pairs = []
    skipped_set = {tuple(p) for p in skipped_pairs}

    non_finite = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if isinstance(ts, float) and not math.isfinite(ts):
            non_finite.append(("ts", ev.get("name")))
        if isinstance(dur, float) and not math.isfinite(dur):
            non_finite.append(("dur", ev.get("name")))
    if non_finite:
        preview = "\n".join(f"{field} {name}" for field, name in non_finite[:10])
        raise RuntimeError(f"Alignment check failed: non-finite values detected ({len(non_finite)}).\n{preview}")

    pairs = {}
    for idx, ev in enumerate(events):
        if ev.get("ph") != "X":
            continue
        kind, cid = _parse_control(ev.get("name", ""))
        if not kind:
            continue
        slot = pairs.setdefault(cid, {})
        if kind in ("EVENT_RECORD", "MEM_WRITE_VALUE"):
            slot["record"] = idx
        elif kind in ("EVENT_WAIT", "MEM_WAIT_VALUE"):
            slot["wait"] = idx

    mismatches = []
    for cid, slot in pairs.items():
        if "record" in slot and "wait" in slot:
            r = events[slot["record"]]
            w = events[slot["wait"]]
            if r.get("tid") == w.get("tid"):
                continue
            if (r.get("name"), w.get("name")) in skipped_set:
                continue
            r_end = r["ts"] + r["dur"]
            w_end = w["ts"] + w["dur"]
            if w["ts"] < r_end - 1e-6 and abs(r_end - w_end) > 1e-6:
                mismatches.append((cid, r_end, w_end))

    overlaps = []
    tid_indices = {}
    for idx, ev in enumerate(events):
        if ev.get("ph") != "X":
            continue
        tid = ev.get("tid")
        tid_indices.setdefault(tid, []).append(idx)
    for tid, indices in tid_indices.items():
        indices.sort(key=lambda i: (events[i].get("ts", 0), i))
        prev_end = None
        for idx in indices:
            ev = events[idx]
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            if prev_end is None:
                if abs(ts - 0.0) > 1e-6:
                    overlaps.append((tid, None, ev.get("name")))
                prev_end = ts + dur
                continue
            expected_ts = prev_end + 0.2
            if abs(ts - expected_ts) > 1e-6:
                overlaps.append((tid, None, ev.get("name")))
                if len(overlaps) >= 10:
                    break
            prev_end = ts + dur
        if len(overlaps) >= 10:
            break

    if overlaps:
        preview = "\n".join(f"{tid}: {right}" for tid, _, right in overlaps)
        raise RuntimeError(f"Alignment check failed: gap mismatch detected ({len(overlaps)}+).\n{preview}")

    return mismatches


def align_trace_json(input, output, merge_gap=0.0):  # pylint: disable=redefined-builtin
    in_path = _Path(input)
    out_path = _Path(output) if output else in_path.with_suffix(in_path.suffix.replace(".json", "") + ".aligned.json")

    with in_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    # Merge virtual streams before alignment.
    events, moved = _merge_stream_active(events, gap=merge_gap)
    if moved:
        log.debug("Merged streams: %s", moved)

    aligned, same_tid_pairs, skipped_pairs = _align_trace(events)
    mismatches = _validate_alignment(aligned, skipped_pairs=skipped_pairs)
    if mismatches:
        preview = "\n".join(f"{cid} record_end={r_end} wait_end={w_end}" for cid, r_end, w_end in mismatches[:10])
        raise RuntimeError(f"Alignment check failed: {len(mismatches)} mismatches.\n{preview}")
    if same_tid_pairs:
        log.debug("Same-tid pairs (no alignment): %s", same_tid_pairs)
    if skipped_pairs:
        log.debug(
            "Cross-tid pairs skipped (wait starts after record end): %s",
            skipped_pairs,
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    log.debug("Aligned trace was written to %s", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align trace control tasks and clean task names.")
    parser.add_argument("input", help="input trace json path")
    parser.add_argument(
        "-o",
        "--output",
        help="output json path (default: input with .aligned suffix)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="time gap inserted when merging virtual streams (default: 0.0)",
    )
    args = parser.parse_args()
    align_trace_json(args.input, args.output, args.merge_gap)
