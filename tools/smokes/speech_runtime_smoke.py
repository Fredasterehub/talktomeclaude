"""Real fixed-voice synthesis and bounded device-interruption smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
import wave
from array import array
from pathlib import Path

from talktomeclaude.speech import (
    PersistentSpeechRuntime,
    SoundDevicePlayback,
    SpawnSynthesisWorker,
    SpeechArtifact,
    SynthesisRequest,
)


class SmokeFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _quiet_copy(source: Path) -> Path:
    target = source.with_name(f"{source.stem}-quiet.wav")
    with wave.open(str(source), "rb") as reader:
        parameters = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if parameters.sampwidth != 2:
        raise SmokeFailure("unsupported_sample_width")
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, value in enumerate(samples):
        samples[index] = round(value * 0.03)
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(target), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(samples.tobytes())
    return target


def _trial(device: SoundDevicePlayback, artifact: SpeechArtifact) -> float:
    device.start(artifact, lambda _outcome: None)
    time.sleep(0.08)
    started = time.perf_counter_ns()
    confirmed = device.abort()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if not confirmed or not device.silence_confirmed:
        raise SmokeFailure("device_silence_unconfirmed")
    return elapsed_ms


def _run(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).resolve()
    config_before = _sha256(config_path)
    runtime = PersistentSpeechRuntime(
        args.voice,
        lambda voice: SpawnSynthesisWorker(voice),
        shutdown_deadline_seconds=10.0,
    )
    result_ready = threading.Event()
    results = []
    artifact: SpeechArtifact | None = None
    quiet_path: Path | None = None
    shutdown_clean = False
    try:
        runtime.submit(
            SynthesisRequest(
                0,
                "g6-physical-interruption",
                "TalkToMeClaude is verifying bounded interruption on the selected voice.",
            ),
            lambda result: (results.append(result), result_ready.set()),
        )
        if not result_ready.wait(args.synthesis_timeout):
            raise SmokeFailure("synthesis_timeout")
        if len(results) != 1 or results[0].artifact is None:
            raise SmokeFailure("synthesis_failed")
        artifact = results[0].artifact
        if not isinstance(artifact.payload, Path):
            raise SmokeFailure("artifact_not_path")
        quiet_path = _quiet_copy(artifact.payload)
        quiet_artifact = SpeechArtifact(
            generation=0,
            unit_id="g6-quiet-interruption",
            payload=quiet_path,
        )
        warmup_device = SoundDevicePlayback(abort_deadline_seconds=2.0)
        warmup_ms = _trial(warmup_device, quiet_artifact)
        device = SoundDevicePlayback(abort_deadline_seconds=0.25)
        timings = [_trial(device, quiet_artifact) for _ in range(args.trials)]
        ordered = sorted(timings)
        p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        if p95 > 250.0:
            raise SmokeFailure("p95_exceeded")
        shutdown_clean = runtime.shutdown()
        config_after = _sha256(config_path)
        if config_after != config_before:
            raise SmokeFailure("config_changed")
        return {
            "config_sha256": config_after,
            "device_silence_confirmed": True,
            "p95_interrupt_ms": round(p95, 3),
            "result_code": "passed",
            "selected_voice": args.voice,
            "shutdown_clean": shutdown_clean,
            "synthesis_artifact_bytes": artifact.payload.stat().st_size,
            "timed_trials": len(timings),
            "warmup_interrupt_ms": round(warmup_ms, 3),
        }
    finally:
        if not shutdown_clean:
            try:
                runtime.shutdown()
            except Exception:
                pass
        if quiet_path is not None:
            quiet_path.unlink(missing_ok=True)
        if artifact is not None:
            artifact.discard()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="rick")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--synthesis-timeout", type=float, default=180.0)
    parser.add_argument(
        "--config",
        default=str(Path.home() / ".config" / "talktomeclaude" / "config.json"),
    )
    args = parser.parse_args()
    if args.trials < 50:
        parser.error("trials must be at least 50")
    try:
        report = _run(args)
    except Exception as exc:
        report = {
            "failure_code": getattr(exc, "code", type(exc).__name__),
            "result_code": "failed",
        }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["result_code"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
