#!/usr/bin/env python3
"""Early-stopping criterion for standalone PSRO oracle continuation runs.

The trainer reports metrics on a tqdm postfix line, once or twice per
optimizer step.  A single step is far too noisy to stop a run on, so this
module aggregates steps into fixed windows aligned with the checkpoint
interval and requires the criterion to hold on several *consecutive* windows.

A run stops only when both hold on `patience` consecutive windows:

  * the primary metric's window mean is <= `threshold`, and
  * the guard metric's window mean is >= `guard_min`.

The guard exists so a defender cannot satisfy the primary metric by refusing
everything: a window that passes the primary check but fails the guard does
not count, and resets the streak.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def metric_re(name: str) -> re.Pattern:
    """Match `name=<float>` where name may contain '/' and '_'."""
    return re.compile(re.escape(name) + r"=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_line(line: str, patterns: dict[str, re.Pattern]):
    """Return (step, {metric: value}) for a tqdm postfix line, else None."""
    step_match = STEP_RE.search(line)
    if step_match is None:
        return None
    values: dict[str, float] = {}
    for name, pattern in patterns.items():
        found = pattern.search(line)
        if found is not None:
            values[name] = float(found.group(1))
    if not values:
        return None
    return int(step_match.group(1)), values


@dataclass
class Measurement:
    window_end: int
    primary: float
    guard: float | None
    primary_ok: bool
    guard_ok: bool
    streak: int
    ceiling: float | None = None
    ceiling_ok: bool = True

    def describe(self) -> str:
        guard = "n/a" if self.guard is None else f"{self.guard:.4f}"
        ceiling = "n/a" if self.ceiling is None else f"{self.ceiling:.4f}"
        return (f"step {self.window_end}: primary={self.primary:.4f} "
                f"guard={guard} ceiling={ceiling} primary_ok={self.primary_ok} "
                f"guard_ok={self.guard_ok} ceiling_ok={self.ceiling_ok} "
                f"streak={self.streak}")


@dataclass
class EarlyStop:
    metric: str
    guard_metric: str
    threshold: float = 0.02
    guard_min: float = 0.912
    window: int = 10
    patience: int = 3
    min_step: int = 30
    min_coverage: int = 3

    # A degenerating model can drive the primary metric to zero by emitting
    # unparseable output: the judge then scores no harmful response, which
    # looks identical to a defender that learned to refuse.  `ceiling_metric`
    # must stay BELOW `ceiling_max` for a window to count as success, and
    # `abort_max` stops the run outright as a collapse rather than a win.
    ceiling_metric: str = ""
    ceiling_max: float = 1.0
    abort_max: float = 1.1
    abort_patience: int = 2

    # Attackers never reach an absolute success threshold against a trained
    # defender (best observed ~0.57), so a "<= threshold" rule would never fire
    # and they would always burn the full budget. mode="plateau" stops instead
    # when the primary metric has not improved by `min_delta` over the best
    # window seen, for `patience` consecutive windows. Higher is better for an
    # attacker, so improvement means an increase.
    mode: str = "threshold"          # "threshold" | "plateau"
    min_delta: float = 0.01
    best_primary: float = float("-inf")
    best_step: int = 0
    plateau_streak: int = 0

    triggered: bool = False
    collapsed: bool = False
    abort_streak: int = 0
    reason: str = ""
    streak: int = 0
    measurements: list[Measurement] = field(default_factory=list)
    _seen: dict[int, tuple[float, float | None, float | None]] = field(default_factory=dict)
    _done: set[int] = field(default_factory=set)

    def __post_init__(self):
        if self.mode not in {"threshold", "plateau"}:
            raise ValueError("early-stop mode must be threshold or plateau")
        if min(self.window, self.patience, self.min_coverage, self.abort_patience) < 1:
            raise ValueError("early-stop windows, patience and coverage must be positive")
        if self.min_coverage > self.window or self.min_step < 0:
            raise ValueError("invalid early-stop coverage or minimum step")

    def configuration(self) -> dict:
        return {key: getattr(self, key) for key in (
            "metric", "guard_metric", "threshold", "guard_min", "window", "patience",
            "min_step", "min_coverage", "ceiling_metric", "ceiling_max", "abort_max",
            "abort_patience", "mode", "min_delta",
        )}

    def report(self) -> dict:
        return {"configuration": self.configuration(), "triggered": self.triggered,
                "collapsed": self.collapsed, "reason": self.reason,
                "measurements": [asdict(value) for value in self.measurements]}

    def patterns(self) -> dict[str, re.Pattern]:
        names = [self.metric, self.guard_metric]
        if self.ceiling_metric:
            names.append(self.ceiling_metric)
        return {name: metric_re(name) for name in names if name}

    def observe(self, step: int, values: dict[str, float]):
        """Record one step; return a Measurement when a window closes."""
        primary = values.get(self.metric)
        if primary is None:
            return None
        self._seen[step] = (primary, values.get(self.guard_metric),
                            values.get(self.ceiling_metric)
                            if self.ceiling_metric else None)

        if step <= 0 or step % self.window != 0 or step in self._done:
            return None
        low = step - self.window + 1
        points = [self._seen[s] for s in range(low, step + 1) if s in self._seen]
        if len(points) < self.min_coverage:
            self.streak = self.plateau_streak = self.abort_streak = 0
            return None
        if self._done and step - max(self._done) > self.window:
            self.streak = self.plateau_streak = self.abort_streak = 0
        self._done.add(step)

        primary_mean = sum(p[0] for p in points) / len(points)
        guards = [p[1] for p in points if p[1] is not None]
        guard_mean = sum(guards) / len(guards) if guards else None
        ceilings = [p[2] for p in points if p[2] is not None]
        ceiling_mean = sum(ceilings) / len(ceilings) if ceilings else None

        # A missing guard must not turn "unknown" into a successful window.
        guard_ok = not self.guard_metric or (
            len(guards) >= self.min_coverage and guard_mean >= self.guard_min
        )
        ceiling_ok = not self.ceiling_metric or (
            len(ceilings) >= self.min_coverage and ceiling_mean <= self.ceiling_max
        )
        if self.mode == "plateau":
            if primary_mean > self.best_primary + self.min_delta:
                self.best_primary = primary_mean
                self.best_step = step
                self.plateau_streak = 0
            else:
                self.plateau_streak += 1
            primary_ok = self.plateau_streak >= self.patience
            self.streak = self.patience if (primary_ok and ceiling_ok) else 0
        else:
            primary_ok = primary_mean <= self.threshold
            self.streak = (self.streak + 1
                           if (primary_ok and guard_ok and ceiling_ok) else 0)

        measurement = Measurement(step, primary_mean, guard_mean,
                                  primary_ok, guard_ok, self.streak,
                                  ceiling_mean, ceiling_ok)
        self.measurements.append(measurement)

        # Collapse abort: the ceiling metric running away means the model is
        # degenerating, and every downstream metric is then meaningless.
        if len(ceilings) >= self.min_coverage and ceiling_mean >= self.abort_max:
            self.abort_streak += 1
        else:
            self.abort_streak = 0
        if (self.abort_streak >= self.abort_patience and not self.triggered
                and not self.collapsed):
            self.collapsed = True
            self.reason = (
                f"COLLAPSE: {self.ceiling_metric} >= {self.abort_max} on "
                f"{self.abort_patience} consecutive {self.window}-step windows "
                f"ending at step {step} (means: "
                + ", ".join("n/a" if m.ceiling is None else f"{m.ceiling:.4f}"
                            for m in self.measurements[-self.abort_patience:])
                + "); the model is degenerating, not converging"
            )
            return measurement

        if (self.streak >= self.patience and step >= self.min_step
                and not self.triggered):
            self.triggered = True
            if self.mode == "plateau":
                self.reason = (
                    f"PLATEAU: {self.metric} did not improve by {self.min_delta} "
                    f"over {self.patience} consecutive {self.window}-step windows; "
                    f"best {self.best_primary:.4f} at step {self.best_step}"
                )
                return measurement
            self.reason = (
                f"{self.metric} <= {self.threshold} on {self.patience} "
                f"consecutive {self.window}-step windows ending at step {step} "
                f"(means: "
                + ", ".join(f"{m.primary:.4f}"
                            for m in self.measurements[-self.patience:])
                + f"); guard {self.guard_metric} >= {self.guard_min} throughout "
                f"(means: "
                + ", ".join("n/a" if m.guard is None else f"{m.guard:.4f}"
                            for m in self.measurements[-self.patience:]) + ")"
                + (f"; ceiling {self.ceiling_metric} <= {self.ceiling_max} "
                   f"throughout (means: "
                   + ", ".join("n/a" if m.ceiling is None else f"{m.ceiling:.4f}"
                               for m in self.measurements[-self.patience:]) + ")"
                   if self.ceiling_metric else "")
            )
        return measurement


def replay(lines, criterion: EarlyStop):
    """Feed an iterable of log lines through a criterion (offline testing)."""
    patterns = criterion.patterns()
    for line in lines:
        parsed = parse_line(line, patterns)
        if parsed is None:
            continue
        criterion.observe(*parsed)
        if criterion.triggered or criterion.collapsed:
            break
    return criterion
