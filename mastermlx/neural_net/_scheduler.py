from __future__ import annotations

import inspect


def step_scheduler(scheduler, monitor_value):
    """Advance a scheduler without masking errors raised inside ``step``."""

    step = scheduler.step
    if monitor_value is None:
        step()
        return

    try:
        parameters = list(inspect.signature(step).parameters.values())
    except (TypeError, ValueError):
        # Some extension callables do not expose a Python signature.  Passing
        # the monitor is the only non-destructive choice; any error from the
        # callable remains visible to the caller.
        step(monitor_value)
        return

    if any(
        parameter.kind
        in {
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.VAR_POSITIONAL,
        }
        for parameter in parameters
    ):
        step(monitor_value)
        return

    keyword_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind == parameter.KEYWORD_ONLY
    ]
    if keyword_parameters:
        preferred = next(
            (
                parameter
                for parameter in keyword_parameters
                if parameter.name in {"metric", "metrics", "loss", "value"}
            ),
            keyword_parameters[0] if len(keyword_parameters) == 1 else None,
        )
        if preferred is not None:
            step(**{preferred.name: monitor_value})
            return
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters):
        step(metric=monitor_value)
        return
    step()


__all__ = ["step_scheduler"]
