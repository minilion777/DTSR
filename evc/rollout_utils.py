from __future__ import annotations

from typing import Sequence


def update_active_vehicle_ids(step_vehicle_ids: Sequence[int], transitions) -> list[int]:
    """Keep vehicle identifiers aligned with non-terminal environment transitions."""

    return [
        int(vehicle_id)
        for vehicle_id, transition in zip(step_vehicle_ids, transitions)
        if not bool(transition.done)
    ]


__all__ = ["update_active_vehicle_ids"]
