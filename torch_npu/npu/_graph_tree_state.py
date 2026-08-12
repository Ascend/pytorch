"""Lightweight state shared by the public NPUGraph marker and graph trees."""


class MarkStepBox:
    # Negative values distinguish explicit user steps from Dynamo generations.
    mark_step_counter = 0


def mark_step_begin() -> None:
    """Indicate that a new inference or training iteration is about to begin."""
    MarkStepBox.mark_step_counter -= 1
