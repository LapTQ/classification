from abc import ABC, abstractmethod
import random

import torch
import torch.nn.functional as F


class IBatchRescaler(ABC):
    """Abstract interface for tensor-level batch rescaling strategies."""

    @abstractmethod
    def get_target_size(
        self, current_epoch: int = 0, global_step: int = 0
    ) -> tuple[int, int]:
        """Calculates or selects the target (height, width) for the current batch.

        Args:
            current_epoch (int): Current epoch number.
            global_step (int): Current global training step.

        Returns:
            tuple[int, int]: Target (height, width).
        """

    def __call__(
        self,
        x: torch.Tensor,
        current_epoch: int = 0,
        global_step: int = 0,
    ) -> torch.Tensor:
        """Rescales the input batch tensor [B, C, H, W] to target size.

        Args:
            x (torch.Tensor): Input batch tensor.
            current_epoch (int): Current epoch number.
            global_step (int): Current global training step.

        Returns:
            torch.Tensor: Rescaled batch tensor.
        """
        target_h, target_w = self.get_target_size(
            current_epoch=current_epoch, global_step=global_step
        )
        if x.shape[-2] == target_h and x.shape[-1] == target_w:
            return x

        return F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )


class RandomBatchRescaler(IBatchRescaler):
    """Randomly selects a resolution for every batch or every step_interval steps."""

    def __init__(
        self,
        sizes: list[int],
        step_interval: int = 1,
    ) -> None:
        if not sizes:
            raise ValueError("sizes list must not be empty.")
        self.sizes: list[tuple[int, int]] = [(s, s) for s in sizes]
        self.step_interval: int = max(1, step_interval)
        self._current_size: tuple[int, int] = self.sizes[0]
        self._last_step: int = -1

    def get_target_size(
        self, current_epoch: int = 0, global_step: int = 0
    ) -> tuple[int, int]:
        if (
            self._last_step < 0
            or (global_step - self._last_step) >= self.step_interval
        ):
            self._current_size = random.choice(self.sizes)
            self._last_step = global_step
        return self._current_size


class ProgressiveBatchRescaler(IBatchRescaler):
    """Progressively changes image resolution according to epoch milestones.

    Each milestone defines a list of square sizes to randomly sample from.
    """

    def __init__(
        self,
        milestones: dict[int, list[int]],
        step_interval: int = 1,
    ) -> None:
        if not milestones:
            raise ValueError("Milestones dictionary must not be empty.")

        self.step_interval: int = max(1, step_interval)
        self._last_step: int = -1
        self._current_milestone_epoch: int = -1

        # Sort milestones by epoch ascending
        parsed: list[tuple[int, list[tuple[int, int]]]] = []
        for epoch, sizes in sorted(milestones.items(), key=lambda item: int(item[0])):
            if not sizes:
                raise ValueError(
                    f"Size list for milestone epoch {epoch} must not be empty."
                )
            candidate_list: list[tuple[int, int]] = [(s, s) for s in sizes]
            parsed.append((int(epoch), candidate_list))

        self.milestones: list[tuple[int, list[tuple[int, int]]]] = parsed
        self._current_size: tuple[int, int] = self.milestones[0][1][0]

    def get_target_size(
        self, current_epoch: int = 0, global_step: int = 0
    ) -> tuple[int, int]:
        candidates: list[tuple[int, int]] = self.milestones[0][1]
        matched_epoch: int = self.milestones[0][0]
        for epoch_milestone, size_list in self.milestones:
            if current_epoch >= epoch_milestone:
                candidates = size_list
                matched_epoch = epoch_milestone
            else:
                break

        if len(candidates) == 1:
            return candidates[0]

        if (
            self._last_step < 0
            or self._current_milestone_epoch != matched_epoch
            or (global_step - self._last_step) >= self.step_interval
        ):
            self._current_size = random.choice(candidates)
            self._last_step = global_step
            self._current_milestone_epoch = matched_epoch

        return self._current_size

