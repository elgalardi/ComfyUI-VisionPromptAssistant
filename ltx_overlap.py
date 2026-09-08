from __future__ import annotations

import torch
from comfy_api.latest import io


class LTX25ForwardOverlapAssemble(io.ComfyNode):
    """Overlap-add two LTX clips while preserving an exact delivered frame count."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25ForwardOverlapAssemble",
            display_name="LTX 2.5 — Forward Context Overlap Assemble",
            category="video/ltx25",
            description=(
                "Delivers the first block, then blends its reserved future frames with "
                "the matching beginning of the continuation before appending new frames."
            ),
            inputs=[
                io.Image.Input("previous_frames"),
                io.Image.Input("continuation_frames"),
                io.Int.Input("first_block_frames", default=240, min=1, max=4096),
                io.Int.Input("overlap_frames", default=17, min=2, max=257),
                io.Int.Input("total_output_frames", default=480, min=2, max=8192),
            ],
            outputs=[io.Image.Output("images"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, previous_frames, continuation_frames, first_block_frames=240,
                overlap_frames=17, total_output_frames=480):
        first = int(first_block_frames)
        overlap = int(overlap_frames)
        total = int(total_output_frames)
        if previous_frames.shape[0] < first + overlap:
            raise ValueError(
                f"Previous block needs at least {first + overlap} frames; "
                f"received {previous_frames.shape[0]}."
            )
        if continuation_frames.shape[0] < overlap + 1:
            raise ValueError(
                f"Continuation needs at least {overlap + 1} frames; "
                f"received {continuation_frames.shape[0]}."
            )
        if previous_frames.shape[1:] != continuation_frames.shape[1:]:
            raise ValueError("Both LTX blocks must have identical output dimensions.")

        head = previous_frames[:first]
        old_future = previous_frames[first:first + overlap]
        new_start = continuation_frames[:overlap]
        # Smoothstep avoids a visible acceleration at either end of the dissolve.
        alpha = torch.linspace(
            0.0, 1.0, overlap, device=old_future.device, dtype=old_future.dtype
        )
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        alpha = alpha.view(overlap, 1, 1, 1)
        blended = old_future * (1.0 - alpha) + new_start * alpha
        tail = continuation_frames[overlap:]
        assembled = torch.cat((head, blended, tail), dim=0)
        if assembled.shape[0] < total:
            raise ValueError(
                f"Assembled result has {assembled.shape[0]} frames; {total} requested."
            )
        assembled = assembled[:total]
        status = (
            f"{total} frames · first delivery {first} · "
            f"forward overlap {overlap} · smoothstep blend"
        )
        return io.NodeOutput(assembled, status)
