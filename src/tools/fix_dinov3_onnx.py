#!/usr/bin/env python3
"""
Script to fix DINOv3 / ViT ONNX models containing dynamic control flow (If nodes) in rope_embed.
TensorRT's ONNX parser requires both branches of IIfConditionalOutputLayer to have identical shapes.
In PyTorch's RoPE export, the dimension diff check creates two 'If' nodes with mismatched branch shapes ([2] vs [1]),
even though at runtime the condition is always static and constant.

This script simplifies the graph by removing the If nodes and replacing them with a constant repeats tensor [1, 2].
"""

import os
import onnx
from onnx import helper

# ================= CẤU HÌNH TRỰC TIẾP =================
INPUT_ONNX_PATH: str = (
    "models/checkpoints/fs26/action_recognition/classification/v39.v37+merge_class_2_3/weights/best-epoch=23-val_f1=0.661.ckpt.onnx"
)
OUTPUT_ONNX_PATH: str = (
    "models/checkpoints/fs26/action_recognition/classification/v39.v37+merge_class_2_3/weights/best-epoch=23-val_f1=0.661.ckpt.fixed.onnx"
)
# =====================================================


def fix_dinov3_rope_embed(input_onnx_path: str, output_onnx_path: str) -> None:
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {input_onnx_path}")

    print(f"Loading ONNX model from: {input_onnx_path}")
    model = onnx.load(input_onnx_path, load_external_data=False)

    nodes_to_remove = {
        '/model/backbone/rope_embed/If',
        '/model/backbone/rope_embed/If_1',
        '/model/backbone/rope_embed/Cast_4',
        '/model/backbone/rope_embed/Less',
        '/model/backbone/rope_embed/Greater',
        '/model/backbone/rope_embed/Sub_1',
        '/model/backbone/rope_embed/Size',
        '/model/backbone/rope_embed/Shape_2',
        '/model/backbone/rope_embed/Constant_14',
        '/model/backbone/rope_embed/Constant_15',
        '/model/backbone/rope_embed/Constant_18',
        'Constant_771',
    }

    # Verify that the target If node is present
    has_target_node = any(n.name == '/model/backbone/rope_embed/If' for n in model.graph.node)
    if not has_target_node:
        print("Notice: '/model/backbone/rope_embed/If' not found. Checking for any If nodes...")
        if_nodes = [n.name for n in model.graph.node if n.op_type == 'If']
        print(f"Found If nodes: {if_nodes}")

    repeats_tensor = helper.make_tensor(
        name='/model/backbone/rope_embed/tile_repeats_tensor',
        data_type=onnx.TensorProto.INT64,
        dims=[2],
        vals=[1, 2]
    )
    repeats_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['/model/backbone/rope_embed/tile_repeats'],
        value=repeats_tensor,
        name='/model/backbone/rope_embed/Constant_tile_repeats'
    )

    new_nodes = []
    fixed = False
    for n in model.graph.node:
        if n.name in nodes_to_remove:
            continue
        if n.name == '/model/backbone/rope_embed/Tile':
            new_nodes.append(repeats_node)
            n.input[0] = '/model/backbone/rope_embed/Flatten_1_output_0'
            n.input[1] = '/model/backbone/rope_embed/tile_repeats'
            fixed = True
        new_nodes.append(n)

    if fixed:
        print("Successfully removed dynamic 'If' nodes and patched 'Tile' node in rope_embed.")
    else:
        print("Warning: Could not find '/model/backbone/rope_embed/Tile'. Check model architecture.")

    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)

    print("Checking ONNX model integrity...")
    onnx.checker.check_model(model)

    output_dir = os.path.dirname(output_onnx_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving fixed ONNX model to: {output_onnx_path}")
    onnx.save(model, output_onnx_path)
    print("Done!")


def main() -> None:
    fix_dinov3_rope_embed(INPUT_ONNX_PATH, OUTPUT_ONNX_PATH)


if __name__ == '__main__':
    main()

