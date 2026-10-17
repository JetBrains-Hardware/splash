"""A small DFlash2 draft checkpoint and the af4g64 rows MLX's affine
quantization rounds each of its projections to, an independent oracle of the
draft's prepared planes, then the golden hashes of its images; then the same
checkpoint as an MLX conversion stores it (mlx_lm.convert -q), its
projections and selector codebooks quantized to those rows, whose layer
images hash as the BF16 checkpoint's and whose codebooks the oracle decodes
to bf16."""

import argparse
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dev.tests.fixture_files import write_safetensors  # noqa: E402

F32 = struct.Struct("<f")


def f32(value):
    """value rounded once to float32: a float32 sum, difference or quotient
    computed in double and rounded once is the float32 operation's result."""
    return F32.unpack(F32.pack(value))[0]


def bf16(value):
    """The bits of the BF16 nearest a finite value, ties to even."""
    bits = struct.unpack("<I", F32.pack(value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def half_away(value):
    """value rounded to an integer, halves away from zero."""
    return math.copysign(math.floor(abs(value) + 0.5), value)


def quantized_group(weights):
    """64 weights quantized to 4 bits as MLX's affine quantization rounds them
    (mlx.core.quantize, its Metal kernel, whose maximum starts at 0), as a
    native af4g64 block: the scale and bias as BF16, then the packed codes."""
    low, high = min(weights), max(0.0, *weights)
    low_edge = abs(low) > abs(high)
    scale = max(f32(f32(high - low) / 15), f32(1e-7))
    if not low_edge:
        scale = -scale
    edge = low if low_edge else high
    q0 = half_away(f32(edge / scale))
    bias = 0.0
    if q0 != 0:
        scale, bias = f32(edge / q0), edge
    codes = [
        int(min(max(half_away(f32(f32(w - bias) / scale)), 0), 15)) for w in weights
    ]
    packed = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))
    return struct.pack("<HH", bf16(scale), bf16(bias)) + packed


def mlx_tensors(name, rows, columns, native):
    """The weight codes (U32), scales and biases (BF16) MLX stores for a
    module of af4g64 rows `native`: each block's packed codes are the
    little-endian bytes of its codes' U32 words."""
    blocks = [native[i : i + 36] for i in range(0, len(native), 36)]
    groups = columns // 64
    return {
        name + ".weight": (
            [rows, columns // 8],
            "U32",
            b"".join(b[4:] for b in blocks),
        ),
        name + ".scales": ([rows, groups], "BF16", b"".join(b[0:2] for b in blocks)),
        name + ".biases": ([rows, groups], "BF16", b"".join(b[2:4] for b in blocks)),
    }


def decoded(native):
    """The bf16 values of af4g64 rows, scale * code + bias rounded once to
    float32, as the gather's fma, then to bf16."""
    out = []
    for start in range(0, len(native), 36):
        scale_bits, bias_bits = struct.unpack_from("<HH", native, start)
        scale = struct.unpack("<f", struct.pack("<I", scale_bits << 16))[0]
        bias = struct.unpack("<f", struct.pack("<I", bias_bits << 16))[0]
        for byte in native[start + 4 : start + 36]:
            for code in (byte & 15, byte >> 4):
                out.append(bf16(f32(code * scale + bias)))
    return struct.pack(f"<{len(out)}H", *out)


def draft_fixture(root, quantized=False):
    """A two-layer DFlash2 checkpoint of width 256, its BF16 safetensors as
    its repository releases them, and expected/native.safetensors: each
    projection's native af4g64 rows, as MLX's affine quantization rounds
    them. With `quantized`, the checkpoint stores those rows as MLX does, and
    the selector codebooks quantized as well, and the expected codebooks are
    their bf16 decode."""
    tensors, values, expected = {}, {}, {}

    def add(name, shape, data=None):
        count = math.prod(shape)
        if data is None:
            # Multiples of 1/64 within 4 of zero, each exactly a BF16.
            # Seeded by the tensor's place, not by the tensors stored so far,
            # which an MLX module's three take the place of one.
            seed = len(values) + 1
            data = [((i * 37 + seed * 11) % 509 - 254) / 64 for i in range(count)]
        values[name] = data
        tensors[name] = (shape, "BF16", struct.pack(f"<{count}H", *map(bf16, data)))

    def projection(name, rows, columns, data=None):
        add(name + ".weight", [rows, columns], data)
        weights = values[name + ".weight"]
        native = b"".join(
            quantized_group(weights[start : start + 64])
            for start in range(0, len(weights), 64)
        )
        expected[name] = ([rows, columns // 64 * 36], "U8", native)
        if quantized:
            del tensors[name + ".weight"]
            tensors.update(mlx_tensors(name, rows, columns, native))

    def codebook(name):
        add(name, [256, 256])
        if not quantized:
            return
        weights = values[name]
        native = b"".join(
            quantized_group(weights[start : start + 64])
            for start in range(0, len(weights), 64)
        )
        del tensors[name]
        tensors.update(mlx_tensors(name, 256, 256, native))
        expected[name] = ([256, 256], "BF16", decoded(native))

    for layer in range(2):
        p = f"layers.{layer}."
        a = p + "self_attn."
        dynamic = None
        if layer == 0:
            # Two first rows of edge cases: all zero, where no scale puts 0 on
            # a code; constant; 0 to 15 with halves, which round away from
            # zero; a minimum farther from zero than the maximum; then groups
            # below zero, whose range still ends at 0: constant, spread, a
            # maximum near zero, and -15 to -1 with halves.
            dynamic = [0.0] * 64 + [0.75] * 64
            dynamic += [2.5, 8.5, *(float(i % 16) for i in range(62))]
            dynamic += [(i % 20) / 4 - 4 for i in range(64)]
            dynamic += [-0.75] * 64
            dynamic += [-(i % 16 + 1) / 4 for i in range(64)]
            dynamic += [-1 / 64, *(-(i % 32 + 1) / 8 for i in range(63))]
            dynamic += [-2.5, -8.5, *(-float(i % 15 + 1) for i in range(62))]
            dynamic += [((i * 37 + 11) % 509 - 254) / 64 for i in range(254 * 256)]
        add(p + "input_layernorm.weight", [256])
        add(p + "attention_conv.base_kernel", [2, 2, 256])
        projection(p + "attention_conv.kernel_projection", 256, 256, dynamic)
        # The fused q|k|v projection's parts.
        for name, rows in (("q_proj", 256), ("k_proj", 128), ("v_proj", 128)):
            projection(a + name, rows, 256)
        add(a + "q_norm.weight", [64])
        add(a + "k_norm.weight", [64])
        projection(a + "o_proj", 256, 256)
        add(p + "post_attention_layernorm.weight", [256])
        add(p + "mlp_conv.base_kernel", [2, 2, 256])
        projection(p + "mlp_conv.kernel_projection", 256, 256)
        projection(p + "mlp.gate_proj", 256, 256)
        projection(p + "mlp.up_proj", 256, 256)
        projection(p + "mlp.down_proj", 256, 256)
    projection("fc", 256, 256)
    add("hidden_norm.weight", [256])
    add("norm.weight", [256])
    projection("candidate_selector.hidden_projection", 256, 256)
    codebook("candidate_selector.predecessor_codebook")
    codebook("candidate_selector.successor_codebook")
    write_safetensors(root / "model.safetensors", tensors)
    (root / "expected").mkdir()
    write_safetensors(root / "expected" / "native.safetensors", expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("goldens", type=Path)
    args = parser.parse_args()
    for quantized, section in ((False, "draft_images"), (True, "draft_images_mlx")):
        with tempfile.TemporaryDirectory(
            prefix="splash-draft-preparation-"
        ) as directory:
            root = Path(directory)
            draft_fixture(root, quantized)
            command = [
                str(args.binary.resolve()),
                str(args.metallib.resolve()),
                str(root),
                str(args.goldens.resolve()),
                section,
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, result.stdout + result.stderr
            print(f"{section}: {result.stdout.splitlines()[-1]}")


if __name__ == "__main__":
    main()
