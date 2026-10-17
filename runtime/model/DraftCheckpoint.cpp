#include "model/DraftCheckpoint.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/Gguf.h"
#include "model/DFlashDraft.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// Plans one image from the checkpoint's tensors, which must have the shapes
// the draft reads: BF16 as a DFlash2 release stores them, or each projection
// and selector codebook quantized as an MLX conversion (mlx_lm.convert -q)
// stores it.
class Builder {
public:
  Builder(const SafetensorsCheckpoint &checkpoint, std::string name, uint32_t layer, uint32_t type)
      : checkpoint_(checkpoint), image_(std::move(name), layer, type) {}

  // A tensor copied as stored: a norm or a convolution's base kernel.
  void stored(const std::string &name, const std::vector<uint64_t> &shape) {
    const SourceTensor &tensor = require(name, shape);
    image_.copy({name, ggml::kBF16, tensor.offset, 1, tensor.bytes, {}, tensor.file});
  }

  // A selector codebook of [rows, columns], which the draft kernels read as
  // bf16: as stored, or the bf16 values of its MLX quantization's decode.
  // A release stores the codebook as a tensor of its own name, without the
  // .weight of a quantized module's codes.
  void codebook(const std::string &name, uint64_t rows, uint64_t columns) {
    if (!checkpoint_.find(name + ".scales")) return stored(name, {rows, columns});
    const std::optional<gguf::TensorRows> source = safetensors::mlxQuantizedRows(checkpoint_, name, rows, columns);
    if (!source || !gguf_embedding_format(gguf_format_of(source->type)))
      throw WeightStoreError(name + " is in a format the draft preparation does not decode");
    image_.copy(*source, gguf::Conversion::DequantizeToBfloat16);
  }

  // A projection of `columns` inputs: its parts' rows in order, as planes of
  // the MLX quantization they share, or af4g64 planes of BF16 parts, which
  // the writer quantizes as MLX's affine quantization rounds them.
  void projection(std::initializer_list<std::pair<std::string, uint64_t>> parts, uint64_t columns) {
    std::vector<gguf::TensorRows> sources;
    uint64_t rows = 0;
    size_t quantized = 0;
    for (const auto &[name, count] : parts) {
      if (std::optional<gguf::TensorRows> source = safetensors::mlxQuantizedRows(checkpoint_, name, count, columns)) {
        if (quantized != sources.size() || (quantized && source->type != sources.front().type))
          throw WeightStoreError(name + " is stored in another format than its projection's other parts");
        sources.push_back(*source);
        ++quantized;
      } else {
        if (quantized) throw WeightStoreError(name + " is stored in another format than its projection's other parts");
        const QuantFormat &format = kQuantFormats[GGUF_FMT_AF4G64];
        const SourceTensor &tensor = require(name + ".weight", {count, columns});
        sources.push_back({name, format.ggml_type, 0, count, ggufRowBytes(format, columns), {}, tensor.file,
                           {.bfloat16 = &tensor}});
      }
      rows += count;
    }
    const uint32_t format = quantized ? gguf_format_of(sources.front().type) : GGUF_FMT_AF4G64;
    gguf::Repack repack = image_.planes(format, rows, columns, parts.begin()->first);
    repack.sources = std::move(sources);
    image_.repack(std::move(repack));
  }

  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const SourceTensor &require(const std::string &name, const std::vector<uint64_t> &shape) const {
    const SourceTensor &tensor = checkpoint_.require(name);
    if (tensor.dtype != "BF16" || tensor.shape != shape)
      throw WeightStoreError("source tensor type or shape does not match: " + name);
    return tensor;
  }

  const SafetensorsCheckpoint &checkpoint_;
  gguf::ImageBuilder image_;
};

gguf::Image layerImage(const SafetensorsCheckpoint &checkpoint, const DFlashDraftLayout &layout, uint32_t layer) {
  Builder b(checkpoint, "layer-" + std::to_string(layer) + ".bin", layer, 0);
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.";
  const uint64_t hidden = layout.hiddenSize, kv = uint64_t{layout.kvHeads} * layout.attentionHeadDimension;
  const auto convolution = [&](const std::string &name) {
    b.stored(name + ".base_kernel", {SPLASH_DRAFT_CONVOLUTION_STAGES, SPLASH_DRAFT_CONVOLUTION_TAPS, hidden});
    b.projection({{name + ".kernel_projection", layout.dynamicSize}}, hidden);
  };
  b.stored(prefix + "input_layernorm.weight", {hidden});
  convolution(prefix + "attention_conv");
  b.projection({{attention + "q_proj", layout.attentionSize}, {attention + "k_proj", kv}, {attention + "v_proj", kv}},
               hidden);
  b.stored(attention + "q_norm.weight", {layout.attentionHeadDimension});
  b.stored(attention + "k_norm.weight", {layout.attentionHeadDimension});
  b.projection({{attention + "o_proj", hidden}}, layout.attentionSize);
  b.stored(prefix + "post_attention_layernorm.weight", {hidden});
  convolution(prefix + "mlp_conv");
  b.projection({{prefix + "mlp.gate_proj", layout.intermediateSize}}, hidden);
  b.projection({{prefix + "mlp.up_proj", layout.intermediateSize}}, hidden);
  b.projection({{prefix + "mlp.down_proj", hidden}}, layout.intermediateSize);
  return b.finish();
}

gguf::Image modelImage(const SafetensorsCheckpoint &checkpoint, const DFlashDraftLayout &layout) {
  Builder b(checkpoint, "model.bin", layout.layers, 1);
  const std::string selector = "candidate_selector.";
  const uint64_t hidden = layout.hiddenSize;
  b.projection({{"fc", hidden}}, layout.targetHiddenSize);
  b.stored("hidden_norm.weight", {hidden});
  b.stored("norm.weight", {hidden});
  b.projection({{selector + "hidden_projection", layout.selectorRank}}, hidden);
  b.codebook(selector + "predecessor_codebook", layout.vocabularySize, layout.selectorRank);
  b.codebook(selector + "successor_codebook", layout.vocabularySize, layout.selectorRank);
  return b.finish();
}

} // namespace

std::vector<gguf::Image> planDraftImages(const SafetensorsCheckpoint &checkpoint, const DFlashDraftLayout &layout) {
  std::vector<gguf::Image> images;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) images.push_back(layerImage(checkpoint, layout, layer));
  images.push_back(modelImage(checkpoint, layout));
  return images;
}

uint64_t draftImageBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout) {
  const SafetensorsCheckpoint checkpoint(directory);
  uint64_t bytes = 0;
  for (const gguf::Image &image : planDraftImages(checkpoint, layout)) bytes += image.bytes;
  return bytes;
}

DraftCheckpointLoader::DraftCheckpointLoader(metal::MetalBackend &backend, WeightImages &images,
                                             const std::filesystem::path &directory, const DFlashDraftLayout &layout)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)) {
  planned_->images = planDraftImages(planned_->checkpoint, layout);
}

WeightFile DraftCheckpointLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"draft/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       }});
}

WeightFile DraftCheckpointLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return open(index);
}

WeightFile DraftCheckpointLoader::model() { return open(planned_->images.size() - 1); }

} // namespace splash::model
