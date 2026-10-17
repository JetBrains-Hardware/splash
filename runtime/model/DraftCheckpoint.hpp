#pragma once

#include "model/GgufImage.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

struct DFlashDraftLayout;

// A DFlash2 checkpoint, config.json and safetensors as its repository
// releases it (BF16) or as an MLX conversion quantized it -> the draft's
// block images (DFlashDraft.cpp), laid out as a target's are
// (model/GgufImage.hpp): every projection quantized into af4g64 planes as
// MLX's affine quantization rounds it, or retiled into the planes of the
// format MLX stored it in; the selector codebooks in bf16, as stored or
// decoded from their quantization and rounded once; every other tensor
// copied as stored. The checkpoint is planned once; each image is written
// into memory when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout);
  DraftCheckpointLoader(const DraftCheckpointLoader &) = delete;
  DraftCheckpointLoader &operator=(const DraftCheckpointLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // layers, then model.bin
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

// The images of a draft checkpoint, in the order DFlashDraft.cpp reads them:
// the layers, then model.bin. The checkpoint outlives them: their rows read
// its tensors.
[[nodiscard]] std::vector<gguf::Image> planDraftImages(const SafetensorsCheckpoint &checkpoint,
                                                       const DFlashDraftLayout &layout);

// The bytes of every image of the draft checkpoint in directory.
[[nodiscard]] uint64_t draftImageBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout);

} // namespace splash::model
