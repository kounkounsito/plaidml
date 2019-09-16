// Copyright 2018, Intel Corporation

#include "base/util/any_factory_map.h"
#include "tile/codegen/cache_ref.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

namespace {
  stripe::Block* largest_cache_block;
  size_t largest_cache_size;
}

// Recursively find the cache block with the largest refinement
void LargestRefBlock(Block* block) {
  if (block->has_tag("cache")) {
    size_t total_size = 0;
    for (const auto& ref : block->refs) {
      if (ref.is_global()) {
        total_size += block->exterior_shape(ref.into()).sizes_product();
      }
    }
    if (total_size > largest_cache_size) {
      largest_cache_size = total_size;
      largest_cache_block = block;
    }
  }
  else {
    for (const auto& stmt : block->stmts) {
      auto sub = Block::Downcast(stmt);
      if (sub) {
        LargestRefBlock(sub.get());
      }
    }
  }
}

void CacheReferencePass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [&](const AliasMap& alias_map, stripe::Block* block) {  //
                // Mark the cache block with the largest refinement as "ref_tile"
                largest_cache_size = 0;
                largest_cache_block = nullptr;
                LargestRefBlock(block);
                if (largest_cache_block) {
                  largest_cache_block->add_tags(FromProto(options_.tags()));
                }
              },
              true);  
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<CacheReferencePass, proto::CacheReferencePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
