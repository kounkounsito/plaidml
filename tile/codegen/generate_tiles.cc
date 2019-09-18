// Copyright 2018, Intel Corporation

#include "base/util/any_factory_map.h"
#include "tile/codegen/generate_tiles.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

TensorShape MakeOddTile(const TensorShape& tile) {
  TensorShape odd_tile = tile;
  for (size_t i = 0; i < odd_tile.dims.size(); ++i) {
    if ((odd_tile.dims[i].size & 0x1) == 0) {
      ++odd_tile.dims[i].size;
    }
  }
  return odd_tile;
}

class TilePlanGenerator {

public:
  TilePlanGenerator(Block* block, const proto::GenerateTilesPass& options);
  void GeneratePlans(size_t k);
  bool IsValidPlan();
  std::string Features(Block *block);
  std::string RefFeatures(Block* block, const Refinement& ref);
  StatementList& BlockList() { return block_list_; }

private:
  Block* target_;
  const proto::GenerateTilesPass options_;
  std::set<const Index*> acc_idxs_;
  std::vector<Index*> index_;
  std::vector<size_t> plan_;
  std::string target_features_;
  StatementList block_list_;
};

TilePlanGenerator::TilePlanGenerator(Block* target, const proto::GenerateTilesPass& options):
    target_{target}, options_{options}, acc_idxs_(target->accumulation_idxs(true)) {
  for (auto& idx : target->idxs) {
    if (idx.affine == Affine()) {
      index_.push_back(&idx);
    }
  }
  plan_.resize(index_.size());
  target_features_ = Features(target_);
};

bool TilePlanGenerator::IsValidPlan() {
  std::map<std::string, size_t> tile_by_name;
  for (size_t i = 0; i < index_.size(); ++i) {
    tile_by_name.emplace(index_[i]->name, plan_[i]);
  }
  // Check memory usage of the inner block
  size_t tot_bytes = 0;
  for (const auto& ref : target_->refs) {
    auto tiled = ref.ApplyTile(tile_by_name);
    int64_t bytes = options_.odd_size() ?
      Codec::Resolve(MakeOddTile(tiled))->byte_size() : Codec::Resolve(tiled)->byte_size();  
    tot_bytes += bytes;
  }
  return tot_bytes <= options_.max_mem_size();
} 

std::string TilePlanGenerator::RefFeatures(Block* block, const Refinement& ref) {
  std::string features;
  auto access = ref.FlatAccess();
  auto acc_map = access.getMap();
  for (const auto& idx : block->idxs) {
    const auto& it = acc_map.find(idx.name);
    if (it == acc_map.end()) {
      features = features + "0 ";
    }
    else {
      features = features + std::to_string(it->second) + " ";
    }
  }
  return features;
}

// Generate the string of features
std::string TilePlanGenerator::Features(Block* block) {
  std::string ref_features;
  for (const auto& ref : block->refs) {
    ref_features = ref_features + ";" + RefFeatures(block, ref);
  }
  std::string idx_features;
  for (const auto& idx : block->idxs) {
    idx_features = idx_features + std::to_string(idx.range) + " ";
  }
  return idx_features + ref_features;
}

// Recursively generate tile plans
void TilePlanGenerator::GeneratePlans(size_t k) {
  if (k >= index_.size()) {
    // Now we have a new plan
    if (IsValidPlan()) {
      // Clone a new block first
      auto new_block = CloneBlock(*target_);
      // Tile the block according to the plan
      ApplyTile(new_block.get(), plan_, false, false, options_.interleave());
      new_block->add_tags(FromProto(options_.outer_set()));
      auto inner = new_block->SubBlock(0);
      inner->add_tags(FromProto(options_.inner_set()));
      // Encode features in the comments
      std::string tile_features = Features(inner.get());
      new_block->comments = target_features_ + "." + tile_features;
      // Add the tiled block into the program
      block_list_.push_back(new_block);
    }
  }
  else {
    size_t range = index_[k]->range;
    if (!options_.acc_idxs() && acc_idxs_.find(index_[k]) != acc_idxs_.end()) {
      // index_[k] is accumulation index
      plan_[k] = range;
      GeneratePlans(k + 1);
    }
    else {
      for (size_t i = 1; i < range; ++i) {
        if (options_.only_even()) {
          if (range % i == 0) {
            plan_[k] = i;
            GeneratePlans(k + 1);
          }
        }
        else if (options_.only_po2()) {
          if (math::IsPo2(i)) {
            plan_[k] = i;
            GeneratePlans(k + 1);
          }
        }
        else {
          plan_[k] = i;
          GeneratePlans(k + 1);
        }
      }
    }
  }
}

void GenerateTiles(const AliasMap& alias_map, Block* block, const proto::GenerateTilesPass& options) {
  if (block->name != "main") {
    return;
  }
  // The last sub-block is our target. The others are just the initialization or helper.
  auto target = block->SubBlock(0, true);
  if (!target) {
    throw std::runtime_error("Nothing to test.");
  }

  auto generator = std::make_shared<TilePlanGenerator>(target.get(), options);
  generator->GeneratePlans(0);

  // Remove the target block from the program
  block->stmts.pop_back();
  // Insert the generated tiled blocks into the program
  block->stmts.insert(block->stmts.end(), generator->BlockList().begin(), generator->BlockList().end());
}

void GenerateTilesPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                GenerateTiles(alias_map, block, options_);
              },
              true);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<GenerateTilesPass, proto::GenerateTilesPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
