// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void CacheReferenceCache(const AliasMap& alias_map, stripe::Block* block, const proto::CacheReferencePass& options);

class CacheReferencePass final : public CompilePass {
 public:
  explicit CacheReferencePass(const proto::CacheReferencePass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::CacheReferencePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
