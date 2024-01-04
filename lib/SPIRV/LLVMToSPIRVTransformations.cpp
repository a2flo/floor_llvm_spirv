//===- LLVMToSPIRVTransformations.cpp -------------------------------------===//
//
//  Flo's Open libRary (floor)
//  Copyright (C) 2004 - 2024 Florian Ziesche
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; version 2 of the License only.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with this program; if not, write to the Free Software Foundation, Inc.,
//  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//===----------------------------------------------------------------------===//
//
// Pre-pass (prior to SPIRVWriter) that performs a few LLVM to SPIR-V
// transformations.
//
//===----------------------------------------------------------------------===//

#include "SPIRVInternal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/LibFloor/VulkanUtils.h"
#include <vector>
using namespace llvm;
using namespace SPIRV;

#define DEBUG_TYPE "LLVMToSPIRVTransformations"

#if 1
#define DBG(x)
#else
#define DBG(x) x
#endif

namespace SPIRV {
struct LLVMToSPIRVTransformations
    : public InstVisitor<LLVMToSPIRVTransformations>,
      ModulePass {
  static char ID; // Pass identification, replacement for typeid

  Module *M{nullptr};
  LLVMContext *ctx{nullptr};
  bool was_modified{false};

  // used by composite construction replacement
  // we keep this both as an unordered_set for fast lookup and in a vector,
  // because erasing must happen in reverse use order
  std::unordered_set<Instruction *> kill_set;
  std::vector<Instruction *> kill_list;

  LLVMToSPIRVTransformations() : ModulePass(ID) {
    initializeLLVMToSPIRVTransformationsPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "LLVM to SPIR-V transformations";
  }

  bool runOnModule(Module &Mod) override {
    M = &Mod;
    ctx = &M->getContext();
    was_modified = false;

    // return if not compiling Vulkan/SPIR-V
    auto Src = getSPIRVSource(M);
    if (std::get<0>(Src) != spv::SourceLanguageGLSL) {
      return false;
    }

    //
    for (auto &F : *M) {
      // only handle actual functions
      if (F.isDeclaration())
        continue;
      runOnFunction(F);
    }

    return was_modified;
  }

  void runOnFunction(Function &F) {
    kill_set.clear();
    kill_list.clear();
    visit(F);
    for (auto iter = kill_list.rbegin(); iter != kill_list.rend(); ++iter) {
      (*iter)->eraseFromParent();
    }
  }

  // InstVisitor overrides
  using InstVisitor<LLVMToSPIRVTransformations>::visit;
  void visit(Instruction &I) {
    InstVisitor<LLVMToSPIRVTransformations>::visit(I);
  }

  // vector insert chain replacement
  void visitInsertElement(InsertElementInst &I) {
    if (kill_set.count(&I) > 0) {
      return;
    }

    // sanity check
    const auto vec_type = dyn_cast_or_null<FixedVectorType>(I.getType());
    if (!vec_type) {
      return;
    }

    // only continue this if the first object to insert into is undef
    if (!isa<UndefValue>(I.getOperand(0))) {
      return;
    }

    const auto elem_type = vec_type->getElementType();
    const auto elem_count = vec_type->getNumElements();

    bool replace = true;
    std::vector<InsertElementInst *> insert_elems{&I};
    std::vector<Value *> elems{I.getOperand(1)};
    InsertElementInst *cur_insert = &I;
    for (uint32_t i = 1;; ++i) {
      // abort if the index isn't constant or is not contiguous in 0..#elems-1
      // NOTE: we do expect that LLVM has ordered these
      auto idx = dyn_cast<ConstantInt>(cur_insert->getOperand(2));
      if (idx == nullptr || idx->getZExtValue() != (i - 1)) {
        replace = false;
        break;
      }

      // done here
      if (i == elem_count) {
        break;
      }

      // abort if not exactly 1 use (this isn't a straight chain of inserts)
      if (cur_insert->getNumUses() != 1) {
        replace = false;
        break;
      }

      //
      cur_insert = dyn_cast<InsertElementInst>(*cur_insert->users().begin());
      if (cur_insert == nullptr) {
        replace = false;
        break;
      }
      insert_elems.emplace_back(cur_insert);
      elems.emplace_back(cur_insert->getOperand(1));
    }
    if (!replace)
      return;

    // we need to create a unique function name for this type
    std::string func_name =
        "floor.composite_construct.llvm." + std::to_string(elem_count) + "x";
    raw_string_ostream func_name_stream(func_name);
    elem_type->print(func_name_stream);
    func_name_stream.flush();

    // create composite construct at the last InsertElementInst + replace all
    // its uses
    std::vector<Type *> param_types(elem_count, elem_type);
    const auto func_type =
        llvm::FunctionType::get(vec_type, param_types, false);
    llvm::CallInst *CI =
        CallInst::Create(M->getOrInsertFunction(func_name, func_type), elems,
                         (I.hasName() ? I.getName() + ".composite_construct"
                                      : "composite_construct"),
                         cur_insert);
    CI->setCallingConv(CallingConv::FLOOR_FUNC);
    CI->setDebugLoc(I.getDebugLoc()); // keep debug loc of first insert
    cur_insert->replaceAllUsesWith(CI);

    // add all replaced InsertElementInsts to the kill list
    for (auto &instr : insert_elems) {
      kill_list.emplace_back(instr);
      kill_set.emplace(instr);
    }

    was_modified = true;
  }

  // aggregate insert chain replacement
  void visitInsertValue(InsertValueInst &) {
    // TODO: implement this
  }

  // vector shuffle with undef replacement
  void visitShuffleVector(ShuffleVectorInst &I) {
    if (isa<UndefValue>(I.getOperand(1))) {
      I.setOperand(1, I.getOperand(0));
      was_modified = true;
      return;
    }
    if (isa<UndefValue>(I.getOperand(0))) {
      I.setOperand(0, I.getOperand(1));
      was_modified = true;
      return;
    }
  }

  //! returns a constant zero val for the specified type
  //! NOTE: validation must have already happened
  static Constant *make_zero_val(llvm::Type *val_type) {
    if (val_type->isVectorTy()) {
      auto vec_type = dyn_cast_or_null<FixedVectorType>(val_type);
      auto vec_elem_type = vec_type->getElementType();
      return ConstantVector::getSplat(
          vec_type->getElementCount(),
          vec_elem_type->isIntegerTy() ? ConstantInt::get(vec_elem_type, 0)
                                       : ConstantFP::get(vec_elem_type, 0.0));
    } else if (val_type->isIntegerTy()) {
      return ConstantInt::get(val_type, 0);
    } else if (val_type->isFloatingPointTy()) {
      return ConstantFP::get(val_type, 0.0);
    } else if (val_type->isPointerTy()) {
      auto phi_ptr_type = dyn_cast_or_null<PointerType>(val_type);
      assert(phi_ptr_type && "invalid ptr type");
      return ConstantPointerNull::get(phi_ptr_type);
    }
    llvm_unreachable("invalid phi type");
  }
  // don't allow undef values in PHIs
  void visitPHINode(PHINode &phi) {
    auto val_type = phi.getType();
    if (val_type->isVectorTy()) {
      auto vec_type = dyn_cast_or_null<FixedVectorType>(val_type);
      if (!vec_type || (!vec_type->getElementType()->isIntegerTy() &&
                        !vec_type->getElementType()->isFloatingPointTy())) {
        return; // can't handle this
      }
    } else if (!val_type->isIntegerTy() &&
               !val_type->isFloatingPointTy() &&
               !val_type->isPointerTy()) {
      return; // can't handle this
    }
    for (uint32_t in_idx = 0, in_count = phi.getNumIncomingValues();
         in_idx < in_count; ++in_idx) {
      auto in_val = phi.getIncomingValue(in_idx);
      if (auto undef = dyn_cast_or_null<UndefValue>(in_val)) {
        phi.setIncomingValue(in_idx, make_zero_val(val_type));
        was_modified = true;
      }
    }
  }

  void visitCallInst(CallInst &CI) {
    // remove placeholder function calls
    if (CI.getCalledFunction()->getName().startswith("floor.merge_block") ||
        CI.getCalledFunction()->getName().startswith("floor.continue_block") ||
        CI.getCalledFunction()->getName().startswith("floor.keep_block")) {
      CI.eraseFromParent();
      was_modified = true;
    }
#if 0
    // remove unnecessary selection merge calls (branch that is no longer conditional)
    if (CI.getCalledFunction()->getName() == "floor.selection_merge") {
      auto term = CI.getParent()->getTerminator();
      if (auto br = dyn_cast_or_null<BranchInst>(term); br && !br->isConditional()) {
        CI.eraseFromParent();
        was_modified = true;
      }
    }
#endif
  }

  void visitGetElementPtrInst(GetElementPtrInst &I) {
    was_modified |= vulkan_utils::simplify_gep_indices(*ctx, I);
  }
};

char LLVMToSPIRVTransformations::ID = 0;
} // namespace SPIRV

ModulePass *llvm::createLLVMToSPIRVTransformations() {
  return new LLVMToSPIRVTransformations();
}
INITIALIZE_PASS(LLVMToSPIRVTransformations, "LLVM to SPIR-V transformations",
                "LLVM to SPIR-V transformations", false, false)
