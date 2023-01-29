//===- SPIRVContainerWriterPass.cpp - SPIRV writing pass ------------------===//
//
//  Flo's Open libRary (floor)
//  Copyright (C) 2004 - 2023 Florian Ziesche
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
// SPIRVContainerWriterPass implementation: this is used to split up (clone) a
// LLVM module into individual per-entry-point modules, which is necessary for
// SPIR-V shaders, because we can't generally have globals that use the same
// descriptor set index and binding index with different layouts, even if an
// entry point only makes specific use of a valid set of descriptors.
// This will then combine all SPIR-V binaries/modules into a single container
// file, with some additional helpful metadata (function names and types per
// SPIR-V module).
//
// TODO: in the future, it might make sense to specify a specific pipeline in
// the source (set of shader functions), for which we can then guarantee (or
// enforce) that only a valid set of descriptors is being used. This would then
// allow these shaders to be emitted in one single SPIR-V module.
//
//
// #### SPIR-V container file format ####
// ## header
// char[4]: identifier "SPVC"
// uint32_t: version (currently 2)
// uint32_t: entry_count
//
// ## header entries [entry_count]
// uint32_t: function_entry_count
// uint32_t: SPIR-V module word count (word == uint32_t)
//
// ## module entries [entry_count]
// uint32_t[header_entry[i].word_count]: SPIR-V module
//
// ## additional metadata [entry_count]
// uint32_t[function_entry_count]: function types
// char[function_entry_count][]: function names (always \0 terminated, with \0
//                                               padding to achieve
//                                               4-byte/uint32_t alignment)
//
// ####
//
// function type (enum):
//  * compute/kernel: 1
//  * vertex: 2
//  * fragment: 3
//  * tessellation control: 4
//  * tessellation evaluation: 5
//
//===----------------------------------------------------------------------===//

#include "SPIRVContainerWriterPass.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "LLVMSPIRVLib.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <string>
#include <unordered_set>
using namespace llvm;

static bool is_used_in_function(const Function *F, const GlobalVariable *GV) {
  // always flag certain builtin constants as used
  switch (F->getCallingConv()) {
  case CallingConv::FLOOR_KERNEL:
    if (GV->getName().find(".vulkan_constant.local_size") !=
        StringRef::npos)
      return true;
    break;
  case CallingConv::FLOOR_VERTEX:
  case CallingConv::FLOOR_FRAGMENT:
  case CallingConv::FLOOR_TESS_CONTROL:
  case CallingConv::FLOOR_TESS_EVAL:
    break;
  }

  for (const auto &user : GV->users()) {
    if (const auto instr = dyn_cast<Instruction>(user)) {
      if (instr->getParent()->getParent() == F) {
        return true;
      }
    }
  }
  return false;
}

static bool write_container(Module &M, raw_ostream &OS) {
  bool success = true;

  // header
  static constexpr const uint32_t container_version{2u};
  OS.write("SPVC", 4);
  OS.write((const char *)&container_version, sizeof(container_version));

  // gather entry point functions that we want to clone/emit
  std::unordered_set<const Function *> clone_functions;
  for (const auto &F : M) {
    if (F.getCallingConv() != CallingConv::FLOOR_KERNEL &&
        F.getCallingConv() != CallingConv::FLOOR_VERTEX &&
        F.getCallingConv() != CallingConv::FLOOR_FRAGMENT &&
        F.getCallingConv() != CallingConv::FLOOR_TESS_CONTROL &&
        F.getCallingConv() != CallingConv::FLOOR_TESS_EVAL) {
      continue;
    }
    clone_functions.emplace(&F);
  }
  // entry count
  const auto entry_count = uint32_t(clone_functions.size());
  OS.write((const char *)&entry_count, sizeof(entry_count));

  // we need a separate stream for the actual spir-v data, since we need to know
  // the size of each spir-v module/file (no way to know this beforehand)
  std::string spirv_data{""};
  raw_string_ostream spirv_stream(spirv_data);

  for (const auto &func : clone_functions) {
    // clone the module with the current entry point function and any global
    // vars that we need
    ValueToValueMapTy VMap;
    auto cloned_mod = CloneModule(M, VMap, [&func](const GlobalValue *GV) {
      if (GV == func) {
        return true;
      }
      // only clone global vars if they are needed in a specific function
      if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
        return is_used_in_function(func, GVar);
      }
      return false;
    });

    // function entry count
    const uint32_t function_entry_count = 1u;
    OS.write((const char *)&function_entry_count, sizeof(function_entry_count));

    // spir-v binary
    const auto cur_pos = spirv_stream.tell();
    std::string err;
    // TODO: only enable extensions that are generally supported (needs host enablement as well)
    SPIRV::TranslatorOpts::ExtensionsStatusMap exts;
    exts[SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_add] = true;
    exts[SPIRV::ExtensionID::SPV_KHR_fragment_shader_barycentric] = true;
    //exts[SPIRV::ExtensionID::SPV_KHR_no_integer_wrap_decoration] = true;
    //exts[SPIRV::ExtensionID::SPV_KHR_float_controls] = true;
    SPIRV::TranslatorOpts opts(SPIRV::VersionNumber::MaximumVersion, exts);
    const auto module_success = writeSpirv(cloned_mod.get(), opts, spirv_stream, err);
    auto module_size = uint32_t(spirv_stream.tell() - cur_pos);
    if (module_size % 4 != 0) {
      success = false;
      errs() << "SPIR-V data size is not a multiple of 4\n";
    }

    // write the SPIR-V data word count
    module_size /= 4;
    OS.write((const char *)&module_size, sizeof(module_size));

    // emit error if unsuccessful (still continue though)
    success &= module_success;
    if (!module_success) {
      errs() << "failed to write/translate module for \"" << func->getName()
             << "\": " << err << "\n";
    }
  }

  // all header entries written -> write actual spir-v data
  spirv_stream.flush();
  OS.write(spirv_data.c_str(), spirv_data.size());

  // write per-module metadata
  for (const auto &func : clone_functions) {
    // function types
    uint32_t function_type = 0;
    switch (func->getCallingConv()) {
    case CallingConv::FLOOR_KERNEL:
      function_type = 1;
      break;
    case CallingConv::FLOOR_VERTEX:
      function_type = 2;
      break;
    case CallingConv::FLOOR_FRAGMENT:
      function_type = 3;
      break;
    case CallingConv::FLOOR_TESS_CONTROL:
      function_type = 4;
      break;
    case CallingConv::FLOOR_TESS_EVAL:
      function_type = 5;
      break;
    default:
      llvm_unreachable("invalid function type");
    }
    OS.write((const char *)&function_type, sizeof(function_type));

    // function names
    const auto name = func->getName().str();
    const auto name_len = (uint32_t)name.size();
    const auto name_padding = 4u - (name_len % 4u);
    OS << name.c_str();
    switch (name_padding) {
    case 4:
      OS << '\0';
    LLVM_FALLTHROUGH; case 3:
      OS << '\0';
    LLVM_FALLTHROUGH; case 2:
      OS << '\0';
    LLVM_FALLTHROUGH; case 1:
      OS << '\0';
      break;
    default:
      llvm_unreachable("bad math");
    }
  }

  return success;
}

PreservedAnalyses SPIRVContainerWriterPass::run(Module &M, ModuleAnalysisManager &) {
  write_container(M, OS);
  return PreservedAnalyses::all();
}

namespace {
class WriteSPIRVContainerPass : public ModulePass {
  raw_ostream &OS; // raw_ostream to print on
public:
  static char ID; // Pass identification, replacement for typeid
  explicit WriteSPIRVContainerPass(raw_ostream &o) : ModulePass(ID), OS(o) {}

  StringRef getPassName() const override { return "SPIR-V Container Writer"; }

  bool runOnModule(Module &M) override {
    write_container(M, OS);
    return false;
  }
};
}

char WriteSPIRVContainerPass::ID = 0;

ModulePass *llvm::createSPIRVContainerWriterPass(raw_ostream &Str) {
  return new WriteSPIRVContainerPass(Str);
}
