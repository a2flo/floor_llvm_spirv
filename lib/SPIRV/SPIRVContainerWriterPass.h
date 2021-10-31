//===-SPIRVContainerWriterPass.h - SPIR-V container writing pass -- C++ -*-===//
//
//  Flo's Open libRary (floor)
//  Copyright (C) 2004 - 2021 Florian Ziesche
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
/// \file
///
/// This file provides a SPIR-V container writing pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SPIRVCONTAINERWRITERPASS_H
#define LLVM_SPIRVCONTAINERWRITERPASS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
class raw_ostream;
class PreservedAnalyses;

/// \brief Create and return a pass that writes the module to the specified
/// ostream. Note that this pass is designed for use with the legacy pass
/// manager.
ModulePass *createSPIRVContainerWriterPass(raw_ostream &Str);

/// \brief Pass for writing a module of IR out to a SPIR-V container file.
///
/// Note that this is intended for use with the new pass manager. To construct
/// a pass for the legacy pass manager, use the function above.
class SPIRVContainerWriterPass : public PassInfoMixin<SPIRVContainerWriterPass> {
  raw_ostream &OS;

public:
  /// \brief Construct a SPIRV writer pass around a particular output stream.
  explicit SPIRVContainerWriterPass(raw_ostream &OS) : OS(OS) {}

  /// \brief Run the SPIRV writer pass, and output the module to the selected
  /// output stream.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

  static StringRef name() { return "SPIRVContainerWriterPass"; }
};
}

#endif
