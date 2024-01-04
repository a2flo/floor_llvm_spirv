//- LLVMToSPIRVDbgTranVulkan.cpp - Converts debug info to SPIR-V ---*- C++ --//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2018 Intel Corporation. All rights reserved.
// Copyright (c) 2024 Florian Ziesche
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Intel Corporation, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This implements Vulkan specific LLVM debug metadata translation to SPIR-V
//
//===----------------------------------------------------------------------===//
#include "LLVMToSPIRVDbgTranVulkan.h"
#include "SPIRVWriter.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace SPIRV;

// llvm.dbg.declare intrinsic.

SPIRVValue *LLVMToSPIRVDbgTranVulkan::createDebugDeclarePlaceholder(
    const DbgVariableIntrinsic *DbgDecl, SPIRVBasicBlock *BB) {
  return nullptr;
}

void LLVMToSPIRVDbgTranVulkan::finalizeDebugDeclare(
    const DbgVariableIntrinsic *DbgDecl) {}

// llvm.dbg.value intrinsic.

SPIRVValue *LLVMToSPIRVDbgTranVulkan::createDebugValuePlaceholder(
    const DbgVariableIntrinsic *DbgValue, SPIRVBasicBlock *BB) {
  return nullptr;
}

void LLVMToSPIRVDbgTranVulkan::finalizeDebugValue(
    const DbgVariableIntrinsic *DbgValue) {}

// Emitting DebugScope and OpLine instructions

void LLVMToSPIRVDbgTranVulkan::transLocationInfo() {
  for (const Function &F : *M) {
    for (const BasicBlock &BB : F) {
      SPIRVValue *V = SPIRVWriter->getTranslatedValue(&BB);
      assert(V && V->isBasicBlock() &&
             "Basic block is expected to be translated");
      SPIRVBasicBlock *SBB = static_cast<SPIRVBasicBlock *>(V);
      MDNode *DbgScope = nullptr;
      MDNode *InlinedAt = nullptr;
      SPIRVString *File = nullptr;
      unsigned LineNo = 0;
      unsigned Col = 0;
      for (const Instruction &I : BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
          if (II->getIntrinsicID() == Intrinsic::dbg_label) {
            // SPIR-V doesn't support llvm.dbg.label intrinsic translation
            continue;
          }
          if (II->getIntrinsicID() == Intrinsic::annotation ||
              II->getIntrinsicID() == Intrinsic::var_annotation ||
              II->getIntrinsicID() == Intrinsic::ptr_annotation) {
            // llvm call instruction for llvm .*annotation intrinsics
            // is translated into SPIR-V instruction only if it represents
            // call of __builtin_intel_fpga_reg() builtin. In other cases this
            // instruction is dropped. In these cases debug info for this call
            // should be skipped too.
            // TODO: Remove skipping of debug info when *.annotation call will
            //       be handled in a better way during SPIR-V translation.
            V = SPIRVWriter->getTranslatedValue(&I);
            if (!V || V->getOpCode() != OpFPGARegINTEL)
              continue;
          }
        }
        V = SPIRVWriter->getTranslatedValue(&I);
        if (!V || isConstantOpCode(V->getOpCode()))
          continue;
        const DebugLoc &DL = I.getDebugLoc();
        if (!DL.get()) {
          if (DbgScope || InlinedAt) { // Emit DebugNoScope
            DbgScope = nullptr;
            InlinedAt = nullptr;
            transDebugLoc(DL, SBB, static_cast<SPIRVInstruction *>(V));
          }
          continue;
        }
        // Once scope or inlining has changed emit another DebugScope
        if (DL.getScope() != DbgScope || DL.getInlinedAt() != InlinedAt) {
          DbgScope = DL.getScope();
          InlinedAt = DL.getInlinedAt();
          transDebugLoc(DL, SBB, static_cast<SPIRVInstruction *>(V));
        }
        // If any component of OpLine has changed emit another OpLine
        SPIRVString *DirAndFile = BM->getString(getFullPath(DL.get()));
        if (File != DirAndFile || LineNo != DL.getLine() ||
            Col != DL.getCol()) {
          File = DirAndFile;
          LineNo = DL.getLine();
          Col = DL.getCol();
          // According to the spec, OpLine for an
          // OpBranch/OpBranchConditional/OpSelectionMerge must precede the
          // merge instruction and not the branch instruction
          if (V->getOpCode() == OpBranch ||
              V->getOpCode() == OpBranchConditional) {
            auto *VPrev = static_cast<SPIRVInstruction *>(V)->getPrevious();
            if (VPrev && (VPrev->getOpCode() == OpLoopMerge ||
                          VPrev->getOpCode() == OpSelectionMerge ||
                          VPrev->getOpCode() == OpLoopControlINTEL)) {
              V = VPrev;
            }
          }
          BM->addLine(V, File ? File->getId() : getDebugInfoNone()->getId(),
                      LineNo, Col);
        }
      } // Instructions
    }   // Basic Blocks
  }     // Functions
}

// Translation of single debug entry

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgEntry(const MDNode *DIEntry) {
  return nullptr;
}

// Compilation unit

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgCompilationUnit(const DICompileUnit *CU) {
  BM->addModuleProcessed(SPIRVDebug::ProducerPrefix + CU->getProducer().str());
  return nullptr;
}

// Types

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgBaseType(const DIBasicType *BT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgPointerType(const DIDerivedType *PT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgQualifiedType(const DIDerivedType *QT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgArrayType(const DICompositeType *AT) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgTypeDef(const DIDerivedType *DT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgSubroutineType(const DISubroutineType *FT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgEnumType(const DICompositeType *ET) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgCompositeType(const DICompositeType *CT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgMemberType(const DIDerivedType *MT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgInheritance(const DIDerivedType *DT) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgPtrToMember(const DIDerivedType *DT) {
  return nullptr;
}

// Templates
SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgTemplateParams(DITemplateParameterArray TPA,
                                                 const SPIRVEntry *Target) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgTemplateParameter(
    const DITemplateParameter *TP) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgTemplateTemplateParameter(
    const DITemplateValueParameter *TVP) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgTemplateParameterPack(
    const DITemplateValueParameter *TVP) {
  return nullptr;
}

// Global objects

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgGlobalVariable(const DIGlobalVariable *GV) {
  return nullptr;
}

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgFunction(const DISubprogram *Func) {
  return nullptr;
}

// Location information

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgScope(const DIScope *S) {
  return nullptr;
}

// Generating DebugScope and DebugNoScope instructions. They can interleave with
// core instructions.
SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDebugLoc(
    const DebugLoc &Loc, SPIRVBasicBlock *BB, SPIRVInstruction *InsertBefore) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgInlinedAt(const DILocation *Loc) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgFileType(const DIFile *F) {
  return nullptr;
}

// Local variables

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgLocalVariable(const DILocalVariable *Var) {
  return nullptr;
}

// DWARF Operations and expressions

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgExpression(const DIExpression *Expr) {
  return nullptr;
}

// Imported entries (C++ using directive)

SPIRVEntry *
LLVMToSPIRVDbgTranVulkan::transDbgImportedEntry(const DIImportedEntity *IE) {
  return nullptr;
}

SPIRVEntry *LLVMToSPIRVDbgTranVulkan::transDbgModule(const DIModule *Module) {
  return nullptr;
}
