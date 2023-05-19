//- LLVMToSPIRVDbgTranVulkan.h - Converts LLVM DebugInfo to SPIR-V -*- C++ --//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2018 Intel Corporation. All rights reserved.
// Copyright (c) 2023 Florian Ziesche
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

#ifndef LLVMTOSPIRVDBGTRANVULKAN_HPP_
#define LLVMTOSPIRVDBGTRANVULKAN_HPP_

#include "SPIRVModule.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "LLVMToSPIRVDbgTran.h"

using namespace llvm;

namespace SPIRV {
class LLVMToSPIRVBase;

class LLVMToSPIRVDbgTranVulkan : public LLVMToSPIRVDbgTran {
public:
  LLVMToSPIRVDbgTranVulkan(Module *TM = nullptr, SPIRVModule *TBM = nullptr,
                           LLVMToSPIRVBase *Writer = nullptr)
      : LLVMToSPIRVDbgTran(TM, TBM, Writer) {}
  ~LLVMToSPIRVDbgTranVulkan() override = default;

  SPIRVValue *createDebugDeclarePlaceholder(const DbgVariableIntrinsic *DbgDecl,
                                            SPIRVBasicBlock *BB) override;
  SPIRVValue *createDebugValuePlaceholder(const DbgVariableIntrinsic *DbgValue,
                                          SPIRVBasicBlock *BB) override;

protected:
  // 2. After translation of all regular instructions we deal with debug info.
  //   We iterate over debug intrinsics stored on the first step, get its mapped
  //   SPIRV instruction and tweak the operands.
  void finalizeDebugDeclare(const DbgVariableIntrinsic *DbgDecl) override;
  void finalizeDebugValue(const DbgVariableIntrinsic *DbgValue) override;

  // Emit DebugScope and OpLine instructions
  void transLocationInfo() override;

  // Dispatcher
  SPIRVEntry *transDbgEntry(const MDNode *DIEntry) override;
  // SPIRVEntry *transDbgEntryImpl(const MDNode *MDN) override;

  // Compilation unit
  SPIRVEntry *transDbgCompilationUnit(const DICompileUnit *CU) override;

  /// The following methods (till the end of the file) implement translation
  /// of debug instrtuctions described in the spec.

  // Types
  SPIRVEntry *transDbgBaseType(const DIBasicType *BT) override;
  SPIRVEntry *transDbgPointerType(const DIDerivedType *PT) override;
  SPIRVEntry *transDbgQualifiedType(const DIDerivedType *QT) override;
  SPIRVEntry *transDbgArrayType(const DICompositeType *AT) override;
  SPIRVEntry *transDbgTypeDef(const DIDerivedType *D) override;
  SPIRVEntry *transDbgSubroutineType(const DISubroutineType *FT) override;
  SPIRVEntry *transDbgEnumType(const DICompositeType *ET) override;
  SPIRVEntry *transDbgCompositeType(const DICompositeType *CT) override;
  SPIRVEntry *transDbgMemberType(const DIDerivedType *MT) override;
  SPIRVEntry *transDbgInheritance(const DIDerivedType *DT) override;
  SPIRVEntry *transDbgPtrToMember(const DIDerivedType *DT) override;

  // Templates
  SPIRVEntry *transDbgTemplateParams(DITemplateParameterArray TPA,
                                     const SPIRVEntry *Target) override;
  SPIRVEntry *transDbgTemplateParameter(const DITemplateParameter *TP) override;
  SPIRVEntry *transDbgTemplateTemplateParameter(
      const DITemplateValueParameter *TP) override;
  SPIRVEntry *
  transDbgTemplateParameterPack(const DITemplateValueParameter *TP) override;

  // Global objects
  SPIRVEntry *transDbgGlobalVariable(const DIGlobalVariable *GV) override;
  SPIRVEntry *transDbgFunction(const DISubprogram *Func) override;

  // Location information
  SPIRVEntry *transDbgScope(const DIScope *S) override;
  SPIRVEntry *transDebugLoc(const DebugLoc &Loc, SPIRVBasicBlock *BB,
                            SPIRVInstruction *InsertBefore = nullptr) override;
  SPIRVEntry *transDbgInlinedAt(const DILocation *D) override;

  SPIRVEntry *transDbgFileType(const DIFile *F) override;

  // Local Variables
  SPIRVEntry *transDbgLocalVariable(const DILocalVariable *Var) override;

  // DWARF expressions
  SPIRVEntry *transDbgExpression(const DIExpression *Expr) override;

  // Imported declarations and modules
  SPIRVEntry *transDbgImportedEntry(const DIImportedEntity *IE) override;

  // A module in programming language. Example - Fortran module, clang module.
  SPIRVEntry *transDbgModule(const DIModule *IE) override;

}; // class LLVMToSPIRVDbgTranVulkan

} // namespace SPIRV

#endif // LLVMTOSPIRVDBGTRANVULKAN_HPP_
