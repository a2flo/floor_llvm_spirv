//===- SPIRVWriter.cpp - Converts LLVM to SPIR-V ----------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2016 - 2024 Florian Ziesche Vulkan/SPIR-V support
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
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
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
/// \file
///
/// This file implements conversion of LLVM intermediate language to SPIR-V
/// binary.
///
//===----------------------------------------------------------------------===//

#include "SPIRVWriter.h"
#include "LLVMToSPIRVDbgTran.h"
#include "LLVMToSPIRVDbgTranVulkan.h"
#include "SPIRVAsm.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVLLVMUtil.h"
#include "SPIRVMDWalker.h"
#include "SPIRVMemAliasingINTEL.h"
#include "SPIRVModule.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"
#include "VectorComputeUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/LibFloor/VulkanSampling.h"
#include "llvm/Transforms/LibFloor/FloorUtils.h"
#include "llvm/Transforms/Utils.h" // loop-simplify pass

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <regex>
#include <set>
#include <vector>

#define DEBUG_TYPE "spirv"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

static void foreachKernelArgMD(
    MDNode *MD, SPIRVFunction *BF,
    std::function<void(const std::string &Str, SPIRVFunctionParameter *BA)>
        Func) {
  for (unsigned I = 0, E = MD->getNumOperands(); I != E; ++I) {
    SPIRVFunctionParameter *BA = BF->getArgument(I);
    Func(getMDOperandAsString(MD, I), BA);
  }
}

static SPIRVMemoryModelKind getMemoryModel(Module &M) {
  auto *MemoryModelMD = M.getNamedMetadata(kSPIRVMD::MemoryModel);
  if (MemoryModelMD && (MemoryModelMD->getNumOperands() > 0)) {
    auto *Ref0 = MemoryModelMD->getOperand(0);
    if (Ref0 && Ref0->getNumOperands() > 1) {
      auto &&ModelOp = Ref0->getOperand(1);
      auto *ModelCI = mdconst::dyn_extract<ConstantInt>(ModelOp);
      if (ModelCI && (ModelCI->getValue().getActiveBits() <= 64)) {
        auto Model = static_cast<SPIRVMemoryModelKind>(ModelCI->getZExtValue());
        return Model;
      }
    }
  }
  return SPIRVMemoryModelKind::MemoryModelMax;
}

static void translateSEVDecoration(Attribute Sev, SPIRVValue *Val) {
  assert(Sev.isStringAttribute() &&
         Sev.getKindAsString() == kVCMetadata::VCSingleElementVector);

  auto *Ty = Val->getType();
  assert((Ty->isTypeBool() || Ty->isTypeFloat() || Ty->isTypeInt() ||
          Ty->isTypePointer()) &&
         "This decoration is valid only for Scalar or Pointer types");

  if (Ty->isTypePointer()) {
    SPIRVWord IndirectLevelsOnElement = 0;
    Sev.getValueAsString().getAsInteger(0, IndirectLevelsOnElement);
    Val->addDecorate(DecorationSingleElementVectorINTEL,
                     IndirectLevelsOnElement);
  } else
    Val->addDecorate(DecorationSingleElementVectorINTEL);
}

LLVMToSPIRVBase::LLVMToSPIRVBase(SPIRVModule *SMod)
    : M(nullptr), Ctx(nullptr), BM(SMod), SrcLang(0), SrcLangVer(0) {
  if (SMod->getDebugInfoEIS() == SPIRVEIS_GLSL) {
    DbgTran = std::make_unique<LLVMToSPIRVDbgTranVulkan>(nullptr, SMod, this);
  } else {
    DbgTran = std::make_unique<LLVMToSPIRVDbgTran>(nullptr, SMod, this);
  }
}

LLVMToSPIRVBase::~LLVMToSPIRVBase() {
  for (auto *I : UnboundInst)
    I->deleteValue();
}

bool LLVMToSPIRVBase::runLLVMToSPIRV(Module &Mod) {
  M = &Mod;
  CG = std::make_unique<CallGraph>(Mod);
  Ctx = &M->getContext();
  DbgTran->setModule(M);
  assert(BM && "SPIR-V module not initialized");
  translate();
  return true;
}

SPIRVValue *LLVMToSPIRVBase::getTranslatedValue(const Value *V) const {
  auto Loc = ValueMap.find(V);
  if (Loc != ValueMap.end())
    return Loc->second;
  return nullptr;
}

bool LLVMToSPIRVBase::isEntryPoint(Function *F) {
  if (F->getCallingConv() == CallingConv::FLOOR_KERNEL ||
      F->getCallingConv() == CallingConv::FLOOR_VERTEX ||
      F->getCallingConv() == CallingConv::FLOOR_FRAGMENT ||
      F->getCallingConv() == CallingConv::FLOOR_TESS_CONTROL ||
      F->getCallingConv() == CallingConv::FLOOR_TESS_EVAL)
    return true;
  return false;
}

spv::ExecutionModel LLVMToSPIRVBase::getEntryPointType(Function *F,
                                                       unsigned int SrcLang) {
  switch (F->getCallingConv()) {
  case CallingConv::FLOOR_KERNEL:
    return (SrcLang == spv::SourceLanguageGLSL
                ? spv::ExecutionModel::ExecutionModelGLCompute
                : spv::ExecutionModel::ExecutionModelKernel);
  case CallingConv::FLOOR_VERTEX:
    return spv::ExecutionModel::ExecutionModelVertex;
  case CallingConv::FLOOR_FRAGMENT:
    return spv::ExecutionModel::ExecutionModelFragment;
  case CallingConv::FLOOR_TESS_CONTROL:
    return spv::ExecutionModel::ExecutionModelTessellationControl;
  case CallingConv::FLOOR_TESS_EVAL:
    return spv::ExecutionModel::ExecutionModelTessellationEvaluation;
  default:
    return spv::ExecutionModel::ExecutionModelInvalid;
  }
}

bool LLVMToSPIRVBase::isBuiltinTransToInst(Function *F) {
  StringRef DemangledName;
  if (!oclIsBuiltin(F->getName(), DemangledName) &&
      !isDecoratedSPIRVFunc(F, DemangledName))
    return false;
  SPIRVDBG(spvdbgs() << "CallInst: demangled name: " << DemangledName.str()
                     << '\n');
  return getSPIRVFuncOC(DemangledName) != OpNop;
}

bool LLVMToSPIRVBase::isBuiltinTransToExtInst(
    Function *F, SPIRVExtInstSetKind *ExtSet, SPIRVWord *ExtOp,
    SmallVectorImpl<std::string> *Dec) {
  StringRef DemangledName;
  if (!oclIsBuiltin(F->getName(), DemangledName))
    return false;
  LLVM_DEBUG(dbgs() << "[oclIsBuiltinTransToExtInst] CallInst: demangled name: "
                    << DemangledName << '\n');
  StringRef S = DemangledName;
  if (!S.startswith(kSPIRVName::Prefix))
    return false;
  S = S.drop_front(strlen(kSPIRVName::Prefix));
  auto Loc = S.find(kSPIRVPostfix::Divider);
  auto ExtSetName = S.substr(0, Loc);
  SPIRVExtInstSetKind Set = SPIRVEIS_Count;
  if (!SPIRVExtSetShortNameMap::rfind(ExtSetName.str(), &Set))
    return false;
  assert((Set == SPIRVEIS_OpenCL || Set == BM->getDebugInfoEIS() ||
          Set == SPIRVEIS_GLSL) &&
         "Unsupported extended instruction set");

  auto ExtOpName = S.substr(Loc + 1);
  auto Splited = ExtOpName.split(kSPIRVPostfix::ExtDivider);
  if (Set == SPIRVEIS_OpenCL) {
    OCLExtOpKind EOC;
    if (!OCLExtOpMap::rfind(Splited.first.str(), &EOC))
      return false;

    if (ExtSet)
      *ExtSet = Set;
    if (ExtOp)
      *ExtOp = EOC;
  } else if (Set == SPIRVEIS_GLSL) {
    GLSLExtOpKind EGLSL;
    if (!GLSLExtOpMap::rfind(Splited.first.str(), &EGLSL))
      return false;

    if (ExtSet)
      *ExtSet = Set;
    if (ExtOp)
      *ExtOp = EGLSL;
  } else {
    llvm_unreachable("unhandled instruction set");
  }

  if (Dec) {
    SmallVector<StringRef, 2> P;
    Splited.second.split(P, kSPIRVPostfix::Divider);
    for (auto &I : P)
      Dec->push_back(I.str());
  }
  return true;
}

static bool recursiveType(const StructType *ST, const Type *Ty) {
  SmallPtrSet<const StructType *, 4> Seen;

  std::function<bool(const Type *Ty)> Run = [&](const Type *Ty) {
    if (auto *StructTy = dyn_cast<StructType>(Ty)) {
      if (StructTy == ST)
        return true;

      if (Seen.count(StructTy))
        return false;

      Seen.insert(StructTy);

      return find_if(StructTy->element_begin(), StructTy->element_end(), Run) !=
             StructTy->element_end();
    }

    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      Type *ElTy = PtrTy->getPointerElementType();
      if (auto *FTy = dyn_cast<FunctionType>(ElTy)) {
        // If we have a function pointer, then argument types and return type of
        // the referenced function also need to be checked
        return Run(FTy->getReturnType()) ||
               any_of(FTy->param_begin(), FTy->param_end(), Run);
      }

      return Run(ElTy);
    }

    if (auto *ArrayTy = dyn_cast<ArrayType>(Ty))
      return Run(ArrayTy->getArrayElementType());

    return false;
  };

  return Run(Ty);
}

SPIRVType *LLVMToSPIRVBase::transType(Type *T) {
  LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
    return Loc->second;

  SPIRVDBG(dbgs() << "[transType] " << *T << '\n');
  if (T->isVoidTy())
    return mapType(T, BM->addVoidType());

  if (T->isIntegerTy(1))
    return mapType(T, BM->addBoolType());

  if (T->isIntegerTy()) {
    unsigned BitWidth = T->getIntegerBitWidth();
    if (SrcLang == spv::SourceLanguageGLSL) {
      // legalize int width
      if (BitWidth <= 8) {
        BitWidth = 8;
      } else if (BitWidth <= 16) {
        BitWidth = 16;
      } else if (BitWidth <= 32) {
        BitWidth = 32;
      } else if (BitWidth <= 64) {
        BitWidth = 64;
      } else {
        assert(false && "bit-width is not supported (too large)");
      }
      // always signed (by default)
      return mapType(T, BM->addIntegerType(BitWidth, true));
    } else { // OpenCL, or others
      // SPIR-V 2.16.1. Universal Validation Rules: Scalar integer types can be
      // parameterized only as 32 bit, plus any additional sizes enabled by
      // capabilities.
      if (BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_arbitrary_precision_integers) ||
          BM->getErrorLog().checkError(BitWidth == 8 || BitWidth == 16 ||
                                           BitWidth == 32 || BitWidth == 64,
                                       SPIRVEC_InvalidBitWidth,
                                       std::to_string(BitWidth))) {
        // always unsigned
        return mapType(T, BM->addIntegerType(BitWidth, false));
      }
    }
  }

  if (T->isFloatingPointTy())
    return mapType(T, BM->addFloatType(T->getPrimitiveSizeInBits()));

  if (T->isTokenTy()) {
    BM->getErrorLog().checkError(
        BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_token_type),
        SPIRVEC_RequiresExtension,
        "SPV_INTEL_token_type\n"
        "NOTE: LLVM module contains token type, which doesn't have analogs in "
        "SPIR-V without extensions");
    return mapType(T, BM->addTokenTypeINTEL());
  }

  // A pointer to image or pipe type in LLVM is translated to a SPIRV
  // (non-pointer) image or pipe type.
  if (T->isPointerTy()) {
    auto ET = T->getPointerElementType();
    if (ET->isFunctionTy() &&
        !BM->checkExtension(ExtensionID::SPV_INTEL_function_pointers,
                            SPIRVEC_FunctionPointers, toString(T)))
      return nullptr;
    auto ST = dyn_cast<StructType>(ET);
    auto AddrSpc = T->getPointerAddressSpace();
    // Lower global_device and global_host address spaces that were added in
    // SYCL as part of SYCL_INTEL_usm_address_spaces extension to just global
    // address space if device doesn't support SPV_INTEL_usm_storage_classes
    // extension
    if (!BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_usm_storage_classes) &&
        ((AddrSpc == SPIRAS_GlobalDevice) || (AddrSpc == SPIRAS_GlobalHost))) {
      auto NewType =
          PointerType::get(T->getPointerElementType(), SPIRAS_Global);
      return mapType(T, transType(NewType));
    }
    if (ST && !ST->isSized()) {
      Op OpCode;
      StringRef STName = ST->getName();
      // Workaround for non-conformant SPIR binary
      if (STName == "struct._event_t") {
        STName = kSPR2TypeName::Event;
        ST->setName(STName);
      }
      if (STName.startswith(kSPR2TypeName::PipeRO) ||
          STName.startswith(kSPR2TypeName::PipeWO)) {
        auto PipeT = BM->addPipeType();
        PipeT->setPipeAcessQualifier(STName.startswith(kSPR2TypeName::PipeRO)
                                         ? AccessQualifierReadOnly
                                         : AccessQualifierWriteOnly);
        return mapType(T, PipeT);
      }
      if (STName.startswith(kSPR2TypeName::ImagePrefix)) {
        if (SrcLang != SourceLanguageGLSL) {
          assert(AddrSpc == SPIRAS_Global);
          auto SPIRVImageTy = getSPIRVImageTypeFromOCL(M, T);
          return mapType(T, transType(SPIRVImageTy));
        } else {
          errs() << "invalid trans type: " << *T << "\n";
          assert(false && "should not be here");
        }
      }
      if (STName == kSPR2TypeName::Sampler)
        return mapType(T, transType(getSamplerType(M)));
      if (STName.startswith(kSPIRVTypeName::PrefixAndDelim))
        return transSPIRVOpaqueType(T);

      if (STName.startswith(kOCLSubgroupsAVCIntel::TypePrefix))
        return mapType(
            T, BM->addSubgroupAvcINTELType(
                   OCLSubgroupINTELTypeOpCodeMap::map(ST->getName().str())));

      if (OCLOpaqueTypeOpCodeMap::find(STName.str(), &OpCode)) {
        switch (OpCode) {
        default:
          return mapType(T, BM->addOpaqueGenericType(OpCode));
        case OpTypeDeviceEvent:
          return mapType(T, BM->addDeviceEventType());
        case OpTypeQueue:
          return mapType(T, BM->addQueueType());
        }
      }
      if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute)) {
        if (STName.startswith(kVCType::VCBufferSurface)) {
          // VCBufferSurface always have Access Qualifier
          auto Access = getAccessQualifier(STName);
          return mapType(T, BM->addBufferSurfaceINTELType(Access));
        }
      }

      if (isPointerToOpaqueStructType(T)) {
        return mapType(
            T, BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
                                      static_cast<SPIRAddressSpace>(AddrSpc)),
                                  transType(ET)));
      }
    } else {
      SPIRVType *ElementType = transType(ET);
      // ET, as a recursive type, may contain exactly the same pointer T, so it
      // may happen that after translation of ET we already have translated T,
      // added the translated pointer to the SPIR-V module and mapped T to this
      // pointer. Now we have to check TypeMap again.
      LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
      if (Loc != TypeMap.end()) {
        return Loc->second;
      }
      return mapType(
          T, BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
                                    static_cast<SPIRAddressSpace>(AddrSpc)),
                                ElementType));
    }
  }

  if (auto *VecTy = dyn_cast<FixedVectorType>(T))
    return mapType(T, BM->addVectorType(transType(VecTy->getElementType()),
                                        VecTy->getNumElements()));

  if (T->isArrayTy()) {
    // SPIR-V 1.3 s3.32.6: Length is the number of elements in the array.
    //                     It must be at least 1.
    if (T->getArrayNumElements() < 1) {
      std::string Str;
      llvm::raw_string_ostream OS(Str);
      OS << *T;
      SPIRVCK(T->getArrayNumElements() >= 1, InvalidArraySize, OS.str());
    }
    return mapType(T, BM->addArrayType(
                          transType(T->getArrayElementType()),
                          static_cast<SPIRVConstant *>(transValue(
                              ConstantInt::get(getSizetType(),
                                               T->getArrayNumElements(), false),
                              nullptr))));
  }

  if (T->isStructTy() && !T->isSized()) {
    auto ST = dyn_cast<StructType>(T);
    (void)ST; // Silence warning
    assert(!ST->getName().startswith(kSPR2TypeName::PipeRO));
    assert(!ST->getName().startswith(kSPR2TypeName::PipeWO));
    assert(!ST->getName().startswith(kSPR2TypeName::ImagePrefix));
    return mapType(T, BM->addOpaqueType(T->getStructName().str()));
  }

  if (auto ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    StringRef Name;
    if (ST->hasName())
      Name = ST->getName();

    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
      return transType(getSamplerType(M));
    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
      return transType(getPipeStorageType(M));

    constexpr size_t MaxNumElements = MaxWordCount - SPIRVTypeStruct::FixedWC;
    const size_t NumElements = ST->getNumElements();
    size_t SPIRVStructNumElements = NumElements;
    // In case number of elements is greater than maximum WordCount and
    // SPV_INTEL_long_constant_composite is not enabled, the error will be
    // emitted by validate functionality of SPIRVTypeStruct class.
    if (NumElements > MaxNumElements &&
        BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_long_constant_composite)) {
      SPIRVStructNumElements = MaxNumElements;
    }

    auto *Struct = BM->openStructType(SPIRVStructNumElements, Name.str());
    mapType(T, Struct);

    if (NumElements > MaxNumElements &&
        BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_long_constant_composite)) {
      uint64_t NumOfContinuedInstructions = NumElements / MaxNumElements - 1;
      for (uint64_t J = 0; J < NumOfContinuedInstructions; J++) {
        auto *Continued = BM->addTypeStructContinuedINTEL(MaxNumElements);
        Struct->addContinuedInstruction(
            static_cast<SPIRVTypeStruct::ContinuedInstType>(Continued));
      }
      uint64_t Remains = NumElements % MaxNumElements;
      if (Remains) {
        auto *Continued = BM->addTypeStructContinuedINTEL(Remains);
        Struct->addContinuedInstruction(
            static_cast<SPIRVTypeStruct::ContinuedInstType>(Continued));
      }
    }

    SmallVector<unsigned, 4> ForwardRefs;

    for (unsigned I = 0, E = T->getStructNumElements(); I != E; ++I) {
      auto *ElemTy = ST->getElementType(I);
      if ((isa<StructType>(ElemTy) || isa<ArrayType>(ElemTy) ||
           isa<VectorType>(ElemTy) || isa<PointerType>(ElemTy)) &&
          recursiveType(ST, ElemTy))
        ForwardRefs.push_back(I);
      else
        Struct->setMemberType(I, transType(ST->getElementType(I)));
    }

    BM->closeStructType(Struct, ST->isPacked());

    for (auto I : ForwardRefs)
      Struct->setMemberType(I, transType(ST->getElementType(I)));

    return Struct;
  }

  if (FunctionType *FT = dyn_cast<FunctionType>(T)) {
    SPIRVType *RT = transType(FT->getReturnType());
    std::vector<SPIRVType *> PT;
    for (FunctionType::param_iterator I = FT->param_begin(),
                                      E = FT->param_end();
         I != E; ++I)
      PT.push_back(transType(*I));
    return mapType(T, BM->addFunctionType(RT, PT));
  }

  if (T->isLabelTy()) {
    assert(false && "labels can't be mapped as types - handle this earlier!");
    return nullptr;
  }

  llvm_unreachable("Not implemented!");
  return nullptr;
}

SPIRVType *LLVMToSPIRVBase::transSPIRVOpaqueType(Type *T) {
  auto ET = T->getPointerElementType();
  auto ST = cast<StructType>(ET);
  auto STName = ST->getStructName();
  assert(STName.startswith(kSPIRVTypeName::PrefixAndDelim) &&
         "Invalid SPIR-V opaque type name");
  SmallVector<std::string, 9> Postfixes;
  auto TN = decodeSPIRVTypeName(STName, Postfixes);
  if (TN == kSPIRVTypeName::Pipe) {
    assert(T->getPointerAddressSpace() == SPIRAS_Global);
    assert(Postfixes.size() == 1 && "Invalid pipe type ops");
    auto PipeT = BM->addPipeType();
    PipeT->setPipeAcessQualifier(
        static_cast<spv::AccessQualifier>(atoi(Postfixes[0].c_str())));
    return mapType(T, PipeT);
  } else if (TN == kSPIRVTypeName::Image) {
    assert(T->getPointerAddressSpace() == SPIRAS_Global);

    SPIRVType *SampledT = nullptr;
    const auto type_idx = (Postfixes.size() == 8 ? 0u : 1u);
    if (Postfixes[type_idx] == kSPIRVImageSampledTypeName::Void) {
      SampledT = BM->addVoidType();
    } else if (Postfixes[type_idx] == kSPIRVImageSampledTypeName::Float) {
      SampledT = BM->addFloatType(32);
    } else if (Postfixes[type_idx] == kSPIRVImageSampledTypeName::Half) {
      // float16 is currently not supported by Vulkan
      SampledT = BM->addFloatType(SrcLang == spv::SourceLanguageGLSL ? 32 : 16);
    } else if (Postfixes[type_idx] == kSPIRVImageSampledTypeName::UInt) {
      SampledT = BM->addIntegerType(32, false);
    } else if (Postfixes[type_idx] == kSPIRVImageSampledTypeName::Int) {
      SampledT = BM->addIntegerType(32, true);
    } else {
      assert(false && "Invalid sampled type postfix");
    }

    SmallVector<int, 7> Ops;
    const auto start_idx = (Postfixes.size() == 8 ? 1u : 2u);
    const auto end_idx = (Postfixes.size() == 8 ? 8u : 9u);
    for (unsigned I = start_idx; I < end_idx; ++I)
      Ops.push_back(atoi(Postfixes[I].c_str()));
    SPIRVTypeImageDescriptor Desc(static_cast<SPIRVImageDimKind>(Ops[0]),
                                  Ops[1], Ops[2], Ops[3], Ops[4], Ops[5]);
    auto spirv_image_type =
        (static_cast<spv::AccessQualifier>(Ops[6]) != spv::AccessQualifierNone
             ? BM->addImageType(SampledT, Desc,
                                static_cast<spv::AccessQualifier>(Ops[6]))
             : BM->addImageType(SampledT, Desc));
    return mapType(T, spirv_image_type);
  } else if (TN == kSPIRVTypeName::SampledImg) {
    return mapType(
        T, BM->addSampledImageType(static_cast<SPIRVTypeImage *>(
               transType(getSPIRVTypeByChangeBaseTypeName(
                   M, T, kSPIRVTypeName::SampledImg, kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::VmeImageINTEL) {
    // This type is the same as SampledImageType, but consumed by Subgroup AVC
    // Intel extension instructions.
    return mapType(
        T,
        BM->addVmeImageINTELType(static_cast<SPIRVTypeImage *>(
            transType(getSPIRVTypeByChangeBaseTypeName(
                M, T, kSPIRVTypeName::VmeImageINTEL, kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::Sampler)
    return mapType(T, BM->addSamplerType());
  else if (TN == kSPIRVTypeName::DeviceEvent)
    return mapType(T, BM->addDeviceEventType());
  else if (TN == kSPIRVTypeName::Queue)
    return mapType(T, BM->addQueueType());
  else if (TN == kSPIRVTypeName::PipeStorage)
    return mapType(T, BM->addPipeStorageType());
  else if (TN == kSPIRVTypeName::JointMatrixINTEL) {
    Type *ElemTy = nullptr;
    StringRef Ty{Postfixes[0]};
    auto NumBits = llvm::StringSwitch<unsigned>(Ty)
                       .Case("char", 8)
                       .Case("short", 16)
                       .Case("int", 32)
                       .Case("long", 64)
                       .Default(0);
    if (NumBits)
      ElemTy = IntegerType::get(M->getContext(), NumBits);
    else if (Ty == "half")
      ElemTy = Type::getHalfTy(M->getContext());
    else if (Ty == "float")
      ElemTy = Type::getFloatTy(M->getContext());
    else if (Ty == "double")
      ElemTy = Type::getDoubleTy(M->getContext());
    else
      llvm_unreachable("Unexpected type for matrix!");

    auto ParseInteger = [this](StringRef Postfix) -> ConstantInt * {
      unsigned long long N = 0;
      consumeUnsignedInteger(Postfix, 10, N);
      return getUInt32(M, N);
    };
    SPIRVValue *Rows = transConstant(ParseInteger(Postfixes[1]));
    SPIRVValue *Columns = transConstant(ParseInteger(Postfixes[2]));
    SPIRVValue *Layout = transConstant(ParseInteger(Postfixes[3]));
    SPIRVValue *Scope = transConstant(ParseInteger(Postfixes[4]));
    return mapType(T, BM->addJointMatrixINTELType(transType(ElemTy), Rows,
                                                  Columns, Layout, Scope));
  } else
    return mapType(T,
                   BM->addOpaqueGenericType(SPIRVOpaqueTypeOpCodeMap::map(TN)));
}

SPIRVType *LLVMToSPIRVBase::addSignPreservingLLVMType(llvm::Type *type,
                                                      const bool is_signed) {
  const auto add_scalar_uint_type = [this](llvm::Type *scalar_type) {
    assert(scalar_type->isIntegerTy());
    return BM->addIntegerType(cast<IntegerType>(scalar_type)->getBitWidth(),
                              false);
  };

  if (type->isVectorTy()) {
    const auto vec_type = dyn_cast<llvm::FixedVectorType>(type);
    auto elem_type = vec_type->getElementType();
    auto elem_count = vec_type->getNumElements();
    if (is_signed) {
      return BM->addVectorType(transType(type), elem_count);
    } else {
      auto scalar_uint_type = add_scalar_uint_type(elem_type);
      return BM->addVectorType(scalar_uint_type, elem_count);
    }
  } else {
    assert(type->isFloatTy() || type->isIntegerTy());
    if (is_signed) {
      return transType(type);
    } else {
      return add_scalar_uint_type(type);
    }
  }
}

SPIRVFunction *LLVMToSPIRVBase::transFunctionDecl(Function *F) {
  // don't translate/emit entry point declarations when the function only is a
  // declaration, not a definition
  if (F->isDeclaration() && F->getCallingConv() != CallingConv::FLOOR_FUNC)
    return nullptr;

  // skip any floor.* functions, these shouldn't be here
  if (F->getName().startswith("floor."))
    return nullptr;

  // ignore any non-entry-point functions in shader mode
  if (SrcLang == spv::SourceLanguageGLSL &&
      F->getCallingConv() == CallingConv::FLOOR_FUNC)
    return nullptr;

  // return already translated value
  if (auto BF = getTranslatedValue(F))
    return static_cast<SPIRVFunction *>(BF);

  // all shader/glsl entry points need special handling compared to normal and
  // kernel functions
  const auto entry_point_type = getEntryPointType(F, SrcLang);
  SPIRVFunction *BF = nullptr;
  if (entry_point_type == spv::ExecutionModel::ExecutionModelKernel ||
      entry_point_type == spv::ExecutionModel::ExecutionModelInvalid) {
    if (F->isIntrinsic() && (!BM->isSPIRVAllowUnknownIntrinsicsEnabled() ||
                             isKnownIntrinsic(F->getIntrinsicID()))) {
      // We should not translate LLVM intrinsics as a function
      assert(none_of(F->users(),
                     [this](User *U) { return getTranslatedValue(U); }) &&
             "LLVM intrinsics shouldn't be called in SPIRV");
      return nullptr;
    }

    SPIRVTypeFunction *BFT = static_cast<SPIRVTypeFunction *>(
        transType(OCLTypeToSPIRVPtr->getAdaptedType(F)));
    BF = static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
    BF->setFunctionControlMask(transFunctionControlMask(F));
    if (F->hasName())
      BM->setName(BF, F->getName().str());
    if (entry_point_type != spv::ExecutionModel::ExecutionModelInvalid)
      BM->addEntryPoint(ExecutionModelKernel, BF->getId());
    else if (F->getLinkage() != GlobalValue::InternalLinkage)
      BF->setLinkageType(transLinkageType(F));

    // Translate OpenCL/SYCL buffer_location metadata if it's attached to the
    // translated function declaration
    MDNode *BufferLocation = nullptr;
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_buffer_location))
      BufferLocation = F->getMetadata("kernel_arg_buffer_location");

    // Translate runtime_aligned metadata if it's attached to the translated
    // function declaration
    MDNode *RuntimeAligned = nullptr;
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_runtime_aligned))
      RuntimeAligned = F->getMetadata("kernel_arg_runtime_aligned");

    auto Attrs = F->getAttributes();

    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
         ++I) {
      auto ArgNo = I->getArgNo();
      SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
      if (I->hasName())
        BM->setName(BA, I->getName().str());
      if (I->hasByValAttr())
        BA->addAttr(FunctionParameterAttributeByVal);
      if (I->hasNoAliasAttr())
        BA->addAttr(FunctionParameterAttributeNoAlias);
      if (I->hasNoCaptureAttr())
        BA->addAttr(FunctionParameterAttributeNoCapture);
      if (I->hasStructRetAttr())
        BA->addAttr(FunctionParameterAttributeSret);
      if (I->onlyReadsMemory())
        BA->addAttr(FunctionParameterAttributeNoWrite);
      if (Attrs.hasParamAttr(ArgNo, Attribute::ZExt))
        BA->addAttr(FunctionParameterAttributeZext);
      if (Attrs.hasParamAttr(ArgNo, Attribute::SExt))
        BA->addAttr(FunctionParameterAttributeSext);
      if (Attrs.hasParamAttr(ArgNo, Attribute::Alignment)) {
        SPIRVWord AlignmentBytes =
            Attrs.getParamAttr(ArgNo, Attribute::Alignment)
                .getAlignment()
                .valueOrOne()
                .value();
        BA->setAlignment(AlignmentBytes);
      }
      if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1) &&
          Attrs.hasParamAttr(ArgNo, Attribute::Dereferenceable))
        BA->addDecorate(DecorationMaxByteOffset,
                        Attrs.getParamAttr(ArgNo, Attribute::Dereferenceable)
                            .getDereferenceableBytes());
      if (BufferLocation && I->getType()->isPointerTy()) {
        // Order of integer numbers in MD node follows the order of function
        // parameters on which we shall attach the appropriate decoration. Add
        // decoration only if MD value is not negative.
        int LocID = -1;
        if (!isa<MDString>(BufferLocation->getOperand(ArgNo)) &&
            !isa<MDNode>(BufferLocation->getOperand(ArgNo)))
          LocID = getMDOperandAsInt(BufferLocation, ArgNo);
        if (LocID >= 0)
          BA->addDecorate(DecorationBufferLocationINTEL, LocID);
      }
      if (RuntimeAligned && I->getType()->isPointerTy()) {
        // Order of integer numbers in MD node follows the order of function
        // parameters on which we shall attach the appropriate decoration. Add
        // decoration only if MD value is 1.
        int LocID = 0;
        if (!isa<MDString>(RuntimeAligned->getOperand(ArgNo)) &&
            !isa<MDNode>(RuntimeAligned->getOperand(ArgNo)))
          LocID = getMDOperandAsInt(RuntimeAligned, ArgNo);
        if (LocID == 1)
          BA->addDecorate(internal::DecorationRuntimeAlignedINTEL, LocID);
      }
    }
    if (Attrs.hasRetAttr(Attribute::ZExt))
      BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeZext);
    if (Attrs.hasRetAttr(Attribute::SExt))
      BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeSext);
    if (Attrs.hasFnAttr("referenced-indirectly")) {
      assert(!isEntryPoint(F) &&
             "kernel function was marked as referenced-indirectly");
      BF->addDecorate(DecorationReferencedIndirectlyINTEL);
    }

    if (Attrs.hasFnAttr(kVCMetadata::VCCallable) &&
        BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fast_composite)) {
      BF->addDecorate(internal::DecorationCallableFunctionINTEL);
    }

    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
      transVectorComputeMetadata(F);

    transFPGAFunctionMetadata(BF, F);

    SPIRVDBG(dbgs() << "[transFunction (kernel)] " << *F << " => ";
             spvdbgs() << *BF << '\n';)
    return BF;
  } else {
    // shader function is always "void func_name()"
    const auto shader_func_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*Ctx), false);
    SPIRVTypeFunction *BFT =
        static_cast<SPIRVTypeFunction *>(transType(shader_func_type));
    BF = static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
    assert(F->hasName() && "entry point function must have a name");
    BM->setName(BF, F->getName().str());
    BM->addEntryPoint(entry_point_type, BF->getId());
    // NOTE: not handling/adding function parameters here
    SPIRVDBG(dbgs() << "[transFunction (shader)] " << *F << " => ";
             spvdbgs() << *BF << '\n';)
    return BF;
  }
}

void LLVMToSPIRVBase::transVectorComputeMetadata(Function *F) {
  using namespace VectorComputeUtil;
  if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    return;
  auto BF = static_cast<SPIRVFunction *>(getTranslatedValue(F));
  assert(BF && "The SPIRVFunction pointer shouldn't be nullptr");
  auto Attrs = F->getAttributes();

  if (Attrs.hasFnAttr(kVCMetadata::VCStackCall))
    BF->addDecorate(DecorationStackCallINTEL);
  if (Attrs.hasFnAttr(kVCMetadata::VCFunction))
    BF->addDecorate(DecorationVectorComputeFunctionINTEL);
  else
    return;

  if (Attrs.hasFnAttr(kVCMetadata::VCSIMTCall)) {
    SPIRVWord SIMTMode = 0;
    Attrs.getFnAttr(kVCMetadata::VCSIMTCall)
        .getValueAsString()
        .getAsInteger(0, SIMTMode);
    BF->addDecorate(DecorationSIMTCallINTEL, SIMTMode);
  }

  if (Attrs.hasRetAttr(kVCMetadata::VCSingleElementVector))
    translateSEVDecoration(
        Attrs.getAttributeAtIndex(AttributeList::ReturnIndex,
                                  kVCMetadata::VCSingleElementVector),
        BF);

  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto ArgNo = I->getArgNo();
    SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCArgumentIOKind)) {
      SPIRVWord Kind = {};
      Attrs.getParamAttr(ArgNo, kVCMetadata::VCArgumentIOKind)
          .getValueAsString()
          .getAsInteger(0, Kind);
      BA->addDecorate(DecorationFuncParamIOKindINTEL, Kind);
    }
    if (Attrs.hasParamAttr(ArgNo, kVCMetadata::VCSingleElementVector))
      translateSEVDecoration(
          Attrs.getParamAttr(ArgNo, kVCMetadata::VCSingleElementVector), BA);
  }
  if (!isEntryPoint(F) &&
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_float_controls2) &&
      Attrs.hasFnAttr(kVCMetadata::VCFloatControl)) {

    SPIRVWord Mode = 0;
    Attrs.getFnAttr(kVCMetadata::VCFloatControl)
        .getValueAsString()
        .getAsInteger(0, Mode);
    VCFloatTypeSizeMap::foreach (
        [&](VCFloatType FloatType, unsigned TargetWidth) {
          BF->addDecorate(new SPIRVDecorateFunctionDenormModeINTEL(
              BF, TargetWidth, getFPDenormMode(Mode, FloatType)));

          BF->addDecorate(new SPIRVDecorateFunctionRoundingModeINTEL(
              BF, TargetWidth, getFPRoundingMode(Mode)));

          BF->addDecorate(new SPIRVDecorateFunctionFloatingPointModeINTEL(
              BF, TargetWidth, getFPOperationMode(Mode)));
        });
  }
}

void LLVMToSPIRVBase::transFPGAFunctionMetadata(SPIRVFunction *BF,
                                                Function *F) {
  if (MDNode *StallEnable = F->getMetadata(kSPIR2MD::StallEnable)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_cluster_attributes)) {
      if (getMDOperandAsInt(StallEnable, 0))
        BF->addDecorate(new SPIRVDecorateStallEnableINTEL(BF));
    }
  }
  if (MDNode *LoopFuse = F->getMetadata(kSPIR2MD::LoopFuse)) {
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_loop_fuse)) {
      size_t Depth = getMDOperandAsInt(LoopFuse, 0);
      size_t Independent = getMDOperandAsInt(LoopFuse, 1);
      BF->addDecorate(
          new SPIRVDecorateFuseLoopsInFunctionINTEL(BF, Depth, Independent));
    }
  }
  if (MDNode *PreferDSP = F->getMetadata(kSPIR2MD::PreferDSP)) {
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_dsp_control)) {
      size_t Mode = getMDOperandAsInt(PreferDSP, 0);
      MDNode *PropDSPPref = F->getMetadata(kSPIR2MD::PropDSPPref);
      size_t Propagate = PropDSPPref ? getMDOperandAsInt(PropDSPPref, 0) : 0;
      BF->addDecorate(new SPIRVDecorateMathOpDSPModeINTEL(BF, Mode, Propagate));
    }
  }
  if (MDNode *InitiationInterval =
          F->getMetadata(kSPIR2MD::InitiationInterval)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      if (size_t Cycles = getMDOperandAsInt(InitiationInterval, 0))
        BF->addDecorate(new SPIRVDecorateInitiationIntervalINTEL(BF, Cycles));
    }
  }
  if (MDNode *MaxConcurrency = F->getMetadata(kSPIR2MD::MaxConcurrency)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      size_t Invocations = getMDOperandAsInt(MaxConcurrency, 0);
      BF->addDecorate(new SPIRVDecorateMaxConcurrencyINTEL(BF, Invocations));
    }
  }
  if (MDNode *DisableLoopPipelining =
          F->getMetadata(kSPIR2MD::DisableLoopPipelining)) {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes)) {
      if (size_t Disable = getMDOperandAsInt(DisableLoopPipelining, 0))
        BF->addDecorate(new SPIRVDecoratePipelineEnableINTEL(BF, !Disable));
    }
  }
}

SPIRVValue *LLVMToSPIRVBase::transConstant(Value *V) {
  if (auto CPNull = dyn_cast<ConstantPointerNull>(V))
    return BM->addNullConstant(
        bcast<SPIRVTypePointer>(transType(CPNull->getType())));

  if (auto CAZero = dyn_cast<ConstantAggregateZero>(V)) {
    Type *AggType = CAZero->getType();
    if (const StructType *ST = dyn_cast<StructType>(AggType))
      if (ST->hasName() &&
          ST->getName() == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
        return BM->addSamplerConstant(transType(AggType), 0, 0, 0);

    return BM->addNullConstant(transType(AggType));
  }

  if (auto ConstI = dyn_cast<ConstantInt>(V)) {
    unsigned BitWidth = ConstI->getType()->getBitWidth();
    if (BitWidth > 64) {
      BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_arbitrary_precision_integers),
          SPIRVEC_InvalidBitWidth, std::to_string(BitWidth));
      return BM->addConstant(transType(V->getType()), ConstI->getValue());
    }
    return BM->addConstant(transType(V->getType()), ConstI->getZExtValue());
  }

  if (auto ConstFP = dyn_cast<ConstantFP>(V)) {
    auto BT = static_cast<SPIRVType *>(transType(V->getType()));
    return BM->addConstant(
        BT, ConstFP->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  if (auto ConstDA = dyn_cast<ConstantDataArray>(V)) {
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDA->getNumElements(); I != E; ++I)
      BV.push_back(transValue(ConstDA->getElementAsConstant(I), nullptr, true,
                              FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstA = dyn_cast<ConstantArray>(V)) {
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstA->op_begin(), E = ConstA->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr, true, FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstDV = dyn_cast<ConstantDataVector>(V)) {
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDV->getNumElements(); I != E; ++I)
      BV.push_back(transValue(ConstDV->getElementAsConstant(I), nullptr, true,
                              FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstV = dyn_cast<ConstantVector>(V)) {
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr, true, FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (const auto *ConstV = dyn_cast<ConstantStruct>(V)) {
    StringRef StructName;
    if (ConstV->getType()->hasName())
      StructName = ConstV->getType()->getName();
    if (StructName == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler)) {
      assert(ConstV->getNumOperands() == 3);
      SPIRVWord AddrMode =
                    ConstV->getOperand(0)->getUniqueInteger().getZExtValue(),
                Normalized =
                    ConstV->getOperand(1)->getUniqueInteger().getZExtValue(),
                FilterMode =
                    ConstV->getOperand(2)->getUniqueInteger().getZExtValue();
      assert(AddrMode < 5 && "Invalid addressing mode");
      assert(Normalized < 2 && "Invalid value of normalized coords");
      assert(FilterMode < 2 && "Invalid filter mode");
      SPIRVType *SamplerTy = transType(ConstV->getType());
      return BM->addSamplerConstant(SamplerTy, AddrMode, Normalized,
                                    FilterMode);
    }
    if (StructName == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage)) {
      assert(ConstV->getNumOperands() == 3);
      SPIRVWord PacketSize =
                    ConstV->getOperand(0)->getUniqueInteger().getZExtValue(),
                PacketAlign =
                    ConstV->getOperand(1)->getUniqueInteger().getZExtValue(),
                Capacity =
                    ConstV->getOperand(2)->getUniqueInteger().getZExtValue();
      assert(PacketAlign >= 1 && "Invalid packet alignment");
      assert(PacketSize >= PacketAlign && PacketSize % PacketAlign == 0 &&
             "Invalid packet size and/or alignment.");
      SPIRVType *PipeStorageTy = transType(ConstV->getType());
      return BM->addPipeStorageConstant(PipeStorageTy, PacketSize, PacketAlign,
                                        Capacity);
    }
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr, true, FuncTransMode::Pointer));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstUE = dyn_cast<ConstantExpr>(V)) {
    auto Inst = ConstUE->getAsInstruction();
    SPIRVDBG(dbgs() << "ConstantExpr: " << *ConstUE << '\n';
             dbgs() << "Instruction: " << *Inst << '\n';)
    auto BI = transValue(Inst, nullptr, false);
    Inst->dropAllReferences();
    UnboundInst.push_back(Inst);
    return BI;
  }

  if (isa<UndefValue>(V)) {
    // TODO/NOTE: don't allow global undef constants in Vulkan/GLSL until
    // drivers (AMD) catch up
    if (SrcLang == spv::SourceLanguageGLSL) {
      return nullptr;
    } else {
      return BM->addUndef(transType(V->getType()));
    }
  }

  return nullptr;
}

SPIRVValue *LLVMToSPIRVBase::transValue(Value *V, SPIRVBasicBlock *BB,
                                        bool CreateForward,
                                        FuncTransMode FuncTrans) {
  LLVMToSPIRVValueMap::iterator Loc = ValueMap.find(V);
  if (Loc != ValueMap.end() && (!Loc->second->isForward() || CreateForward) &&
      // do not return forward-decl of a function if we
      // actually want to create a function pointer
      !(FuncTrans == FuncTransMode::Pointer && isa<Function>(V)))
    return Loc->second;

  SPIRVDBG(dbgs() << "[transValue] " << *V << '\n');
  assert((!isa<Instruction>(V) || isa<GetElementPtrInst>(V) ||
          isa<CastInst>(V) || BB) &&
         "Invalid SPIRV BB");

  auto BV = transValueWithoutDecoration(V, BB, CreateForward, FuncTrans);
  if (!BV || !transDecoration(V, BV))
    return nullptr;
  StringRef Name = V->getName();
  if (!Name.empty()) // Don't erase the name, which BM might already have
    BM->setName(BV, Name.str());
  return BV;
}

SPIRVInstruction *LLVMToSPIRVBase::transBinaryInst(BinaryOperator *B,
                                                   SPIRVBasicBlock *BB) {
  // in LLVM (fneg x) is represented as (fsub +/-0 x) -> special case this to
  // produce OpFNegate instead
  Value *fneg_val;
  if (PatternMatch::match(
          B, PatternMatch::m_FNeg(PatternMatch::m_Value(fneg_val)))) {
    return BM->addUnaryInst(spv::OpFNegate, transType(B->getType()),
                            transValue(B->getOperand(1), BB), BB);
  }

  unsigned LLVMOC = B->getOpcode();
  Op BOC = OpCodeMap::map(LLVMOC);
  auto Op0 = transValue(B->getOperand(0), BB);

  // take care of signed/unsigned type conversion mismatches,
  // TODO: as with unary instructions, we need to do this properly at some point
  const auto type = transType(B->getType());
  const auto is_int = type->isTypeInt();
  const auto is_sint = (is_int ? ((SPIRVTypeInt *)type)->isSigned() : false);
  const auto is_uint = (is_int ? !((SPIRVTypeInt *)type)->isSigned() : false);
  switch (BOC) {
  case spv::OpUMod:
    if (is_sint) {
      BOC = OpSMod;
    }
    break;
  case spv::OpSMod:
    if (is_uint) {
      BOC = OpUMod;
    }
    break;
  case spv::OpUDiv:
    if (is_sint) {
      BOC = OpSDiv;
    }
    break;
  case spv::OpSDiv:
    if (is_uint) {
      BOC = OpUDiv;
    }
    break;
  default:
    break;
  }

  SPIRVInstruction *BI =
      BM->addBinaryInst(transBoolOpCode(Op0, BOC), type, Op0,
                        transValue(B->getOperand(1), BB), BB);

#if 0 // this is stupid
  if (isUnfusedMulAdd(B)) {
    Function *F = B->getFunction();
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                    << ": possible fma candidate " << *B << '\n');
    joinFPContract(F, FPContract::DISABLED);
  }
#endif

  return BI;
}

SPIRVInstruction *LLVMToSPIRVBase::transCmpInst(CmpInst *Cmp,
                                                SPIRVBasicBlock *BB) {
  auto *Op0 = Cmp->getOperand(0);
  SPIRVValue *TOp0 = transValue(Op0, BB);
  SPIRVValue *TOp1 = transValue(Cmp->getOperand(1), BB);
  if (Op0->getType()->isPointerTy()) {
    // TODO: once OpenCL supports SPIR-V 1.4, use this as well
    if (SrcLang == spv::SourceLanguageGLSL) {
      // -> can use PtrEqual/PtrNotEqual
      const auto pred = Cmp->getPredicate();
      assert(pred == CmpInst::ICMP_EQ || pred == CmpInst::ICMP_NE);
      getErrorLog().checkError(
          pred == CmpInst::ICMP_EQ || pred == CmpInst::ICMP_NE,
          SPIRVEC_InvalidInstruction, Cmp,
          "pointer compare predicate must be equal or not-equal\n");
      const auto op =
          (pred == CmpInst::ICMP_EQ ? spv::OpPtrEqual : spv::OpPtrNotEqual);
      return BM->addPtrCmpInst(op, transType(Cmp->getType()), TOp0, TOp1, BB);
    }

    unsigned AS = cast<PointerType>(Op0->getType())->getAddressSpace();
    SPIRVType *Ty = transType(getSizetType(AS));
    TOp0 = BM->addUnaryInst(OpConvertPtrToU, Ty, TOp0, BB);
    TOp1 = BM->addUnaryInst(OpConvertPtrToU, Ty, TOp1, BB);
  }
  SPIRVInstruction *BI =
      BM->addCmpInst(transBoolOpCode(TOp0, CmpMap::map(Cmp->getPredicate())),
                     transType(Cmp->getType()), TOp0, TOp1, BB);
  return BI;
}

SPIRV::SPIRVInstruction *LLVMToSPIRVBase::transUnaryInst(UnaryInstruction *U,
                                                         SPIRVBasicBlock *BB) {
  // TODO: properly handle int/uint conversions and type handling
  Op BOC = OpNop;
  if (auto Cast = dyn_cast<AddrSpaceCastInst>(U)) {
    const auto SrcAddrSpace = Cast->getSrcTy()->getPointerAddressSpace();
    const auto DestAddrSpace = Cast->getDestTy()->getPointerAddressSpace();
    if (DestAddrSpace == SPIRAS_Generic) {
      getErrorLog().checkError(
          SrcAddrSpace != SPIRAS_Constant, SPIRVEC_InvalidModule, U,
          "Casts from constant address space to generic are illegal\n");
      BOC = OpPtrCastToGeneric;
      // In SPIR-V only casts to/from generic are allowed. But with
      // SPV_INTEL_usm_storage_classes we can also have casts from global_device
      // and global_host to global addr space and vice versa.
    } else if (SrcAddrSpace == SPIRAS_GlobalDevice ||
               SrcAddrSpace == SPIRAS_GlobalHost) {
      getErrorLog().checkError(DestAddrSpace == SPIRAS_Global ||
                                   DestAddrSpace == SPIRAS_Generic,
                               SPIRVEC_InvalidModule, U,
                               "Casts from global_device/global_host only "
                               "allowed to global/generic\n");
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_usm_storage_classes)) {
        if (DestAddrSpace == SPIRAS_Global)
          return nullptr;
        BOC = OpPtrCastToGeneric;
      } else {
        BOC = OpPtrCastToCrossWorkgroupINTEL;
      }
    } else if (DestAddrSpace == SPIRAS_GlobalDevice ||
               DestAddrSpace == SPIRAS_GlobalHost) {
      getErrorLog().checkError(SrcAddrSpace == SPIRAS_Global ||
                                   SrcAddrSpace == SPIRAS_Generic,
                               SPIRVEC_InvalidModule, U,
                               "Casts to global_device/global_host only "
                               "allowed from global/generic\n");
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_usm_storage_classes)) {
        if (SrcAddrSpace == SPIRAS_Global)
          return nullptr;
        BOC = OpGenericCastToPtr;
      } else {
        BOC = OpCrossWorkgroupCastToPtrINTEL;
      }
    } else {
      getErrorLog().checkError(
          SrcAddrSpace == SPIRAS_Generic, SPIRVEC_InvalidModule, U,
          "Casts from private/local/global address space are allowed only to "
          "generic\n");
      getErrorLog().checkError(
          DestAddrSpace != SPIRAS_Constant, SPIRVEC_InvalidModule, U,
          "Casts from generic address space to constant are illegal\n");
      BOC = OpGenericCastToPtr;
    }
  } else {
    auto OpCode = U->getOpcode();
    BOC = OpCodeMap::map(OpCode);
  }

  if (SrcLang != spv::SourceLanguageGLSL) {
    auto Op = transValue(U->getOperand(0), BB, true, FuncTransMode::Pointer);
    return BM->addUnaryInst(transBoolOpCode(Op, BOC), transType(U->getType()),
                            Op, BB);
  }

  auto val = transValue(U->getOperand(0), BB, true, FuncTransMode::Pointer);

  // take care of signed/unsigned type conversion mismatches,
  // as stated above this should be done properly at some point
  // -> fixup superficial stuff caused by unary int conversion translation
  const auto type = transType(U->getType());
  const auto is_int = type->isTypeInt();
  const auto is_sint = (is_int ? ((SPIRVTypeInt *)type)->isSigned() : false);
  const auto is_uint = (is_int ? !((SPIRVTypeInt *)type)->isSigned() : false);
  bool bitcast_output = false;
  SPIRVType *conv_type = nullptr;
  switch (BOC) {
  case spv::OpUConvert:
    if (is_sint) {
      bitcast_output = true;
      conv_type = ((SPIRVTypeInt *)type)->getUnsigned();
    }
    break;
  case spv::OpSConvert:
    if (is_uint) {
      bitcast_output = true;
      conv_type = ((SPIRVTypeInt *)type)->getSigned();
    }
    break;
  case spv::OpConvertFToU:
    if (is_sint) {
      bitcast_output = true;
      conv_type = ((SPIRVTypeInt *)type)->getUnsigned();
    }
    break;
  case spv::OpConvertFToS:
    if (is_uint) {
      bitcast_output = true;
      conv_type = ((SPIRVTypeInt *)type)->getSigned();
    }
    break;
  // also handle input value conversion
  case spv::OpConvertUToF:
    if (is_sint) {
      val = BM->addUnaryInst(spv::OpBitcast,
                             ((SPIRVTypeInt *)type)->getUnsigned(), val, BB);
    }
    break;
  case spv::OpConvertSToF:
    if (is_uint) {
      val = BM->addUnaryInst(spv::OpBitcast,
                             ((SPIRVTypeInt *)type)->getSigned(), val, BB);
    }
    break;
  default:
    break;
  }
  auto conv = BM->addUnaryInst(transBoolOpCode(val, BOC),
                               conv_type ? conv_type : type, val, BB);
  if (bitcast_output) {
    return BM->addUnaryInst(spv::OpBitcast, type, conv, BB);
  }
  return conv;
}

/// This helper class encapsulates information extraction from
/// "llvm.loop.parallel_access_indices" metadata hints. Initialize
/// with a pointer to an MDNode with the following structure:
/// !<Node> = !{!"llvm.loop.parallel_access_indices", !<Node>, !<Node>, ...}
/// OR:
/// !<Node> = !{!"llvm.loop.parallel_access_indices", !<Nodes...>, i32 <value>}
///
/// All of the MDNode-type operands mark the index groups for particular
/// array variables. An optional i32 value indicates the safelen (safe
/// number of iterations) for the optimization application to these
/// array variables. If the safelen value is absent, an infinite
/// number of iterations is implied.
class LLVMParallelAccessIndices {
public:
  LLVMParallelAccessIndices(
      MDNode *Node, LLVMToSPIRVBase::LLVMToSPIRVMetadataMap &IndexGroupArrayMap)
      : Node(Node), IndexGroupArrayMap(IndexGroupArrayMap) {}

  void initialize() {
    assert(isValid() &&
           "LLVMParallelAccessIndices initialized from an invalid MDNode");

    unsigned NumOperands = Node->getNumOperands();
    auto *SafeLenExpression = mdconst::dyn_extract_or_null<ConstantInt>(
        Node->getOperand(NumOperands - 1));
    // If no safelen value is specified and the last operand
    // casts to an MDNode* rather than an int, 0 will be stored
    SafeLen = SafeLenExpression ? SafeLenExpression->getZExtValue() : 0;

    // Count MDNode operands that refer to index groups:
    // - operand [0] is a string literal and should be ignored;
    // - depending on whether a particular safelen is specified as the
    //   last operand, we may or may not want to extract the latter
    //   as an index group
    unsigned NumIdxGroups = SafeLen ? NumOperands - 2 : NumOperands - 1;
    for (unsigned I = 1; I <= NumIdxGroups; ++I) {
      MDNode *IdxGroupNode = getMDOperandAsMDNode(Node, I);
      assert(IdxGroupNode &&
             "Invalid operand in the MDNode for LLVMParallelAccessIndices");
      auto IdxGroupArrayPairIt = IndexGroupArrayMap.find(IdxGroupNode);
      // TODO: Some LLVM IR optimizations (e.g. loop inlining as part of
      // the function inlining) can result in invalid parallel_access_indices
      // metadata. Only valid cases will pass the subsequent check and
      // survive the translation. This check should be replaced with an
      // assertion once all known cases are handled.
      if (IdxGroupArrayPairIt != IndexGroupArrayMap.end())
        for (SPIRVId ArrayAccessId : IdxGroupArrayPairIt->second)
          ArrayVariablesVec.push_back(ArrayAccessId);
    }
  }

  bool isValid() {
    bool IsNamedCorrectly = getMDOperandAsString(Node, 0) == ExpectedName;
    return Node && IsNamedCorrectly;
  }

  unsigned getSafeLen() { return SafeLen; }
  const std::vector<SPIRVId> &getArrayVariables() { return ArrayVariablesVec; }

private:
  MDNode *Node;
  LLVMToSPIRVBase::LLVMToSPIRVMetadataMap &IndexGroupArrayMap;
  const std::string ExpectedName = "llvm.loop.parallel_access_indices";
  std::vector<SPIRVId> ArrayVariablesVec;
  unsigned SafeLen;
};

/// Go through the operands !llvm.loop metadata attached to the branch
/// instruction, fill the Loop Control mask and possible parameters for its
/// fields.
spv::LoopControlMask
LLVMToSPIRVBase::getLoopControl(const BranchInst *Branch,
                                std::vector<SPIRVWord> &Parameters) {
  // do not allow this for Vulkan at all
  if (!Branch || SrcLang == spv::SourceLanguageGLSL)
    return spv::LoopControlMaskNone;
  MDNode *LoopMD = Branch->getMetadata("llvm.loop");
  if (!LoopMD)
    return spv::LoopControlMaskNone;

  size_t LoopControl = spv::LoopControlMaskNone;
  std::vector<std::pair<SPIRVWord, SPIRVWord>> ParametersToSort;
  // If only a subset of loop count parameters is defined in metadata
  // then undefined ones should have a default value -1 in SPIR-V.
  // Preset all loop count parameters with the default value.
  struct LoopCountInfo {
    int64_t Min = -1, Max = -1, Avg = -1;
  } LoopCount;

  // Unlike with most of the cases, some loop metadata specifications
  // can occur multiple times - for these, all correspondent tokens
  // need to be collected first, and only then added to SPIR-V loop
  // parameters in a separate routine
  std::vector<std::pair<SPIRVWord, SPIRVWord>> DependencyArrayParameters;

  for (const MDOperand &MDOp : LoopMD->operands()) {
    if (MDNode *Node = dyn_cast<MDNode>(MDOp)) {
      std::string S = getMDOperandAsString(Node, 0);
      // Set the loop control bits. Parameters are set in the order described
      // in 3.23 SPIR-V Spec. rev. 1.4:
      // Bits that are set can indicate whether an additional operand follows,
      // as described by the table. If there are multiple following operands
      // indicated, they are ordered: Those indicated by smaller-numbered bits
      // appear first.
      if (S == "llvm.loop.unroll.disable")
        LoopControl |= spv::LoopControlDontUnrollMask;
      else if (S == "llvm.loop.unroll.full" || S == "llvm.loop.unroll.enable")
        LoopControl |= spv::LoopControlUnrollMask;
      // PartialCount must not be used with the DontUnroll bit
      else if (S == "llvm.loop.unroll.count" &&
               !(LoopControl & LoopControlDontUnrollMask)) {
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)) {
          BM->setMinSPIRVVersion(
              static_cast<SPIRVWord>(VersionNumber::SPIRV_1_4));
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlPartialCountMask, I);
          LoopControl |= spv::LoopControlPartialCountMask;
        }
      } else if (S == "llvm.loop.ivdep.enable")
        LoopControl |= spv::LoopControlDependencyInfiniteMask;
      else if (S == "llvm.loop.ivdep.safelen") {
        size_t I = getMDOperandAsInt(Node, 1);
        ParametersToSort.emplace_back(spv::LoopControlDependencyLengthMask, I);
        LoopControl |= spv::LoopControlDependencyLengthMask;
      } else if (BM->isAllowedToUseExtension(
                     ExtensionID::SPV_INTEL_fpga_loop_controls)) {
        // Add Intel specific Loop Control masks
        if (S == "llvm.loop.ii.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlInitiationIntervalINTELMask, I);
          LoopControl |= spv::LoopControlInitiationIntervalINTELMask;
        } else if (S == "llvm.loop.max_concurrency.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlMaxConcurrencyINTELMask,
                                        I);
          LoopControl |= spv::LoopControlMaxConcurrencyINTELMask;
        } else if (S == "llvm.loop.parallel_access_indices") {
          // Intel FPGA IVDep loop attribute
          LLVMParallelAccessIndices IVDep(Node, IndexGroupArrayMap);
          IVDep.initialize();
          // Store IVDep-specific parameters into an intermediate
          // container to address the case when there're multiple
          // IVDep metadata nodes and this condition gets entered multiple
          // times. The update of the main parameters vector & the loop control
          // mask will be done later, in the main scope of the function
          unsigned SafeLen = IVDep.getSafeLen();
          for (auto &ArrayId : IVDep.getArrayVariables())
            DependencyArrayParameters.emplace_back(ArrayId, SafeLen);
        } else if (S == "llvm.loop.intel.pipelining.enable") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlPipelineEnableINTELMask,
                                        I);
          LoopControl |= spv::LoopControlPipelineEnableINTELMask;
        } else if (S == "llvm.loop.coalesce.enable") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          ParametersToSort.emplace_back(spv::LoopControlLoopCoalesceINTELMask,
                                        0);
          LoopControl |= spv::LoopControlLoopCoalesceINTELMask;
        } else if (S == "llvm.loop.coalesce.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(spv::LoopControlLoopCoalesceINTELMask,
                                        I);
          LoopControl |= spv::LoopControlLoopCoalesceINTELMask;
        } else if (S == "llvm.loop.max_interleaving.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlMaxInterleavingINTELMask, I);
          LoopControl |= spv::LoopControlMaxInterleavingINTELMask;
        } else if (S == "llvm.loop.intel.speculated.iterations.count") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          size_t I = getMDOperandAsInt(Node, 1);
          ParametersToSort.emplace_back(
              spv::LoopControlSpeculatedIterationsINTELMask, I);
          LoopControl |= spv::LoopControlSpeculatedIterationsINTELMask;
        } else if (S == "llvm.loop.fusion.disable") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopControl |= spv::LoopControlNoFusionINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_min") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Min = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::internal::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_max") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Max = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::internal::LoopControlLoopCountINTELMask;
        } else if (S == "llvm.loop.intel.loopcount_avg") {
          BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
          BM->addCapability(CapabilityFPGALoopControlsINTEL);
          LoopCount.Avg = getMDOperandAsInt(Node, 1);
          LoopControl |= spv::internal::LoopControlLoopCountINTELMask;
        }
      }
    }
  }
  if (LoopControl & spv::internal::LoopControlLoopCountINTELMask) {
    // LoopCountINTELMask have int64 literal parameters and we need to store
    // int64 into 2 SPIRVWords
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Min));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Min >> 32));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Max));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Max >> 32));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Avg));
    ParametersToSort.emplace_back(spv::internal::LoopControlLoopCountINTELMask,
                                  static_cast<SPIRVWord>(LoopCount.Avg >> 32));
  }
  // If any loop control parameters were held back until fully collected,
  // now is the time to move the information to the main parameters collection
  if (!DependencyArrayParameters.empty()) {
    // The first parameter states the number of <array, safelen> pairs to be
    // listed
    ParametersToSort.emplace_back(spv::LoopControlDependencyArrayINTELMask,
                                  DependencyArrayParameters.size());
    for (auto &ArraySflnPair : DependencyArrayParameters) {
      ParametersToSort.emplace_back(spv::LoopControlDependencyArrayINTELMask,
                                    ArraySflnPair.first);
      ParametersToSort.emplace_back(spv::LoopControlDependencyArrayINTELMask,
                                    ArraySflnPair.second);
    }
    BM->addExtension(ExtensionID::SPV_INTEL_fpga_loop_controls);
    BM->addCapability(CapabilityFPGALoopControlsINTEL);
    LoopControl |= spv::LoopControlDependencyArrayINTELMask;
  }

  std::stable_sort(ParametersToSort.begin(), ParametersToSort.end(),
                   [](const std::pair<SPIRVWord, SPIRVWord> &CompareLeft,
                      const std::pair<SPIRVWord, SPIRVWord> &CompareRight) {
                     return CompareLeft.first < CompareRight.first;
                   });
  for (auto Param : ParametersToSort)
    Parameters.push_back(Param.second);

  return static_cast<spv::LoopControlMask>(LoopControl);
}

static int transAtomicOrdering(llvm::AtomicOrdering Ordering) {
  return OCLMemOrderMap::map(
      static_cast<OCLMemOrderKind>(llvm::toCABI(Ordering)));
}

SPIRVValue *LLVMToSPIRVBase::transAtomicStore(StoreInst *ST,
                                              SPIRVBasicBlock *BB) {
  std::vector<Value *> Ops{ST->getPointerOperand(),
                           getUInt32(M, spv::ScopeDevice),
                           getUInt32(M, transAtomicOrdering(ST->getOrdering())),
                           ST->getValueOperand()};
  std::vector<SPIRVValue *> SPIRVOps = transValue(Ops, BB);

  return mapValue(ST, BM->addInstTemplate(OpAtomicStore, BM->getIds(SPIRVOps),
                                          BB, nullptr));
}

SPIRVValue *LLVMToSPIRVBase::transAtomicLoad(LoadInst *LD,
                                             SPIRVBasicBlock *BB) {
  std::vector<Value *> Ops{
      LD->getPointerOperand(), getUInt32(M, spv::ScopeDevice),
      getUInt32(M, transAtomicOrdering(LD->getOrdering()))};
  std::vector<SPIRVValue *> SPIRVOps = transValue(Ops, BB);

  return mapValue(LD, BM->addInstTemplate(OpAtomicLoad, BM->getIds(SPIRVOps),
                                          BB, transType(LD->getType())));
}

// Aliasing list MD contains several scope MD nodes whithin it. Each scope MD
// has a selfreference and an extra MD node for aliasing domain and also it
// can contain an optional string operand. Domain MD contains a self-reference
// with an optional string operand. Here we unfold the list, creating SPIR-V
// aliasing instructions.
// TODO: add support for an optional string operand.
SPIRVEntry *addMemAliasingINTELInstructions(SPIRVModule *M,
                                            MDNode *AliasingListMD) {
  if (AliasingListMD->getNumOperands() == 0)
    return nullptr;
  std::vector<SPIRVId> ListId;
  for (const MDOperand &MDListOp : AliasingListMD->operands()) {
    if (MDNode *ScopeMD = dyn_cast<MDNode>(MDListOp)) {
      if (ScopeMD->getNumOperands() < 2)
        return nullptr;
      MDNode *DomainMD = dyn_cast<MDNode>(ScopeMD->getOperand(1));
      if (!DomainMD)
        return nullptr;
      auto *Domain =
          M->getOrAddAliasDomainDeclINTELInst(std::vector<SPIRVId>(), DomainMD);
      auto *Scope =
          M->getOrAddAliasScopeDeclINTELInst({Domain->getId()}, ScopeMD);
      ListId.push_back(Scope->getId());
    }
  }
  return M->getOrAddAliasScopeListDeclINTELInst(ListId, AliasingListMD);
}

// Translate alias.scope/noalias metadata attached to store and load
// instructions.
void transAliasingMemAccess(SPIRVModule *BM, MDNode *AliasingListMD,
                            std::vector<uint32_t> &MemoryAccess,
                            SPIRVWord MemAccessMask) {
  if (!BM->isAllowedToUseExtension(
          ExtensionID::SPV_INTEL_memory_access_aliasing))
    return;
  auto *MemAliasList = addMemAliasingINTELInstructions(BM, AliasingListMD);
  if (!MemAliasList)
    return;
  MemoryAccess[0] |= MemAccessMask;
  MemoryAccess.push_back(MemAliasList->getId());
}

/// An instruction may use an instruction from another BB which has not been
/// translated. SPIRVForward should be created as place holder for these
/// instructions and replaced later by the real instructions.
/// Use CreateForward = true to indicate such situation.
SPIRVValue *
LLVMToSPIRVBase::transValueWithoutDecoration(Value *V, SPIRVBasicBlock *BB,
                                             bool CreateForward,
                                             FuncTransMode FuncTrans) {
  if (auto LBB = dyn_cast<BasicBlock>(V)) {
    auto BF =
        static_cast<SPIRVFunction *>(getTranslatedValue(LBB->getParent()));
    assert(BF && "Function not translated");
    BB = static_cast<SPIRVBasicBlock *>(mapValue(V, BM->addBasicBlock(BF)));
    BM->setName(BB, LBB->getName().str());
    return BB;
  }

  if (auto *F = dyn_cast<Function>(V)) {
    if (FuncTrans == FuncTransMode::Decl)
      return transFunctionDecl(F);
    if (!BM->checkExtension(ExtensionID::SPV_INTEL_function_pointers,
                            SPIRVEC_FunctionPointers, toString(V)))
      return nullptr;
    return BM->addConstFunctionPointerINTEL(
        transType(F->getType()),
        static_cast<SPIRVFunction *>(transValue(F, nullptr)));
  }

  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    llvm::PointerType *Ty = GV->getType();

    if (GV->hasName() && GV->getName().find(".vulkan") != std::string::npos) {
      // special global variables are handled/added in transFunction
      // (note: should only be output vars here)
      return nullptr;
    }

    // Though variables with common linkage type are initialized by 0,
    // they can be represented in SPIR-V as uninitialized variables with
    // 'Export' linkage type, just as tentative definitions look in C
    llvm::Value *Init = GV->hasInitializer() && !GV->hasCommonLinkage()
                            ? GV->getInitializer()
                            : nullptr;

    SPIRVStorageClassKind StorageClass;
    auto AddressSpace = static_cast<SPIRAddressSpace>(Ty->getAddressSpace());
    bool IsVectorCompute =
        BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute) &&
        GV->hasAttribute(kVCMetadata::VCGlobalVariable);
    if (IsVectorCompute)
      StorageClass =
          VectorComputeUtil::getVCGlobalVarStorageClass(AddressSpace);
    else {
      // Lower global_device and global_host address spaces that were added in
      // SYCL as part of SYCL_INTEL_usm_address_spaces extension to just global
      // address space if device doesn't support SPV_INTEL_usm_storage_classes
      // extension
      if ((AddressSpace == SPIRAS_GlobalDevice ||
           AddressSpace == SPIRAS_GlobalHost) &&
          !BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_usm_storage_classes))
        AddressSpace = SPIRAS_Global;
      StorageClass = SPIRSPIRVAddrSpaceMap::map(AddressSpace);
      assert(((StorageClass == spv::StorageClassFunction && BB != nullptr) ||
              StorageClass != spv::StorageClassFunction) &&
             "invalid GV/BB");
    }

    // for Vulkan, we will remove invalid initializers (zero or undef needs to
    // be present in LLVM,  but not in SPIR-V)
    SPIRVValue *BVarInit = nullptr;
    StructType *ST = Init ? dyn_cast<StructType>(Init->getType()) : nullptr;
    if (ST && ST->hasName() && isSPIRVConstantName(ST->getName())) {
      auto BV = transConstant(Init);
      assert(BV);
      return mapValue(V, BV);
    } else if (ConstantExpr *ConstUE = dyn_cast_or_null<ConstantExpr>(Init)) {
      Instruction *Inst = ConstUE->getAsInstruction();
      if (isSpecialTypeInitializer(Inst)) {
        Init = Inst->getOperand(0);
        Ty = static_cast<PointerType *>(Init->getType());
      }
      Inst->dropAllReferences();
      UnboundInst.push_back(Inst);
      BVarInit = (StorageClass == StorageClassWorkgroup || Init == nullptr
                      ? nullptr
                      : transValue(Init, nullptr));
    } else if (ST && isa<UndefValue>(Init)) {
      // Undef initializer for LLVM structure be can translated to
      // OpConstantComposite with OpUndef constituents.
      if (SrcLang == spv::SourceLanguageGLSL) {
        auto I = ValueMap.find(Init);
        if (I == ValueMap.end()) {
          std::vector<SPIRVValue *> Elements;
          for (Type *E : ST->elements())
            Elements.push_back(transValue(UndefValue::get(E), nullptr));
          BVarInit = BM->addCompositeConstant(transType(ST), Elements);
          ValueMap[Init] = BVarInit;
        } else {
          BVarInit = I->second;
        }
      }
    } else if (Init && !isa<UndefValue>(Init)) {
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_long_constant_composite)) {
        if (auto ArrTy = dyn_cast_or_null<ArrayType>(Init->getType())) {
          // First 3 words of OpConstantComposite encode: 1) word count &
          // opcode, 2) Result Type and 3) Result Id. Max length of SPIRV
          // instruction = 65535 words.
          constexpr int MaxNumElements =
              MaxWordCount - SPIRVSpecConstantComposite::FixedWC;
          if (ArrTy->getNumElements() > MaxNumElements &&
              !isa<ConstantAggregateZero>(Init)) {
            std::stringstream SS;
            SS << "Global variable has a constant array initializer with a "
               << "number of elements greater than OpConstantComposite can "
               << "have (" << MaxNumElements << "). Should the array be "
               << "split?\n Original LLVM value:\n"
               << toString(GV);
            getErrorLog().checkError(false, SPIRVEC_InvalidWordCount, SS.str());
          }
        }
      }
      BVarInit = (StorageClass == StorageClassWorkgroup || Init == nullptr
                      ? nullptr
                      : transValue(Init, nullptr));
    }

    // Vulkan/GLSL doesn't do linkage
    auto linkage = (SrcLang != spv::SourceLanguageGLSL
                        ? transLinkageType(GV)
                        : spv::internal::LinkageTypeInternal);

    // vars with Function storage must be added to the entry block
    SPIRVBasicBlock *var_bb = nullptr;
    if (StorageClass == spv::StorageClassFunction) {
      var_bb = BB->getParent()->getBasicBlock(0);
    }
    auto BVar = static_cast<SPIRVVariable *>(
        BM->addVariable(transType(Ty), GV->isConstant(), linkage, BVarInit,
                        GV->getName().str(), StorageClass, var_bb));

    if (IsVectorCompute) {
      BVar->addDecorate(DecorationVectorComputeVariableINTEL);
      if (GV->hasAttribute(kVCMetadata::VCByteOffset)) {
        SPIRVWord Offset = {};
        GV->getAttribute(kVCMetadata::VCByteOffset)
            .getValueAsString()
            .getAsInteger(0, Offset);
        BVar->addDecorate(DecorationGlobalVariableOffsetINTEL, Offset);
      }
      if (GV->hasAttribute(kVCMetadata::VCVolatile))
        BVar->addDecorate(DecorationVolatile);

      if (GV->hasAttribute(kVCMetadata::VCSingleElementVector))
        translateSEVDecoration(
            GV->getAttribute(kVCMetadata::VCSingleElementVector), BVar);
    }

    mapValue(V, BVar);
    if (Ty->isPointerTy()) {
      auto elem_type = Ty->getPointerElementType();
      auto spirv_elem_type = transType(elem_type);
      decorateComposite(elem_type, spirv_elem_type);
    }
    spv::BuiltIn Builtin = spv::BuiltInPosition;
    if (!GV->hasName() || !getSPIRVBuiltin(GV->getName().str(), Builtin))
      return BVar;
    if (static_cast<uint32_t>(Builtin) >= internal::BuiltInSubDeviceIDINTEL &&
        static_cast<uint32_t>(Builtin) <=
            internal::BuiltInMaxHWThreadIDPerSubDeviceINTEL) {
      if (!BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_hw_thread_queries)) {
        std::string ErrorStr = "Intel HW thread queries must be enabled by "
                               "SPV_INTEL_hw_thread_queries extension.\n"
                               "LLVM value that is being translated:\n";
        getErrorLog().checkError(false, SPIRVEC_InvalidModule, V, ErrorStr);
      }
      BM->addExtension(ExtensionID::SPV_INTEL_hw_thread_queries);
    }

    BVar->setBuiltin(Builtin);
    return BVar;
  }

  // always create a new undef variable for vulkan/glsl to workaround driver
  // issues
  if (isa<UndefValue>(V) && SrcLang == spv::SourceLanguageGLSL) {
    if (BB != nullptr) {
      return BM->addUndefInst(transType(V->getType()), BB);
    }
    // if this isn't inside a BB (e.g. inside a constant), just use 0
    return BM->addConstant(transType(V->getType()), 0);
  }

  if (isa<Constant>(V)) {
    auto BV = transConstant(V);
    assert(BV);
    return mapValue(V, BV);
  }

  if (auto Arg = dyn_cast<Argument>(V)) {
    unsigned ArgNo = Arg->getArgNo();
    SPIRVFunction *BF = BB->getParent();
    // assert(BF->existArgument(ArgNo));
    return mapValue(V, BF->getArgument(ArgNo));
  }

  if (CreateForward)
    return mapValue(V, BM->addForward(transType(V->getType())));

  if (StoreInst *ST = dyn_cast<StoreInst>(V)) {
    if (ST->isAtomic())
      return transAtomicStore(ST, BB);

    // Keep this vector to store MemoryAccess operands for both Alignment and
    // Aliasing information.
    std::vector<SPIRVWord> MemoryAccess(1, 0);
    if (ST->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    if (ST->getAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      MemoryAccess.push_back(ST->getAlignment());
    }
    if (ST->getMetadata(LLVMContext::MD_nontemporal))
      MemoryAccess[0] |= MemoryAccessNontemporalMask;
    // always mark global/device pointer with "MakePointerAvailable"
    if (auto addr_space = ST->getPointerAddressSpace();
        SrcLang == spv::SourceLanguageGLSL &&
        (addr_space == SPIRAS_StorageBuffer ||
         addr_space == SPIRAS_PhysicalStorageBuffer)) {
      MemoryAccess[0] |= MemoryAccessMakePointerAvailableMask |
                         MemoryAccessNonPrivatePointerMask;
      MemoryAccess.push_back(
          BM->addIntegerConstant(BM->addIntegerType(32, true), ScopeDevice)
              ->getId());
    }
    if (MDNode *AliasingListMD = ST->getMetadata(LLVMContext::MD_alias_scope))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             internal::MemoryAccessAliasScopeINTELMask);
    if (MDNode *AliasingListMD = ST->getMetadata(LLVMContext::MD_noalias))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             internal::MemoryAccessNoAliasINTELMask);
    if (MemoryAccess.front() == 0)
      MemoryAccess.clear();

    // check if we need to do int <-> uint casting
    auto dst = transValue(ST->getPointerOperand(), BB);
    auto src =
        transValue(ST->getValueOperand(), BB, true, FuncTransMode::Pointer);
    assert(dst->getType()->isTypePointer());
    auto dst_elem_type =
        ((SPIRVTypePointer *)dst->getType())->getPointerElementType();
    auto src_elem_type = src->getType();
    if (dst_elem_type != src_elem_type) {
      bool emit_bitcast = false;
      if (dst_elem_type->isTypeInt()) {
        assert(src_elem_type->isTypeInt());
        if (((SPIRVTypeInt *)dst_elem_type)->isSigned() !=
            ((SPIRVTypeInt *)src_elem_type)->isSigned()) {
          emit_bitcast = true;
        }
      } else if (dst_elem_type->isTypeVectorInt()) {
        assert(src_elem_type->isTypeVectorInt());
        if (((SPIRVTypeInt *)((SPIRVTypeVector *)dst_elem_type)
                 ->getComponentType())
                ->isSigned() !=
            ((SPIRVTypeInt *)((SPIRVTypeVector *)src_elem_type)
                 ->getComponentType())
                ->isSigned()) {
          emit_bitcast = true;
        }
      }
      if (emit_bitcast) {
        src = BM->addUnaryInst(spv::OpBitcast, dst_elem_type, src, BB);
      }
    }

    return mapValue(V, BM->addStoreInst(dst, src, MemoryAccess, BB));
  }

  if (LoadInst *LD = dyn_cast<LoadInst>(V)) {
    if (LD->isAtomic())
      return transAtomicLoad(LD, BB);

    // Keep this vector to store MemoryAccess operands for both Alignment and
    // Aliasing information.
    std::vector<uint32_t> MemoryAccess(1, 0);
    if (LD->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    if (LD->getAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      MemoryAccess.push_back(LD->getAlignment());
    }
    if (LD->getMetadata(LLVMContext::MD_nontemporal))
      MemoryAccess[0] |= MemoryAccessNontemporalMask;
    // always mark global/device pointer with "MakePointerVisible"
    if (auto addr_space = LD->getPointerAddressSpace();
        SrcLang == spv::SourceLanguageGLSL &&
        (addr_space == SPIRAS_StorageBuffer ||
         addr_space == SPIRAS_PhysicalStorageBuffer)) {
      MemoryAccess[0] |= MemoryAccessMakePointerVisibleMask |
                         MemoryAccessNonPrivatePointerMask;
      MemoryAccess.push_back(
          BM->addIntegerConstant(BM->addIntegerType(32, true), ScopeDevice)
              ->getId());
    }
    if (MDNode *AliasingListMD = LD->getMetadata(LLVMContext::MD_alias_scope))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             internal::MemoryAccessAliasScopeINTELMask);
    if (MDNode *AliasingListMD = LD->getMetadata(LLVMContext::MD_noalias))
      transAliasingMemAccess(BM, AliasingListMD, MemoryAccess,
                             internal::MemoryAccessNoAliasINTELMask);
    if (MemoryAccess.front() == 0)
      MemoryAccess.clear();
    return mapValue(V, BM->addLoadInst(transValue(LD->getPointerOperand(), BB),
                                       MemoryAccess, BB));
  }

  if (BinaryOperator *B = dyn_cast<BinaryOperator>(V)) {
    SPIRVInstruction *BI = transBinaryInst(B, BB);
    return mapValue(V, BI);
  }

  if (dyn_cast<UnreachableInst>(V)) {
    // TODO: fix this be either creating a llvm OpKill instruction or metadata
    if (ignore_next_unreachable) {
      ignore_next_unreachable = false;
      return nullptr;
    }
    return mapValue(V, BM->addUnreachableInst(BB));
  }

  if (auto RI = dyn_cast<ReturnInst>(V)) {
    if (auto RV = RI->getReturnValue())
      return mapValue(V, BM->addReturnValueInst(transValue(RV, BB), BB));
    return mapValue(V, BM->addReturnInst(BB));
  }

  if (CmpInst *Cmp = dyn_cast<CmpInst>(V)) {
    SPIRVInstruction *BI = transCmpInst(Cmp, BB);
    return mapValue(V, BI);
  }

  if (SelectInst *Sel = dyn_cast<SelectInst>(V))
    return mapValue(
        V,
        BM->addSelectInst(
            transValue(Sel->getCondition(), BB),
            transValue(Sel->getTrueValue(), BB, true, FuncTransMode::Pointer),
            transValue(Sel->getFalseValue(), BB, true, FuncTransMode::Pointer),
            BB));

  if (AllocaInst *Alc = dyn_cast<AllocaInst>(V)) {
    if (Alc->isArrayAllocation()) {
      if (!BM->checkExtension(ExtensionID::SPV_INTEL_variable_length_array,
                              SPIRVEC_InvalidInstruction,
                              toString(Alc) +
                                  "\nTranslation of dynamic alloca requires "
                                  "SPV_INTEL_variable_length_array extension."))
        return nullptr;

      SPIRVValue *Length = transValue(Alc->getArraySize(), BB);
      assert(Length && "Couldn't translate array size!");
      return mapValue(V, BM->addInstTemplate(OpVariableLengthArrayINTEL,
                                             {Length->getId()}, BB,
                                             transType(Alc->getType())));
    }
    return mapValue(V, BM->addVariable(transType(Alc->getType()), false,
                                       spv::internal::LinkageTypeInternal,
                                       nullptr, Alc->getName().str(),
                                       StorageClassFunction, BB));
  }

  if (auto *Switch = dyn_cast<SwitchInst>(V)) {
    std::vector<SPIRVSwitch::PairTy> Pairs;
    auto Select = transValue(Switch->getCondition(), BB);

    for (auto I = Switch->case_begin(), E = Switch->case_end(); I != E; ++I) {
      SPIRVSwitch::LiteralTy Lit;
      uint64_t CaseValue = I->getCaseValue()->getZExtValue();

      Lit.push_back(CaseValue);
      assert(Select->getType()->getBitWidth() <= 64 &&
             "unexpected selector bitwidth");
      if (Select->getType()->getBitWidth() == 64)
        Lit.push_back(CaseValue >> 32);

      Pairs.push_back(
          std::make_pair(Lit, static_cast<SPIRVBasicBlock *>(
                                  transValue(I->getCaseSuccessor(), nullptr))));
    }

    return mapValue(
        V, BM->addSwitchInst(Select,
                             static_cast<SPIRVBasicBlock *>(
                                 transValue(Switch->getDefaultDest(), nullptr)),
                             Pairs, BB));
  }

  if (BranchInst *Branch = dyn_cast<BranchInst>(V)) {
    SPIRVLabel *SuccessorTrue =
        static_cast<SPIRVLabel *>(transValue(Branch->getSuccessor(0), BB));

    /// Clang attaches !llvm.loop metadata to "latch" BB. This kind of blocks
    /// has an edge directed to the loop header. Thus latch BB matching to
    /// "Continue Target" per the SPIR-V spec. This statement is true only after
    /// applying the loop-simplify pass to the LLVM module.
    /// For "for" and "while" loops latch BB is terminated by an
    /// unconditional branch. Also for this kind of loops "Merge Block" can
    /// be found as block targeted by false edge of the "Header" BB.
    /// For "do while" loop the latch is terminated by a conditional branch
    /// with true edge going to the header and the false edge going out of
    /// the loop, which corresponds to a "Merge Block" per the SPIR-V spec.
    std::vector<SPIRVWord> Parameters;
    spv::LoopControlMask LoopControl = getLoopControl(Branch, Parameters);

    if (Branch->isUnconditional()) {
      // Usually, "for" and "while" loops llvm.loop metadata is attached to an
      // unconditional branch instruction.
      if (LoopControl != spv::LoopControlMaskNone) {
        // SuccessorTrue is the loop header BB.
        const SPIRVInstruction *Term = SuccessorTrue->getTerminateInstr();
        if (Term && Term->getOpCode() == OpBranchConditional) {
          const auto *Br = static_cast<const SPIRVBranchConditional *>(Term);
          BM->addLoopMergeInst(Br->getFalseLabel()->getId(), // Merge Block
                               BB->getId(),                  // Continue Target
                               LoopControl, Parameters, SuccessorTrue);
        } else {
          if (BM->isAllowedToUseExtension(
                  ExtensionID::SPV_INTEL_unstructured_loop_controls)) {
            // For unstructured loop we add a special loop control instruction.
            // Simple example of unstructured loop is an infinite loop, that has
            // no terminate instruction.
            BM->addLoopControlINTELInst(LoopControl, Parameters, SuccessorTrue);
          }
        }
      }
      return mapValue(V, BM->addBranchInst(SuccessorTrue, BB));
    }
    // For "do-while" (and in some cases, for "for" and "while") loops,
    // llvm.loop metadata is attached to a conditional branch instructions
    SPIRVLabel *SuccessorFalse =
        static_cast<SPIRVLabel *>(transValue(Branch->getSuccessor(1), BB));
    if (LoopControl != spv::LoopControlMaskNone) {
      Function *Fun = Branch->getFunction();
      DominatorTree DomTree(*Fun);
      LoopInfo LI(DomTree);
      for (const auto *LoopObj : LI.getLoopsInPreorder()) {
        // Check whether SuccessorFalse or SuccessorTrue is the loop header BB.
        // For example consider following LLVM IR:
        // br i1 %compare, label %for.body, label %for.end
        //   <- SuccessorTrue is 'for.body' aka successor(0)
        // br i1 %compare.not, label %for.end, label %for.body
        //   <- SuccessorTrue is 'for.end' aka successor(1)
        // meanwhile the true successor (by definition) should be a loop header
        // aka 'for.body'
        if (LoopObj->getHeader() == Branch->getSuccessor(1))
          // SuccessorFalse is the loop header BB.
          BM->addLoopMergeInst(SuccessorTrue->getId(), // Merge Block
                               BB->getId(),            // Continue Target
                               LoopControl, Parameters, SuccessorFalse);
        else
          // SuccessorTrue is the loop header BB.
          BM->addLoopMergeInst(SuccessorFalse->getId(), // Merge Block
                               BB->getId(),             // Continue Target
                               LoopControl, Parameters, SuccessorTrue);
      }
    }
    return mapValue(
        V, BM->addBranchConditionalInst(transValue(Branch->getCondition(), BB),
                                        SuccessorTrue, SuccessorFalse, BB));
  }

  if (auto Phi = dyn_cast<PHINode>(V)) {
    // NOTE: LLVM has the tendency to allow duplicate predecessors and PHI
    // incoming blocks
    // -> ensure we only add incoming values for unique predecessors
    // NOTE: also, we need to ensure all values have the same type, or fail
    // otherwise

    std::unordered_set<BasicBlock *> unique_llvm_bbs;
    std::vector<std::pair<llvm::Value *, llvm::BasicBlock *>>
        incoming_llvm_pairs;
    for (size_t I = 0, E = Phi->getNumIncomingValues(); I != E; ++I) {
      auto in_bb = Phi->getIncomingBlock(I);
      if (unique_llvm_bbs.count(in_bb) > 0) {
        continue;
      }
      unique_llvm_bbs.emplace(in_bb);
      incoming_llvm_pairs.emplace_back(Phi->getIncomingValue(I),
                                       Phi->getIncomingBlock(I));
    }

    llvm::Type *common_non_const_type = nullptr;
    for (const auto &incoming : incoming_llvm_pairs) {
      if (auto const_val = dyn_cast_or_null<Constant>(incoming.first);
          const_val) {
        continue;
      }
      if (!common_non_const_type) {
        common_non_const_type = incoming.first->getType();
        continue;
      }
      if (common_non_const_type != incoming.first->getType()) {
        BM->getErrorLog().checkError(
            false, SPIRVErrorCode::SPIRVEC_InvalidInstruction, incoming.first,
            "PHI type mismatch: all non-const PHI value types must be "
            "identical");
        return nullptr;
      }
    }
    if (common_non_const_type && (!common_non_const_type->isIntegerTy() ||
                                  common_non_const_type->isIntegerTy(1))) {
      // ignore this for all non-integer and bool types
      // TODO: also int vector types ...
      common_non_const_type = nullptr;
    }

    std::vector<SPIRVValue *> IncomingPairs;
    // add all non-const pairs first, or everything if we have no common
    // non-const type
    for (auto &incoming : incoming_llvm_pairs) {
      if (!common_non_const_type ||
          dyn_cast_or_null<Constant>(incoming.first) == nullptr) {
        IncomingPairs.push_back(
            transValue(incoming.first, BB, true, FuncTransMode::Pointer));
        IncomingPairs.push_back(transValue(incoming.second, nullptr));
        continue;
      }
    }
    // handle all constant and make sure they have the same type as the
    // non-const type
    if (common_non_const_type) {
      assert(!IncomingPairs.empty() &&
             "IncomingPairs should not be empty at this point");
      const auto &non_const_spirv_type = IncomingPairs[0]->getType();
      assert(non_const_spirv_type->isTypeInt());
      const auto non_const_spirv_int_type =
          (const SPIRVTypeInt *)non_const_spirv_type;
      for (auto &incoming : incoming_llvm_pairs) {
        auto const_val = dyn_cast_or_null<Constant>(incoming.first);
        if (!const_val) {
          continue;
        }
        // constant value type mismatch
        // -> cast to correct type
        if (auto const_int = dyn_cast<ConstantInt>(incoming.first); const_int) {
          auto int_type =
              BM->addIntegerType(non_const_spirv_int_type->getBitWidth(),
                                 non_const_spirv_int_type->isSigned());
          IncomingPairs.push_back(
              BM->addIntegerConstant(int_type, const_int->getZExtValue()));
        } else {
          // fallback
          IncomingPairs.push_back(
              transValue(incoming.first, BB, true, FuncTransMode::Pointer));
        }
        IncomingPairs.push_back(transValue(incoming.second, nullptr));
      }
    }

    assert(!common_non_const_type || (common_non_const_type == Phi->getType()));
    return mapValue(
        V, BM->addPhiInst(IncomingPairs[0]->getType(), IncomingPairs, BB));
  }

  if (auto Ext = dyn_cast<ExtractValueInst>(V)) {
    return mapValue(V, BM->addCompositeExtractInst(
                           transType(Ext->getType()),
                           transValue(Ext->getAggregateOperand(), BB),
                           Ext->getIndices(), BB));
  }

  if (auto Ins = dyn_cast<InsertValueInst>(V)) {
    return mapValue(V, BM->addCompositeInsertInst(
                           transValue(Ins->getInsertedValueOperand(), BB),
                           transValue(Ins->getAggregateOperand(), BB),
                           Ins->getIndices(), BB));
  }

  if (UnaryInstruction *U = dyn_cast<UnaryInstruction>(V)) {
    if (isSpecialTypeInitializer(U))
      return mapValue(V, transValue(U->getOperand(0), BB));
    auto UI = transUnaryInst(U, BB);
    return mapValue(V, UI ? UI : transValue(U->getOperand(0), BB));
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    std::vector<SPIRVValue *> Indices;
    for (unsigned I = 0, E = GEP->getNumIndices(); I != E; ++I)
      Indices.push_back(transValue(GEP->getOperand(I + 1), BB));
    if (SrcLang != spv::SourceLanguageGLSL) {
      auto *PointerOperand = GEP->getPointerOperand();
      auto *TransPointerOperand = transValue(PointerOperand, BB);

      // Certain array-related optimization hints can be expressed via
      // LLVM metadata. For the purpose of linking this metadata with
      // the accessed array variables, our GEP may have been marked into
      // a so-called index group, an MDNode by itself.
      if (MDNode *IndexGroup = GEP->getMetadata("llvm.index.group")) {
        SPIRVValue *ActualMemoryPtr = TransPointerOperand;
        if (auto *Load = dyn_cast<LoadInst>(PointerOperand)) {
          ActualMemoryPtr = transValue(Load->getPointerOperand(), BB);
        }
        SPIRVId AccessedArrayId = ActualMemoryPtr->getId();
        unsigned NumOperands = IndexGroup->getNumOperands();
        // When we're working with embedded loops, it's natural that
        // the outer loop's hints apply to all code contained within.
        // The inner loop's specific hints, however, should stay private
        // to the inner loop's scope.
        // Consequently, the following division of the index group metadata
        // nodes emerges:

        // 1) The metadata node has no operands. It will be directly referenced
        //    from within the optimization hint metadata.
        if (NumOperands == 0)
          IndexGroupArrayMap[IndexGroup].insert(AccessedArrayId);
        // 2) The metadata node has several operands. It serves to link an index
        //    group specific to some embedded loop with other index groups that
        //    mark the same array variable for the outer loop(s).
        for (unsigned I = 0; I < NumOperands; ++I) {
          auto *ContainedIndexGroup = getMDOperandAsMDNode(IndexGroup, I);
          IndexGroupArrayMap[ContainedIndexGroup].insert(AccessedArrayId);
        }
      }

      return mapValue(V, BM->addPtrAccessChainInst(transType(GEP->getType()),
                                                   TransPointerOperand, Indices,
                                                   BB, GEP->isInBounds()));
    } else {
      // with variable pointers we can now use PtrAccessChain instead of the
      // simple AccessChain (for SSBOs, local memory and physical SSBOs)
      auto gep_type = transType(GEP->getType());
      auto gep_value = transValue(GEP->getPointerOperand(), BB);
      auto gep_value_type = gep_value->getType();
      const auto storage_class = gep_value_type->getPointerStorageClass();
      if (storage_class == spv::StorageClassWorkgroup ||
          storage_class == spv::StorageClassPhysicalStorageBuffer ||
          storage_class == spv::StorageClassStorageBuffer) {
        // must still treat access to SSBO runtime arrays specially by adding
        // two additional 0 indices
        if (storage_class == spv::StorageClassStorageBuffer) {
          auto gep_value_elem_type = gep_value_type->getPointerElementType();
          if (gep_value_elem_type->isTypeStruct()) {
            auto gep_struct_type =
                (SPIRV::SPIRVTypeStruct *)gep_value_elem_type;
            if (gep_struct_type->getMemberCount() > 0 &&
                gep_struct_type->getMemberType(0)->isTypeRuntimeArray()) {
              auto zero_const = transValue(
                  llvm::ConstantInt::get(llvm::Type::getInt32Ty(*Ctx), 0), BB);
              Indices.insert(Indices.begin(), zero_const);
              Indices.insert(Indices.begin(), zero_const);
            }
          }
        }

        // fun additional requirement: base ptr (gep_value) must have been
        // decorated with ArrayStride -> this usually already has been done when
        // directly using a function input, but will not be the case for GEP
        // chains that couldn't be fused (e.g. when used via PHIs)
        if (auto array_stride_iter = base_array_strides.find(gep_value_type);
            array_stride_iter == base_array_strides.end()) {
          // need to compute and add a stride for this base: base must point to
          // a sized type -> will use that as the stride
          auto pointee_type =
              GEP->getPointerOperand()->getType()->getPointerElementType();
          assert(pointee_type->isSized());
          auto stride = M->getDataLayout().getTypeStoreSize(pointee_type);
          add_array_stride_decoration(gep_value_type, stride);
        }

        return mapValue(
            V, BM->addPtrAccessChainInst(gep_type, gep_value, Indices, BB,
                                         false /* never emit inbounds */));
      } else {
        // for all other storage classes: fall back to (Inbounds)AccessChain
        return mapValue(
            V, BM->addAccessChainInst(transType(GEP->getType()),
                                      transValue(GEP->getPointerOperand(), BB),
                                      Indices, BB, GEP->isInBounds()));
      }
    }
  }

  if (auto Ext = dyn_cast<ExtractElementInst>(V)) {
    auto Index = Ext->getIndexOperand();
    if (auto Const = dyn_cast<ConstantInt>(Index)) {
      auto val = transValue(Ext->getVectorOperand(), BB);
      SPIRVType *type = nullptr;
      if (val->getType()->isTypeVector()) {
        // assume component type that we've already mapped
        type = val->getType()->getVectorComponentType();
      } else {
        type = transType(Ext->getType());
      }
      return mapValue(
          V,
          BM->addCompositeExtractInst(
              type, val, std::vector<SPIRVWord>(1, Const->getZExtValue()), BB));
    } else {
      return mapValue(V, BM->addVectorExtractDynamicInst(
                             transValue(Ext->getVectorOperand(), BB),
                             transValue(Index, BB), BB));
    }
  }

  if (auto Ins = dyn_cast<InsertElementInst>(V)) {
    auto Index = Ins->getOperand(2);
    if (auto Const = dyn_cast<ConstantInt>(Index)) {
      return mapValue(
          V,
          BM->addCompositeInsertInst(
              transValue(Ins->getOperand(1), BB, true, FuncTransMode::Pointer),
              transValue(Ins->getOperand(0), BB),
              std::vector<SPIRVWord>(1, Const->getZExtValue()), BB));
    } else
      return mapValue(
          V, BM->addVectorInsertDynamicInst(transValue(Ins->getOperand(0), BB),
                                            transValue(Ins->getOperand(1), BB),
                                            transValue(Index, BB), BB));
  }

  if (auto SF = dyn_cast<ShuffleVectorInst>(V)) {
    std::vector<SPIRVWord> Comp;
    for (auto &I : SF->getShuffleMask())
      Comp.push_back(I);
    return mapValue(V, BM->addVectorShuffleInst(
                           transType(SF->getType()),
                           transValue(SF->getOperand(0), BB),
                           transValue(SF->getOperand(1), BB), Comp, BB));
  }

  if (AtomicRMWInst *ARMW = dyn_cast<AtomicRMWInst>(V)) {
    AtomicRMWInst::BinOp Op = ARMW->getOperation();
    if (!BM->getErrorLog().checkError(
            !AtomicRMWInst::isFPOperation(Op) && Op != AtomicRMWInst::Nand,
            SPIRVEC_InvalidInstruction, V,
            "Atomic " + AtomicRMWInst::getOperationName(Op).str() +
                " is not supported in SPIR-V!\n"))
      return nullptr;

    spv::Op OC = LLVMSPIRVAtomicRmwOpCodeMap::map(Op);
    AtomicOrderingCABI Ordering = llvm::toCABI(ARMW->getOrdering());
    auto MemSem = OCLMemOrderMap::map(static_cast<OCLMemOrderKind>(Ordering));
    std::vector<Value *> Operands(4);
    Operands[0] = ARMW->getPointerOperand();
    // To get the memory scope argument we might use ARMW->getSyncScopeID(), but
    // atomicrmw LLVM instruction is not aware of OpenCL(or SPIR-V) memory scope
    // enumeration. And assuming the produced SPIR-V module will be consumed in
    // an OpenCL environment, we can use the same memory scope as OpenCL atomic
    // functions that don't have memory_scope argument i.e. memory_scope_device.
    // See the OpenCL C specification p6.13.11. "Atomic Functions"
    Operands[1] = getUInt32(M, spv::ScopeDevice);
    Operands[2] = getUInt32(M, MemSem);
    Operands[3] = ARMW->getValOperand();
    std::vector<SPIRVId> Ops = BM->getIds(transValue(Operands, BB));
    SPIRVType *Ty = transType(ARMW->getType());

    return mapValue(V, BM->addInstTemplate(OC, Ops, BB, Ty));
  }

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(V)) {
    SPIRVValue *BV = transIntrinsicInst(II, BB);
    return BV ? mapValue(V, BV) : nullptr;
  }

  if (InlineAsm *IA = dyn_cast<InlineAsm>(V))
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_inline_assembly))
      return mapValue(V, transAsmINTEL(IA));

  if (CallInst *CI = dyn_cast<CallInst>(V)) {
    if (auto Alias =
            dyn_cast_or_null<llvm::GlobalAlias>(CI->getCalledOperand())) {
      CI->setCalledFunction(cast<Function>(Alias->getAliasee()));
    }
    return mapValue(V, transCallInst(CI, BB));
  }

  if (Instruction *Inst = dyn_cast<Instruction>(V)) {
    BM->SPIRVCK(false, InvalidInstruction, toString(Inst));
  }

  errs() << "not implemented: " << *V << "\n";
  llvm_unreachable("Not implemented");
  return nullptr;
}

SPIRVType *LLVMToSPIRVBase::mapType(Type *T, SPIRVType *BT) {
  auto EmplaceStatus = TypeMap.try_emplace(T, BT);
  // TODO: Uncomment the assertion, once the type mapping issue is resolved
  // assert(EmplaceStatus.second && "The type was already added to the map");
  SPIRVDBG(dbgs() << "[mapType] " << *T << " => "; spvdbgs() << *BT << '\n');
  if (!EmplaceStatus.second)
    return TypeMap[T];
  return BT;
}

SPIRVValue *LLVMToSPIRVBase::mapValue(const Value *V, SPIRVValue *BV) {
  auto Loc = ValueMap.find(V);
  if (Loc != ValueMap.end()) {
    if (Loc->second == BV)
      return BV;
    assert(Loc->second->isForward() &&
           "LLVM Value is mapped to different SPIRV Values");
    auto Forward = static_cast<SPIRVForward *>(Loc->second);
    BM->replaceForward(Forward, BV);
  }
  ValueMap[V] = BV;
  SPIRVDBG(dbgs() << "[mapValue] " << *V << " => "; spvdbgs() << BV << "\n");
  return BV;
}

bool LLVMToSPIRVBase::shouldTryToAddMemAliasingDecoration(Instruction *Inst) {
  // Limit translation of aliasing metadata with only this set of instructions
  // gracefully considering others as compilation mistakes and ignoring them
  if (!Inst->mayReadOrWriteMemory())
    return false;
  // Loads and Stores are handled during memory access mask addition
  if (isa<StoreInst>(Inst) || isa<LoadInst>(Inst))
    return false;
  CallInst *CI = dyn_cast<CallInst>(Inst);
  if (!CI)
    return true;
  if (Function *Fun = CI->getCalledFunction()) {
    // Calls to intrinsics are skipped. At some point lifetime start/end will be
    // handled separately, but specification isn't ready.
    if (Fun->isIntrinsic())
      return false;
    // Also skip SPIR-V instructions that don't have result id to attach the
    // decorations
    if (isBuiltinTransToInst(Fun))
      if (Fun->getReturnType()->isVoidTy())
        return false;
  }
  return true;
}

bool LLVMToSPIRVBase::transDecoration(Value *V, SPIRVValue *BV) {
  if (!transAlign(V, BV))
    return false;
  if ((isa<AtomicCmpXchgInst>(V) && cast<AtomicCmpXchgInst>(V)->isVolatile()) ||
      (isa<AtomicRMWInst>(V) && cast<AtomicRMWInst>(V)->isVolatile()))
    BV->setVolatile(true);

  if (auto BVO = dyn_cast_or_null<OverflowingBinaryOperator>(V)) {
    if (BVO->hasNoSignedWrap()) {
      BV->setNoIntegerDecorationWrap<DecorationNoSignedWrap>(true);
    }
    if (BVO->hasNoUnsignedWrap()) {
      BV->setNoIntegerDecorationWrap<DecorationNoUnsignedWrap>(true);
    }
  }

  if (auto BVF = dyn_cast_or_null<FPMathOperator>(V);
      BVF && SrcLang != spv::SourceLanguageGLSL) {
    auto Opcode = BVF->getOpcode();
    if (Opcode == Instruction::FAdd || Opcode == Instruction::FSub ||
        Opcode == Instruction::FMul || Opcode == Instruction::FDiv ||
        Opcode == Instruction::FRem) {
      FastMathFlags FMF = BVF->getFastMathFlags();
      SPIRVWord M{0};
      if (FMF.isFast())
        M |= FPFastMathModeFastMask;
      else {
        if (FMF.noNaNs())
          M |= FPFastMathModeNotNaNMask;
        if (FMF.noInfs())
          M |= FPFastMathModeNotInfMask;
        if (FMF.noSignedZeros())
          M |= FPFastMathModeNSZMask;
        if (FMF.allowReciprocal())
          M |= FPFastMathModeAllowRecipMask;
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_fp_fast_math_mode)) {
          if (FMF.allowContract()) {
            M |= FPFastMathModeAllowContractFastINTELMask;
            BM->addCapability(CapabilityFPFastMathModeINTEL);
          }
          if (FMF.allowReassoc()) {
            M |= FPFastMathModeAllowReassocINTELMask;
            BM->addCapability(CapabilityFPFastMathModeINTEL);
          }
        }
      }
      if (M != 0)
        BV->setFPFastMathMode(M);
    }
  }
  if (Instruction *Inst = dyn_cast<Instruction>(V))
    if (shouldTryToAddMemAliasingDecoration(Inst))
      transMemAliasingINTELDecorations(Inst, BV);

  if (auto *CI = dyn_cast<CallInst>(V)) {
    auto OC = BV->getOpCode();
    if (OC == OpSpecConstantTrue || OC == OpSpecConstantFalse ||
        OC == OpSpecConstant) {
      auto SpecId = cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
      BV->addDecorate(DecorationSpecId, SpecId);
    }
  }

  return true;
}

bool LLVMToSPIRVBase::transAlign(Value *V, SPIRVValue *BV) {
  // shader doesn't have the alignment decoration -> just return
  if (SrcLang == spv::SourceLanguageGLSL) {
    return true;
  }

  if (auto AL = dyn_cast<AllocaInst>(V)) {
    BM->setAlignment(BV, AL->getAlignment());
    return true;
  }
  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    BM->setAlignment(BV, GV->getAlignment());
    return true;
  }
  return true;
}

// Apply aliasing decorations to instructions annotated with aliasing metadata.
// Do it for any instruction but loads and stores.
void LLVMToSPIRVBase::transMemAliasingINTELDecorations(Instruction *Inst,
                                                       SPIRVValue *BV) {
  if (!BM->isAllowedToUseExtension(
          ExtensionID::SPV_INTEL_memory_access_aliasing))
    return;
  if (MDNode *AliasingListMD = Inst->getMetadata(LLVMContext::MD_alias_scope)) {
    auto *MemAliasList = addMemAliasingINTELInstructions(BM, AliasingListMD);
    if (!MemAliasList)
      return;
    BV->addDecorate(new SPIRVDecorateId(internal::DecorationAliasScopeINTEL, BV,
                                        MemAliasList->getId()));
  }
  if (MDNode *AliasingListMD = Inst->getMetadata(LLVMContext::MD_noalias)) {
    auto *MemAliasList = addMemAliasingINTELInstructions(BM, AliasingListMD);
    if (!MemAliasList)
      return;
    BV->addDecorate(new SPIRVDecorateId(internal::DecorationNoAliasINTEL, BV,
                                        MemAliasList->getId()));
  }
}

/// Do this after source language is set.
bool LLVMToSPIRVBase::transBuiltinSet() {
  SPIRVWord Ver = 0;
  SourceLanguage Kind = BM->getSourceLanguage(&Ver);
  assert((Kind == SourceLanguageOpenCL_C || Kind == SourceLanguageOpenCL_CPP ||
          Kind == SourceLanguageGLSL) &&
         "not supported");

  SPIRVId EISId;
  if (Kind != SourceLanguageGLSL) {
    if (!BM->importBuiltinSet("OpenCL.std", &EISId))
      return false;
  } else {
    if (!BM->importBuiltinSet("GLSL.std.450", &EISId))
      return false;
  }
  if (SPIRVMDWalker(*M).getNamedMD("llvm.dbg.cu")) {
    if (BM->getDebugInfoEIS() == SPIRVEIS_GLSL) {
      return true;
    }
    if (!BM->importBuiltinSet(
            SPIRVBuiltinSetNameMap::map(BM->getDebugInfoEIS()), &EISId))
      return false;
  }
  return true;
}

/// Transforms SPV-IR work-item builtin calls to SPIRV builtin variables.
/// e.g.
///  SPV-IR: @_Z33__spirv_BuiltInGlobalInvocationIdi(i)
///    is transformed as:
///  x = load GlobalInvocationId; extract x, i
/// e.g.
///  SPV-IR: @_Z22__spirv_BuiltInWorkDim()
///    is transformed as:
///  load WorkDim
bool LLVMToSPIRVBase::transWorkItemBuiltinCallsToVariables() {
  LLVM_DEBUG(dbgs() << "Enter transWorkItemBuiltinCallsToVariables\n");
  // Store instructions and functions that need to be removed.
  SmallVector<Value *, 16> ToRemove;
  for (auto &F : *M) {
    // Builtins should be declaration only.
    if (!F.isDeclaration())
      continue;
    StringRef DemangledName;
    if (!oclIsBuiltin(F.getName(), DemangledName))
      continue;
    LLVM_DEBUG(dbgs() << "Function demangled name: " << DemangledName << '\n');
    SmallVector<StringRef, 2> Postfix;
    // Deprefix "__spirv_"
    StringRef Name = dePrefixSPIRVName(DemangledName, Postfix);
    // Lookup SPIRV Builtin map.
    if (!SPIRVBuiltInNameMap::rfind(Name.str(), nullptr))
      continue;
    std::string BuiltinVarName = DemangledName.str();
    LLVM_DEBUG(dbgs() << "builtin variable name: " << BuiltinVarName << '\n');
    bool IsVec = F.getFunctionType()->getNumParams() > 0;
    Type *GVType =
        IsVec ? FixedVectorType::get(F.getReturnType(), 3) : F.getReturnType();
    auto *BV = new GlobalVariable(
        *M, GVType, /*isConstant=*/true, GlobalValue::ExternalLinkage, nullptr,
        BuiltinVarName, 0, GlobalVariable::NotThreadLocal, SPIRAS_Input);
    for (auto *U : F.users()) {
      auto *CI = dyn_cast<CallInst>(U);
      assert(CI && "invalid instruction");
      const DebugLoc &DLoc = CI->getDebugLoc();
      Instruction *NewValue = new LoadInst(GVType, BV, "", CI);
      if (DLoc)
        NewValue->setDebugLoc(DLoc);
      LLVM_DEBUG(dbgs() << "Transform: " << *CI << " => " << *NewValue << '\n');
      if (IsVec) {
        NewValue =
            ExtractElementInst::Create(NewValue, CI->getArgOperand(0), "", CI);
        if (DLoc)
          NewValue->setDebugLoc(DLoc);
        LLVM_DEBUG(dbgs() << *NewValue << '\n');
      }
      NewValue->takeName(CI);
      CI->replaceAllUsesWith(NewValue);
      ToRemove.push_back(CI);
    }
    ToRemove.push_back(&F);
  }
  for (auto *V : ToRemove) {
    if (auto *I = dyn_cast<Instruction>(V))
      I->eraseFromParent();
    else if (auto *F = dyn_cast<Function>(V))
      F->eraseFromParent();
    else
      llvm_unreachable("Unexpected value to remove!");
  }
  return true;
}

/// Translate sampler* spcv.cast(i32 arg) or
/// sampler* __translate_sampler_initializer(i32 arg)
/// Three cases are possible:
///   arg = ConstantInt x -> SPIRVConstantSampler
///   arg = i32 argument -> transValue(arg)
///   arg = load from sampler -> look through load
SPIRVValue *LLVMToSPIRVBase::oclTransSpvcCastSampler(CallInst *CI,
                                                     SPIRVBasicBlock *BB) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  llvm::Function *F = CI->getCalledFunction();
  auto FT = F->getFunctionType();
  auto RT = FT->getReturnType();
  assert(FT->getNumParams() == 1);
  assert((isSPIRVType(RT, kSPIRVTypeName::Sampler) ||
          isPointerToOpaqueStructType(RT, kSPR2TypeName::Sampler)) &&
         FT->getParamType(0)->isIntegerTy() && "Invalid sampler type");
  auto Arg = CI->getArgOperand(0);

  auto GetSamplerConstant = [&](uint64_t SamplerValue) {
    auto AddrMode = (SamplerValue & 0xE) >> 1;
    auto Param = SamplerValue & 0x1;
    auto Filter = SamplerValue ? ((SamplerValue & 0x30) >> 4) - 1 : 0;
    auto BV = BM->addSamplerConstant(transType(RT), AddrMode, Param, Filter);
    return BV;
  };

  if (auto Const = dyn_cast<ConstantInt>(Arg)) {
    // Sampler is declared as a kernel scope constant
    return GetSamplerConstant(Const->getZExtValue());
  } else if (auto Load = dyn_cast<LoadInst>(Arg)) {
    // If value of the sampler is loaded from a global constant, use its
    // initializer for initialization of the sampler.
    auto Op = Load->getPointerOperand();
    assert(isa<GlobalVariable>(Op) && "Unknown sampler pattern!");
    auto GV = cast<GlobalVariable>(Op);
    assert(GV->isConstant() ||
           GV->getType()->getPointerAddressSpace() == SPIRAS_Constant);
    auto Initializer = GV->getInitializer();
    assert(isa<ConstantInt>(Initializer) && "sampler not constant int?");
    return GetSamplerConstant(cast<ConstantInt>(Initializer)->getZExtValue());
  }
  // Sampler is a function argument
  auto BV = transValue(Arg, BB);
  assert(BV && BV->getType() == transType(RT));
  return BV;
}

using DecorationsInfoVec = std::vector<std::pair<Decoration, std::string>>;

struct AnnotationDecorations {
  DecorationsInfoVec MemoryAttributesVec;
  DecorationsInfoVec MemoryAccessesVec;
};

struct IntelLSUControlsInfo {
  void setWithBitMask(unsigned ParamsBitMask) {
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::BurstCoalesce)
      BurstCoalesce = true;
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::CacheSizeFlag)
      CacheSizeInfo = 0;
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::DontStaticallyCoalesce)
      DontStaticallyCoalesce = true;
    if (ParamsBitMask & IntelFPGAMemoryAccessesVal::PrefetchFlag)
      PrefetchInfo = 0;
  }

  DecorationsInfoVec getDecorationsFromCurrentState() {
    DecorationsInfoVec ResultVec;
    // Simple flags
    if (BurstCoalesce)
      ResultVec.emplace_back(DecorationBurstCoalesceINTEL, "");
    if (DontStaticallyCoalesce)
      ResultVec.emplace_back(DecorationDontStaticallyCoalesceINTEL, "");
    // Conditional values
    if (CacheSizeInfo.hasValue()) {
      ResultVec.emplace_back(DecorationCacheSizeINTEL,
                             std::to_string(CacheSizeInfo.getValue()));
    }
    if (PrefetchInfo.hasValue()) {
      ResultVec.emplace_back(DecorationPrefetchINTEL,
                             std::to_string(PrefetchInfo.getValue()));
    }
    return ResultVec;
  }

  bool BurstCoalesce = false;
  llvm::Optional<unsigned> CacheSizeInfo;
  bool DontStaticallyCoalesce = false;
  llvm::Optional<unsigned> PrefetchInfo;
};

// Handle optional var/ptr/global annotation parameter. It can be for example
// { %struct.S, i8*, void ()* } { %struct.S undef, i8* null,
//                                void ()* @_Z4blahv }
// Now we will just handle integer constants (wrapped in a constant
// struct, that is being bitcasted to i8*), converting them to string.
// TODO: remove this workaround when/if an extension spec that allows or adds
// variadic-arguments UserSemantic decoration
void processOptionalAnnotationInfo(Constant *Const,
                                   std::string &AnnotationString) {
  if (!Const->getNumOperands())
    return;
  if (auto *CStruct = dyn_cast<ConstantStruct>(Const->getOperand(0))) {
    uint32_t NumOperands = CStruct->getNumOperands();
    if (!NumOperands)
      return;
    if (auto *CInt = dyn_cast<ConstantInt>(CStruct->getOperand(0))) {
      AnnotationString += ": ";
      AnnotationString += std::to_string(CInt->getSExtValue());
    }
    for (uint32_t I = 1; I != NumOperands; ++I) {
      if (auto *CInt = dyn_cast<ConstantInt>(CStruct->getOperand(I))) {
        AnnotationString += ", ";
        AnnotationString += std::to_string(CInt->getSExtValue());
      }
    }
  }
}

// Process main var/ptr/global annotation string with the attached optional
// integer parameters
void processAnnotationString(IntrinsicInst *II, std::string &AnnotationString) {
  if (auto *GEP = dyn_cast<GetElementPtrInst>(II->getArgOperand(1))) {
    if (auto *C = dyn_cast<Constant>(GEP->getOperand(0))) {
      StringRef StrRef;
      getConstantStringInfo(C, StrRef);
      AnnotationString += StrRef.str();
    }
  }
  if (auto *Cast = dyn_cast<BitCastInst>(II->getArgOperand(4)))
    if (auto *C = dyn_cast_or_null<Constant>(Cast->getOperand(0)))
      processOptionalAnnotationInfo(C, AnnotationString);
}

AnnotationDecorations tryParseAnnotationString(SPIRVModule *BM,
                                               StringRef AnnotatedCode) {
  AnnotationDecorations Decorates;
  const bool AllowFPGAMemAccesses =
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_memory_accesses);
  const bool AllowFPGAMemAttr = BM->isAllowedToUseExtension(
      ExtensionID::SPV_INTEL_fpga_memory_attributes);
  if (!AllowFPGAMemAccesses && !AllowFPGAMemAttr) {
    Decorates.MemoryAttributesVec.emplace_back(DecorationUserSemantic,
                                               AnnotatedCode.str());
    return Decorates;
  }

  IntelLSUControlsInfo LSUControls;

  // Intel FPGA decorations are separated into
  // {word} OR {word:value,value,...} blocks
  std::regex DecorationRegex("\\{[\\w:,-]+\\}");
  using RegexIterT = std::regex_iterator<StringRef::const_iterator>;
  RegexIterT DecorationsIt(AnnotatedCode.begin(), AnnotatedCode.end(),
                           DecorationRegex);
  RegexIterT DecorationsEnd;
  // If we didn't find any FPGA specific annotations that are seprated as
  // described above, then add a UserSemantic decoration
  if (DecorationsIt == DecorationsEnd)
    Decorates.MemoryAttributesVec.emplace_back(DecorationUserSemantic,
                                               AnnotatedCode.str());
  bool IntelFPGADecorationFound = false;
  DecorationsInfoVec IntelFPGADecorationsVec;
  for (; DecorationsIt != DecorationsEnd; ++DecorationsIt) {
    // Drop the braces surrounding the actual decoration
    const StringRef AnnotatedDecoration = AnnotatedCode.substr(
        DecorationsIt->position() + 1, DecorationsIt->length() - 2);

    std::pair<StringRef, StringRef> Split = AnnotatedDecoration.split(':');
    StringRef Name = Split.first, ValueStr = Split.second;
    if (AllowFPGAMemAccesses) {
      if (Name == "params") {
        IntelFPGADecorationFound = true;
        unsigned ParamsBitMask = 0;
        bool Failure = ValueStr.getAsInteger(10, ParamsBitMask);
        assert(!Failure && "Non-integer LSU controls value");
        (void)Failure;
        LSUControls.setWithBitMask(ParamsBitMask);
      } else if (Name == "cache-size") {
        IntelFPGADecorationFound = true;
        if (!LSUControls.CacheSizeInfo.hasValue())
          continue;
        unsigned CacheSizeValue = 0;
        bool Failure = ValueStr.getAsInteger(10, CacheSizeValue);
        assert(!Failure && "Non-integer cache size value");
        (void)Failure;
        LSUControls.CacheSizeInfo = CacheSizeValue;
      } // TODO: Support LSU prefetch size, which currently defaults to 0
    }
    if (AllowFPGAMemAttr) {
      StringRef Annotation;
      Decoration Dec;
      if (Name == "pump") {
        IntelFPGADecorationFound = true;
        Dec = llvm::StringSwitch<Decoration>(ValueStr)
                  .Case("1", DecorationSinglepumpINTEL)
                  .Case("2", DecorationDoublepumpINTEL);
      } else if (Name == "register") {
        IntelFPGADecorationFound = true;
        Dec = DecorationRegisterINTEL;
      } else if (Name == "simple_dual_port") {
        IntelFPGADecorationFound = true;
        Dec = DecorationSimpleDualPortINTEL;
      } else {
        Dec = llvm::StringSwitch<Decoration>(Name)
                  .Case("memory", DecorationMemoryINTEL)
                  .Case("numbanks", DecorationNumbanksINTEL)
                  .Case("bankwidth", DecorationBankwidthINTEL)
                  .Case("private_copies", DecorationMaxPrivateCopiesINTEL)
                  .Case("max_replicates", DecorationMaxReplicatesINTEL)
                  .Case("bank_bits", DecorationBankBitsINTEL)
                  .Case("merge", DecorationMergeINTEL)
                  .Case("force_pow2_depth", DecorationForcePow2DepthINTEL)
                  .Default(DecorationUserSemantic);
        if (Dec == DecorationUserSemantic)
          Annotation = AnnotatedDecoration;
        else {
          IntelFPGADecorationFound = true;
          Annotation = ValueStr;
        }
      }
      IntelFPGADecorationsVec.emplace_back(Dec, Annotation.str());
    }
  }
  // Even if there is an annotation string that is split in blocks like Intel
  // FPGA annotation, it's not necessarily an FPGA annotation. Translate the
  // whole string as UserSemantic decoration in this case.
  if (IntelFPGADecorationFound)
    Decorates.MemoryAttributesVec = IntelFPGADecorationsVec;
  else
    Decorates.MemoryAttributesVec.emplace_back(DecorationUserSemantic,
                                               AnnotatedCode.str());
  Decorates.MemoryAccessesVec = LSUControls.getDecorationsFromCurrentState();

  return Decorates;
}

std::vector<SPIRVWord> getBankBitsFromString(StringRef S) {
  SmallVector<StringRef, 4> BitsString;
  S.split(BitsString, ',');

  std::vector<SPIRVWord> Bits(BitsString.size());
  for (size_t J = 0; J < BitsString.size(); ++J)
    if (BitsString[J].getAsInteger(10, Bits[J]))
      return {};

  return Bits;
}

void addAnnotationDecorations(SPIRVEntry *E, DecorationsInfoVec &Decorations) {
  SPIRVModule *M = E->getModule();
  for (const auto &I : Decorations) {
    // Such decoration already exists on a type, skip it
    if (E->hasDecorate(I.first, /*Index=*/0, /*Result=*/nullptr)) {
      continue;
    }

    switch (I.first) {
    case DecorationUserSemantic:
      E->addDecorate(new SPIRVDecorateUserSemanticAttr(E, I.second));
      break;
    case DecorationMemoryINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes))
        E->addDecorate(new SPIRVDecorateMemoryINTELAttr(E, I.second));
    } break;
    case DecorationMergeINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        StringRef Name = StringRef(I.second).split(':').first;
        StringRef Direction = StringRef(I.second).split(':').second;
        E->addDecorate(
            new SPIRVDecorateMergeINTELAttr(E, Name.str(), Direction.str()));
      }
    } break;
    case DecorationBankBitsINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes))
        E->addDecorate(new SPIRVDecorateBankBitsINTELAttr(
            E, getBankBitsFromString(I.second)));
    } break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationSimpleDualPortINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        assert(I.second.empty());
        E->addDecorate(I.first);
      }
    } break;
    case DecorationBurstCoalesceINTEL:
    case DecorationDontStaticallyCoalesceINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_accesses)) {
        assert(I.second.empty());
        E->addDecorate(I.first);
      }
    } break;
    case DecorationNumbanksINTEL:
    case DecorationBankwidthINTEL:
    case DecorationMaxPrivateCopiesINTEL:
    case DecorationMaxReplicatesINTEL:
    case DecorationForcePow2DepthINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_attributes)) {
        SPIRVWord Result = 0;
        StringRef(I.second).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
    case DecorationCacheSizeINTEL:
    case DecorationPrefetchINTEL: {
      if (M->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_fpga_memory_accesses)) {
        SPIRVWord Result = 0;
        StringRef(I.second).getAsInteger(10, Result);
        E->addDecorate(I.first, Result);
      }
    } break;
    default:
      // Other decorations are either not supported by the translator or
      // handled in other places.
      break;
    }
  }
}

void addAnnotationDecorationsForStructMember(SPIRVEntry *E,
                                             SPIRVWord MemberNumber,
                                             DecorationsInfoVec &Decorations) {
  for (const auto &I : Decorations) {
    // Such decoration already exists on a type, skip it
    if (E->hasMemberDecorate(I.first, /*Index=*/0, MemberNumber,
                             /*Result=*/nullptr)) {
      continue;
    }

    switch (I.first) {
    case DecorationUserSemantic:
      E->addMemberDecorate(
          new SPIRVMemberDecorateUserSemanticAttr(E, MemberNumber, I.second));
      break;
    case DecorationMemoryINTEL:
      E->addMemberDecorate(
          new SPIRVMemberDecorateMemoryINTELAttr(E, MemberNumber, I.second));
      break;
    case DecorationMergeINTEL: {
      StringRef Name = StringRef(I.second).split(':').first;
      StringRef Direction = StringRef(I.second).split(':').second;
      E->addMemberDecorate(new SPIRVMemberDecorateMergeINTELAttr(
          E, MemberNumber, Name.str(), Direction.str()));
    } break;
    case DecorationBankBitsINTEL:
      E->addMemberDecorate(new SPIRVMemberDecorateBankBitsINTELAttr(
          E, MemberNumber, getBankBitsFromString(I.second)));
      break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationSimpleDualPortINTEL:
      assert(I.second.empty());
      E->addMemberDecorate(MemberNumber, I.first);
      break;
    // The rest of IntelFPGA decorations:
    // DecorationNumbanksINTEL
    // DecorationBankwidthINTEL
    // DecorationMaxPrivateCopiesINTEL
    // DecorationMaxReplicatesINTEL
    // DecorationForcePow2DepthINTEL
    default:
      SPIRVWord Result = 0;
      StringRef(I.second).getAsInteger(10, Result);
      E->addMemberDecorate(MemberNumber, I.first, Result);
      break;
    }
  }
}

bool LLVMToSPIRVBase::isKnownIntrinsic(Intrinsic::ID Id) {
  // Known intrinsics usually do not need translation of their declaration
  switch (Id) {
  case Intrinsic::abs:
  case Intrinsic::assume:
  case Intrinsic::bitreverse:
  case Intrinsic::ceil:
  case Intrinsic::copysign:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::fma:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::maximum:
  case Intrinsic::maxnum:
  case Intrinsic::smax:
  case Intrinsic::umax:
  case Intrinsic::minimum:
  case Intrinsic::minnum:
  case Intrinsic::smin:
  case Intrinsic::umin:
  case Intrinsic::nearbyint:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::sin:
  case Intrinsic::sqrt:
  case Intrinsic::trunc:
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::expect:
  case Intrinsic::experimental_noalias_scope_decl:
  case Intrinsic::experimental_constrained_fadd:
  case Intrinsic::experimental_constrained_fsub:
  case Intrinsic::experimental_constrained_fmul:
  case Intrinsic::experimental_constrained_fdiv:
  case Intrinsic::experimental_constrained_frem:
  case Intrinsic::experimental_constrained_fma:
  case Intrinsic::experimental_constrained_fptoui:
  case Intrinsic::experimental_constrained_fptosi:
  case Intrinsic::experimental_constrained_uitofp:
  case Intrinsic::experimental_constrained_sitofp:
  case Intrinsic::experimental_constrained_fptrunc:
  case Intrinsic::experimental_constrained_fpext:
  case Intrinsic::experimental_constrained_fcmp:
  case Intrinsic::experimental_constrained_fcmps:
  case Intrinsic::experimental_constrained_fmuladd:
  case Intrinsic::fmuladd:
  case Intrinsic::memset:
  case Intrinsic::memcpy:
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_value:
  case Intrinsic::annotation:
  case Intrinsic::var_annotation:
  case Intrinsic::ptr_annotation:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::dbg_label:
  case Intrinsic::trap:
  case Intrinsic::arithmetic_fence:
    return true;
  default:
    // Unknown intrinsics' declarations should always be translated
    return false;
  }
}

// Performs mapping of LLVM IR rounding mode to SPIR-V rounding mode
// Value *V is metadata <rounding mode> argument of
// llvm.experimental.constrained.* intrinsics
SPIRVInstruction *
LLVMToSPIRVBase::applyRoundingModeConstraint(Value *V, SPIRVInstruction *I) {
  StringRef RMode =
      cast<MDString>(cast<MetadataAsValue>(V)->getMetadata())->getString();
  if (RMode.endswith("tonearest"))
    I->addFPRoundingMode(FPRoundingModeRTE);
  else if (RMode.endswith("towardzero"))
    I->addFPRoundingMode(FPRoundingModeRTZ);
  else if (RMode.endswith("upward"))
    I->addFPRoundingMode(FPRoundingModeRTP);
  else if (RMode.endswith("downward"))
    I->addFPRoundingMode(FPRoundingModeRTN);
  return I;
}

static SPIRVWord getBuiltinIdForIntrinsicGLSL(Intrinsic::ID IID) {
  switch (IID) {
  case Intrinsic::ceil:
    return GLSLLIB::Ceil;
  case Intrinsic::cos:
    return GLSLLIB::Cos;
  case Intrinsic::exp:
    return GLSLLIB::Exp;
  case Intrinsic::exp2:
    return GLSLLIB::Exp2;
  case Intrinsic::fabs:
    return GLSLLIB::FAbs;
  case Intrinsic::floor:
    return GLSLLIB::Floor;
  case Intrinsic::fma:
    return GLSLLIB::Fma;
  case Intrinsic::log:
    return GLSLLIB::Log;
  case Intrinsic::log2:
    return GLSLLIB::Log2;
  case Intrinsic::maximum:
    return GLSLLIB::FMax;
  case Intrinsic::maxnum:
    return GLSLLIB::FMax;
  case Intrinsic::minimum:
    return GLSLLIB::FMin;
  case Intrinsic::minnum:
    return GLSLLIB::FMin;
  case Intrinsic::nearbyint:
    return GLSLLIB::RoundEven;
  case Intrinsic::pow:
    return GLSLLIB::Pow;
  case Intrinsic::rint:
    return GLSLLIB::RoundEven;
  case Intrinsic::round:
    return GLSLLIB::Round;
  case Intrinsic::roundeven:
    return GLSLLIB::RoundEven;
  case Intrinsic::sin:
    return GLSLLIB::Sin;
  case Intrinsic::sqrt:
    return GLSLLIB::Sqrt;
  case Intrinsic::trunc:
    return GLSLLIB::Trunc;
  case Intrinsic::abs:
    return GLSLLIB::SAbs;
  case Intrinsic::ctlz:
    assert(false && "clz is not supported with GLSL/Vulkan - use libfloor "
                    "wrappers instead");
    return 0;
  case Intrinsic::cttz:
    // not fully usable, but good enough
    return GLSLLIB::FindILsb;
  case Intrinsic::copysign:
  case Intrinsic::log10:
  case Intrinsic::powi:
    assert(false && "intrinsic not supported with GLSL/Vulkan!");
    return 0;
  default:
    assert(false && "Builtin ID requested for Unhandled intrinsic!");
    return 0;
  }
}

static SPIRVWord getBuiltinIdForIntrinsicOpenCL(Intrinsic::ID IID) {
  switch (IID) {
  // Note: In some cases the semantics of the OpenCL builtin are not identical
  //       to the semantics of the corresponding LLVM IR intrinsic. The LLVM
  //       intrinsics handled here assume the default floating point environment
  //       (no unmasked exceptions, round-to-nearest-ties-even rounding mode)
  //       and assume that the operations have no side effects (FP status flags
  //       aren't maintained), so the OpenCL builtin behavior should be
  //       acceptable.
  case Intrinsic::ceil:
    return OpenCLLIB::Ceil;
  case Intrinsic::copysign:
    return OpenCLLIB::Copysign;
  case Intrinsic::cos:
    return OpenCLLIB::Cos;
  case Intrinsic::exp:
    return OpenCLLIB::Exp;
  case Intrinsic::exp2:
    return OpenCLLIB::Exp2;
  case Intrinsic::fabs:
    return OpenCLLIB::Fabs;
  case Intrinsic::floor:
    return OpenCLLIB::Floor;
  case Intrinsic::fma:
    return OpenCLLIB::Fma;
  case Intrinsic::log:
    return OpenCLLIB::Log;
  case Intrinsic::log10:
    return OpenCLLIB::Log10;
  case Intrinsic::log2:
    return OpenCLLIB::Log2;
  case Intrinsic::maximum:
    return OpenCLLIB::Fmax;
  case Intrinsic::maxnum:
    return OpenCLLIB::Fmax;
  case Intrinsic::minimum:
    return OpenCLLIB::Fmin;
  case Intrinsic::minnum:
    return OpenCLLIB::Fmin;
  case Intrinsic::nearbyint:
    return OpenCLLIB::Rint;
  case Intrinsic::pow:
    return OpenCLLIB::Pow;
  case Intrinsic::powi:
    return OpenCLLIB::Pown;
  case Intrinsic::rint:
    return OpenCLLIB::Rint;
  case Intrinsic::round:
    return OpenCLLIB::Round;
  case Intrinsic::roundeven:
    return OpenCLLIB::Rint;
  case Intrinsic::sin:
    return OpenCLLIB::Sin;
  case Intrinsic::sqrt:
    return OpenCLLIB::Sqrt;
  case Intrinsic::trunc:
    return OpenCLLIB::Trunc;
  case Intrinsic::abs:
    return OpenCLLIB::SAbs;
  case Intrinsic::ctlz:
    return OpenCLLIB::Clz;
  case Intrinsic::cttz:
    return OpenCLLIB::Ctz;
  default:
    assert(false && "Builtin ID requested for Unhandled intrinsic!");
    return 0;
  }
}

static SPIRVWord getBuiltinIdForIntrinsic(Intrinsic::ID IID,
                                          SPIRVWord src_lang) {
  if ((spv::SourceLanguage)src_lang ==
      spv::SourceLanguage::SourceLanguageGLSL) {
    return getBuiltinIdForIntrinsicGLSL(IID);
  }
  return getBuiltinIdForIntrinsicOpenCL(IID);
}

SPIRVValue *LLVMToSPIRVBase::transIntrinsicInst(IntrinsicInst *II,
                                                SPIRVBasicBlock *BB) {
  auto GetMemoryAccess = [](MemIntrinsic *MI) -> std::vector<SPIRVWord> {
    std::vector<SPIRVWord> MemoryAccess(1, MemoryAccessMaskNone);
    if (SPIRVWord AlignVal = MI->getDestAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
        SPIRVWord SourceAlignVal = MTI->getSourceAlignment();
        assert(SourceAlignVal && "Missed Source alignment!");

        // In a case when alignment of source differs from dest one
        // least value is guaranteed anyway.
        AlignVal = std::min(AlignVal, SourceAlignVal);
      }
      MemoryAccess.push_back(AlignVal);
    }
    if (MI->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    return MemoryAccess;
  };

  // LLVM intrinsics with known translation to SPIR-V are handled here. They
  // also must be registered at isKnownIntrinsic function in order to make
  // -spirv-allow-unknown-intrinsics work correctly.
  const auto ext_set =
      (SrcLang == SourceLanguageGLSL ? SPIRVEIS_GLSL : SPIRVEIS_OpenCL);
  switch (II->getIntrinsicID()) {
  case Intrinsic::assume: {
    // llvm.assume translation is currently supported only within
    // SPV_KHR_expect_assume extension, ignore it otherwise, since it's
    // an optimization hint
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_expect_assume)) {
      SPIRVValue *Condition = transValue(II->getArgOperand(0), BB);
      return BM->addAssumeTrueKHRInst(Condition, BB);
    }
    return nullptr;
  }
  case Intrinsic::bitreverse: {
    BM->addCapability(CapabilityShader);
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Op = transValue(II->getArgOperand(0), BB);
    return BM->addUnaryInst(OpBitReverse, Ty, Op, BB);
  }

  // Unary FP intrinsic
  case Intrinsic::ceil:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::nearbyint:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::sin:
  case Intrinsic::sqrt:
  case Intrinsic::trunc: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID(), SrcLang);
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(STy, BM->getExtInstSetId(ext_set), ExtOp, Ops, BB);
  }
  // Binary FP intrinsics
  case Intrinsic::copysign:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::maximum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::minnum: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID(), SrcLang);
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                  transValue(II->getArgOperand(1), BB)};
    return BM->addExtInst(STy, BM->getExtInstSetId(ext_set), ExtOp, Ops, BB);
  }
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::smin:
  case Intrinsic::smax: {
    Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
    SPIRVValue *FirstArgVal = transValue(II->getArgOperand(0), BB);
    SPIRVValue *SecondArgVal = transValue(II->getArgOperand(1), BB);

    Op OC = (II->getIntrinsicID() == Intrinsic::smin)
                ? OpSLessThan
                : ((II->getIntrinsicID() == Intrinsic::smax)
                       ? OpSGreaterThan
                       : ((II->getIntrinsicID() == Intrinsic::umin)
                              ? OpULessThan
                              : OpUGreaterThan));
    if (auto *VecTy = dyn_cast<VectorType>(II->getArgOperand(0)->getType()))
      BoolTy = VectorType::get(BoolTy, VecTy->getElementCount());
    SPIRVValue *Cmp =
        BM->addCmpInst(OC, transType(BoolTy), FirstArgVal, SecondArgVal, BB);
    return BM->addSelectInst(Cmp, FirstArgVal, SecondArgVal, BB);
  }
  case Intrinsic::fma: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID(), SrcLang);
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                  transValue(II->getArgOperand(1), BB),
                                  transValue(II->getArgOperand(2), BB)};
    return BM->addExtInst(STy, BM->getExtInstSetId(ext_set), ExtOp, Ops, BB);
  }
  case Intrinsic::abs: {
    if (!checkTypeForSPIRVExtendedInstLowering(II, BM))
      break;
    // LLVM has only one version of abs and it is only for signed integers. We
    // unconditionally choose SAbs here
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID(), SrcLang);
    SPIRVType *STy = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(STy, BM->getExtInstSetId(ext_set), ExtOp, Ops, BB);
  }
  case Intrinsic::ctpop: {
    return BM->addUnaryInst(OpBitCount, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::ctlz:
  case Intrinsic::cttz: {
    SPIRVWord ExtOp = getBuiltinIdForIntrinsic(II->getIntrinsicID(), SrcLang);
    SPIRVType *Ty = transType(II->getType());
    std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
    return BM->addExtInst(Ty, BM->getExtInstSetId(ext_set), ExtOp, Ops, BB);
  }
  case Intrinsic::expect: {
    // llvm.expect translation is currently supported only within
    // SPV_KHR_expect_assume extension, replace it with a translated value of #0
    // operand otherwise, since it's an optimization hint
    SPIRVValue *Value = transValue(II->getArgOperand(0), BB);
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_expect_assume)) {
      SPIRVType *Ty = transType(II->getType());
      SPIRVValue *ExpectedValue = transValue(II->getArgOperand(1), BB);
      return BM->addExpectKHRInst(Ty, Value, ExpectedValue, BB);
    }
    return Value;
  }
  case Intrinsic::experimental_constrained_fadd: {
    auto BI = BM->addBinaryInst(OpFAdd, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fsub: {
    auto BI = BM->addBinaryInst(OpFSub, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fmul: {
    auto BI = BM->addBinaryInst(OpFMul, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fdiv: {
    auto BI = BM->addBinaryInst(OpFDiv, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_frem: {
    auto BI = BM->addBinaryInst(OpFRem, transType(II->getType()),
                                transValue(II->getArgOperand(0), BB),
                                transValue(II->getArgOperand(1), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(2), BI);
  }
  case Intrinsic::experimental_constrained_fma: {
    std::vector<SPIRVValue *> Args{transValue(II->getArgOperand(0), BB),
                                   transValue(II->getArgOperand(1), BB),
                                   transValue(II->getArgOperand(2), BB)};
    auto BI = BM->addExtInst(transType(II->getType()),
                             BM->getExtInstSetId(SPIRVEIS_OpenCL),
                             OpenCLLIB::Fma, Args, BB);
    return applyRoundingModeConstraint(II->getOperand(3), BI);
  }
  case Intrinsic::experimental_constrained_fptoui: {
    return BM->addUnaryInst(OpConvertFToU, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_fptosi: {
    return BM->addUnaryInst(OpConvertFToS, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_uitofp: {
    auto BI = BM->addUnaryInst(OpConvertUToF, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_sitofp: {
    auto BI = BM->addUnaryInst(OpConvertSToF, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_fpext: {
    return BM->addUnaryInst(OpFConvert, transType(II->getType()),
                            transValue(II->getArgOperand(0), BB), BB);
  }
  case Intrinsic::experimental_constrained_fptrunc: {
    auto BI = BM->addUnaryInst(OpFConvert, transType(II->getType()),
                               transValue(II->getArgOperand(0), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(1), BI);
  }
  case Intrinsic::experimental_constrained_fcmp:
  case Intrinsic::experimental_constrained_fcmps: {
    auto MetaMod = cast<MetadataAsValue>(II->getOperand(2))->getMetadata();
    Op CmpTypeOp = StringSwitch<Op>(cast<MDString>(MetaMod)->getString())
                       .Case("oeq", OpFOrdEqual)
                       .Case("ogt", OpFOrdGreaterThan)
                       .Case("oge", OpFOrdGreaterThanEqual)
                       .Case("olt", OpFOrdLessThan)
                       .Case("ole", OpFOrdLessThanEqual)
                       .Case("one", OpFOrdNotEqual)
                       .Case("ord", OpOrdered)
                       .Case("ueq", OpFUnordEqual)
                       .Case("ugt", OpFUnordGreaterThan)
                       .Case("uge", OpFUnordGreaterThanEqual)
                       .Case("ult", OpFUnordLessThan)
                       .Case("ule", OpFUnordLessThanEqual)
                       .Case("une", OpFUnordNotEqual)
                       .Case("uno", OpUnordered)
                       .Default(OpNop);
    assert(CmpTypeOp != OpNop && "Invalid condition code!");
    return BM->addCmpInst(CmpTypeOp, transType(II->getType()),
                          transValue(II->getOperand(0), BB),
                          transValue(II->getOperand(1), BB), BB);
  }
  case Intrinsic::experimental_constrained_fmuladd: {
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Mul =
        BM->addBinaryInst(OpFMul, Ty, transValue(II->getArgOperand(0), BB),
                          transValue(II->getArgOperand(1), BB), BB);
    auto BI = BM->addBinaryInst(OpFAdd, Ty, Mul,
                                transValue(II->getArgOperand(2), BB), BB);
    return applyRoundingModeConstraint(II->getOperand(3), BI);
  }
  case Intrinsic::fmuladd: {
    // For llvm.fmuladd.* fusion is not guaranteed. If a fused multiply-add
    // is required the corresponding llvm.fma.* intrinsic function should be
    // used instead.
    // If allowed, let's replace llvm.fmuladd.* with mad from OpenCL extended
    // instruction set, as it has the same semantic for FULL_PROFILE OpenCL
    // devices (implementation-defined for EMBEDDED_PROFILE).
    if (BM->shouldReplaceLLVMFmulAddWithOpenCLMad() &&
        ext_set == SPIRVEIS_OpenCL) {
      std::vector<SPIRVValue *> Ops{transValue(II->getArgOperand(0), BB),
                                    transValue(II->getArgOperand(1), BB),
                                    transValue(II->getArgOperand(2), BB)};
      return BM->addExtInst(transType(II->getType()),
                            BM->getExtInstSetId(SPIRVEIS_OpenCL),
                            OpenCLLIB::Mad, Ops, BB);
    }

    // Otherwise, just break llvm.fmuladd.* into a pair of fmul + fadd
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Mul =
        BM->addBinaryInst(OpFMul, Ty, transValue(II->getArgOperand(0), BB),
                          transValue(II->getArgOperand(1), BB), BB);
    return BM->addBinaryInst(OpFAdd, Ty, Mul,
                             transValue(II->getArgOperand(2), BB), BB);
  }
  case Intrinsic::usub_sat: {
    // usub.sat(a, b) -> (a > b) ? a - b : 0
    SPIRVType *Ty = transType(II->getType());
    Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
    SPIRVValue *FirstArgVal = transValue(II->getArgOperand(0), BB);
    SPIRVValue *SecondArgVal = transValue(II->getArgOperand(1), BB);

    SPIRVValue *Sub =
        BM->addBinaryInst(OpISub, Ty, FirstArgVal, SecondArgVal, BB);
    SPIRVValue *Cmp = BM->addCmpInst(OpUGreaterThan, transType(BoolTy),
                                     FirstArgVal, SecondArgVal, BB);
    SPIRVValue *Zero = transValue(Constant::getNullValue(II->getType()), BB);
    return BM->addSelectInst(Cmp, Sub, Zero, BB);
  }
  case Intrinsic::memset: {
    // Generally there is no direct mapping of memset to SPIR-V.  But it turns
    // out that memset is emitted by Clang for initialization in default
    // constructors so we need some basic support.  The code below only handles
    // cases with constant value and constant length.
    MemSetInst *MSI = cast<MemSetInst>(II);
    Value *Val = MSI->getValue();
    if (!isa<Constant>(Val)) {
      assert(false &&
             "Can't translate llvm.memset with non-const `value` argument");
      return nullptr;
    }
    Value *Len = MSI->getLength();
    if (!isa<ConstantInt>(Len)) {
      assert(false &&
             "Can't translate llvm.memset with non-const `length` argument");
      return nullptr;
    }
    uint64_t NumElements = static_cast<ConstantInt *>(Len)->getZExtValue();
    auto *AT = ArrayType::get(Val->getType(), NumElements);
    SPIRVTypeArray *CompositeTy = static_cast<SPIRVTypeArray *>(transType(AT));
    SPIRVValue *Init;
    if (cast<Constant>(Val)->isZeroValue()) {
      Init = BM->addNullConstant(CompositeTy);
    } else {
      // On 32-bit systems, size_type of std::vector is not a 64-bit type. Let's
      // assume that we won't encounter memset for more than 2^32 elements and
      // insert explicit cast to avoid possible warning/error about narrowing
      // conversion
      auto TNumElts =
          static_cast<std::vector<SPIRVValue *>::size_type>(NumElements);
      std::vector<SPIRVValue *> Elts(TNumElts, transValue(Val, BB));
      Init = BM->addCompositeConstant(CompositeTy, Elts);
    }
    SPIRVType *VarTy = transType(PointerType::get(AT, SPIRV::SPIRAS_Constant));
    SPIRVValue *Var = BM->addVariable(VarTy, /*isConstant*/ true,
                                      spv::internal::LinkageTypeInternal, Init,
                                      "", StorageClassUniformConstant, nullptr);
    SPIRVType *SourceTy =
        transType(PointerType::get(Val->getType(), SPIRV::SPIRAS_Constant));
    SPIRVValue *Source = BM->addUnaryInst(OpBitcast, SourceTy, Var, BB);
    SPIRVValue *Target = transValue(MSI->getRawDest(), BB);
    assert(SrcLang != spv::SourceLanguageGLSL &&
           "unhandled memset during Vulkan/SPIR-V emission");
    return BM->addCopyMemorySizedInst(Target, Source, CompositeTy->getLength(),
                                      GetMemoryAccess(MSI), BB);
  } break;
  case Intrinsic::memcpy: {
    auto dst = II->getOperand(0);
    auto src = II->getOperand(1);
    if (SrcLang == spv::SourceLanguageGLSL) {
      // OpCopyMemory has the requirement that we copy to/from the actual
      // underlying type, not i8*
      // -> remove bitcasts
      do {
        auto bitcast = dyn_cast_or_null<BitCastInst>(dst);
        if (!bitcast) {
          break;
        }
        dst = bitcast->getOperand(0);
      } while (true);
      do {
        auto bitcast = dyn_cast_or_null<BitCastInst>(src);
        if (!bitcast) {
          break;
        }
        src = bitcast->getOperand(0);
      } while (true);
      assert(dst->getType()->isPointerTy());
      assert(src->getType()->isPointerTy());
      // -> assert we're actually copying data of the same type
      assert(dst->getType()->getPointerElementType() ==
             src->getType()->getPointerElementType());
      return BM->addCopyMemoryInst(transValue(dst, BB), transValue(src, BB),
                                   GetMemoryAccess(cast<MemIntrinsic>(II)), BB);
    } else {
      return BM->addCopyMemorySizedInst(
          transValue(dst, BB), transValue(src, BB),
          transValue(II->getOperand(2), BB),
          GetMemoryAccess(cast<MemIntrinsic>(II)), BB);
    }
  }
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end: {
    Op OC = (II->getIntrinsicID() == Intrinsic::lifetime_start)
                ? OpLifetimeStart
                : OpLifetimeStop;
    int64_t Size = dyn_cast<ConstantInt>(II->getOperand(0))->getSExtValue();
    if (Size == -1)
      Size = 0;
    return BM->addLifetimeInst(OC, transValue(II->getOperand(1), BB), Size, BB);
  }
  // We don't want to mix translation of regular code and debug info, because
  // it creates a mess, therefore translation of debug intrinsics is
  // postponed until LLVMToSPIRVDbgTran::finalizeDebug...() methods.
  case Intrinsic::dbg_declare:
    return DbgTran->createDebugDeclarePlaceholder(cast<DbgDeclareInst>(II), BB);
  case Intrinsic::dbg_value:
    return DbgTran->createDebugValuePlaceholder(cast<DbgValueInst>(II), BB);
  case Intrinsic::annotation: {
    SPIRVType *Ty = transType(II->getType());

    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(II->getArgOperand(1));
    if (!GEP)
      return nullptr;
    Constant *C = cast<Constant>(GEP->getOperand(0));
    StringRef AnnotationString;
    getConstantStringInfo(C, AnnotationString);

    if (AnnotationString == kOCLBuiltinName::FPGARegIntel) {
      if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_reg))
        return BM->addFPGARegINTELInst(Ty, transValue(II->getOperand(0), BB),
                                       BB);
      else
        return transValue(II->getOperand(0), BB);
    }

    return nullptr;
  }
  case Intrinsic::var_annotation: {
    SPIRVValue *SV;
    if (auto *BI = dyn_cast<BitCastInst>(II->getArgOperand(0))) {
      SV = transValue(BI->getOperand(0), BB);
    } else {
      SV = transValue(II->getOperand(0), BB);
    }

    std::string AnnotationString;
    processAnnotationString(II, AnnotationString);
    DecorationsInfoVec Decorations =
        tryParseAnnotationString(BM, AnnotationString).MemoryAttributesVec;

    // If we didn't find any IntelFPGA-specific decorations, let's add the whole
    // annotation string as UserSemantic Decoration
    if (Decorations.empty()) {
      SV->addDecorate(
          new SPIRVDecorateUserSemanticAttr(SV, AnnotationString.c_str()));
    } else {
      addAnnotationDecorations(SV, Decorations);
    }
    return SV;
  }
  // The layout of llvm.ptr.annotation is:
  // declare iN*   @llvm.ptr.annotation.p<address space>iN(
  // iN* <val>, i8* <str>, i8* <str>, i32  <int>, i8* <ptr>)
  // where N is a power of two number,
  // first i8* <str> stands for the annotation itself,
  // second i8* <str> is for the location (file name),
  // i8* <ptr> is a pointer on a GV, which can carry optinal variadic
  // clang::annotation attribute expression arguments.
  case Intrinsic::ptr_annotation: {
    // Strip all bitcast and addrspace casts from the pointer argument:
    //   llvm annotation intrinsic only takes i8*, so the original pointer
    //   probably had to loose its addrspace and its original type.
    Value *AnnotSubj = II->getArgOperand(0);
    while (isa<BitCastInst>(AnnotSubj) || isa<AddrSpaceCastInst>(AnnotSubj)) {
      AnnotSubj = cast<CastInst>(AnnotSubj)->getOperand(0);
    }

    std::string AnnotationString;
    processAnnotationString(II, AnnotationString);
    AnnotationDecorations Decorations =
        tryParseAnnotationString(BM, AnnotationString);

    // If the pointer is a GEP on a struct, then we have to emit a member
    // decoration for the GEP-accessed struct, or a memory access decoration
    // for the GEP itself.
    auto *GI = dyn_cast<GetElementPtrInst>(AnnotSubj);
    if (GI && isa<StructType>(GI->getSourceElementType())) {
      auto *Ty = transType(GI->getSourceElementType());
      auto *ResPtr = transValue(GI, BB);
      SPIRVWord MemberNumber =
          dyn_cast<ConstantInt>(GI->getOperand(2))->getZExtValue();

      // If we didn't find any IntelFPGA-specific decorations, let's add the
      // whole annotation string as UserSemantic Decoration
      if (Decorations.MemoryAttributesVec.empty() &&
          Decorations.MemoryAccessesVec.empty()) {
        // TODO: Is there a way to detect that the annotation belongs solely
        // to struct member memory atributes or struct member memory access
        // controls? This would allow emitting just the necessary decoration.
        Ty->addMemberDecorate(new SPIRVMemberDecorateUserSemanticAttr(
            Ty, MemberNumber, AnnotationString.c_str()));
        ResPtr->addDecorate(new SPIRVDecorateUserSemanticAttr(
            ResPtr, AnnotationString.c_str()));
      } else {
        addAnnotationDecorationsForStructMember(
            Ty, MemberNumber, Decorations.MemoryAttributesVec);
        // Apply the LSU parameter decoration to the pointer result of a GEP
        // to the given struct member (InBoundsPtrAccessChain in SPIR-V).
        // Decorating the member itself with a MemberDecoration is not feasible,
        // because multiple accesses to the struct-held memory can require
        // different LSU parameters.
        addAnnotationDecorations(ResPtr, Decorations.MemoryAccessesVec);
      }
      II->replaceAllUsesWith(II->getOperand(0));
    } else {
      auto *Ty = transType(II->getType());
      auto *BI = dyn_cast<BitCastInst>(II->getOperand(0));
      if (AnnotationString == kOCLBuiltinName::FPGARegIntel) {
        if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_reg))
          return BM->addFPGARegINTELInst(Ty, transValue(BI, BB), BB);
        return transValue(BI, BB);
      } else {
        // Memory accesses to a standalone pointer variable
        auto *DecSubj = transValue(II->getArgOperand(0), BB);
        if (Decorations.MemoryAccessesVec.empty())
          DecSubj->addDecorate(new SPIRVDecorateUserSemanticAttr(
              DecSubj, AnnotationString.c_str()));
        else
          // Apply the LSU parameter decoration to the pointer result of an
          // instruction. Note it's the address to the accessed memory that's
          // loaded from the original pointer variable, and not the value
          // accessed by the latter.
          addAnnotationDecorations(DecSubj, Decorations.MemoryAccessesVec);
        II->replaceAllUsesWith(II->getOperand(0));
      }
    }
    return nullptr;
  }
  case Intrinsic::stacksave: {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_variable_length_array)) {
      auto *Ty = transType(II->getType());
      return BM->addInstTemplate(OpSaveMemoryINTEL, BB, Ty);
    }
    BM->getErrorLog().checkError(
        BM->isUnknownIntrinsicAllowed(II), SPIRVEC_InvalidFunctionCall, II,
        "Translation of llvm.stacksave intrinsic requires "
        "SPV_INTEL_variable_length_array extension or "
        "-spirv-allow-unknown-intrinsics option.");
    break;
  }
  case Intrinsic::stackrestore: {
    if (BM->isAllowedToUseExtension(
            ExtensionID::SPV_INTEL_variable_length_array)) {
      auto *Ptr = transValue(II->getArgOperand(0), BB);
      return BM->addInstTemplate(OpRestoreMemoryINTEL, {Ptr->getId()}, BB,
                                 nullptr);
    }
    BM->getErrorLog().checkError(
        BM->isUnknownIntrinsicAllowed(II), SPIRVEC_InvalidFunctionCall, II,
        "Translation of llvm.restore intrinsic requires "
        "SPV_INTEL_variable_length_array extension or "
        "-spirv-allow-unknown-intrinsics option.");
    break;
  }
  // We can just ignore/drop some intrinsics, like optimizations hint.
  case Intrinsic::experimental_noalias_scope_decl:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::dbg_label:
  // llvm.trap intrinsic is not implemented. But for now don't crash. This
  // change is pending the trap/abort intrinsic implementation.
  case Intrinsic::trap:
  // llvm.instrprof.* intrinsics are not supported
  case Intrinsic::instrprof_increment:
  case Intrinsic::instrprof_increment_step:
  case Intrinsic::instrprof_value_profile:
    return nullptr;
  case Intrinsic::is_constant: {
    auto *CO = dyn_cast<Constant>(II->getOperand(0));
    if (CO && isManifestConstant(CO))
      return transValue(ConstantInt::getTrue(II->getType()), BB, false);
    else
      return transValue(ConstantInt::getFalse(II->getType()), BB, false);
  }
  case Intrinsic::arithmetic_fence: {
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Op = transValue(II->getArgOperand(0), BB);
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_arithmetic_fence)) {
      BM->addCapability(internal::CapabilityFPArithmeticFenceINTEL);
      BM->addExtension(ExtensionID::SPV_INTEL_arithmetic_fence);
      return BM->addUnaryInst(internal::OpArithmeticFenceINTEL, Ty, Op, BB);
    }
    return Op;
  }
  default:
    if (BM->isUnknownIntrinsicAllowed(II))
      return BM->addCallInst(
          transFunctionDecl(II->getCalledFunction()),
          transArguments(II, BB,
                         SPIRVEntry::createUnique(OpFunctionCall).get()),
          BB);
    else
      // Other LLVM intrinsics shouldn't get to SPIRV, because they
      // can't be represented in SPIRV or aren't implemented yet.
      BM->SPIRVCK(false, InvalidFunctionCall,
                  II->getCalledOperand()->getName().str());
  }
  return nullptr;
}

SPIRVValue *LLVMToSPIRVBase::transCallInst(CallInst *CI, SPIRVBasicBlock *BB) {
  assert(CI);
  Function *F = CI->getFunction();
  if (isa<InlineAsm>(CI->getCalledOperand()) &&
      BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_inline_assembly)) {
    // Inline asm is opaque, so we cannot reason about its FP contraction
    // requirements.
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                    << ": inline asm " << *CI << '\n');
    joinFPContract(F, FPContract::DISABLED);
    return transAsmCallINTEL(CI, BB);
  }

  if (CI->isIndirectCall()) {
    // The function is not known in advance
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                    << ": indirect call " << *CI << '\n');
    joinFPContract(F, FPContract::DISABLED);
    return transIndirectCallInst(CI, BB);
  }
  return transDirectCallInst(CI, BB);
}

SPIRVValue *LLVMToSPIRVBase::add_libfloor_sub_group_simd_shuffle(
    StringRef MangledName, CallInst *CI, SPIRVBasicBlock *BB) {
  if (CI->getNumOperands() < 2) {
    assert(false &&
           "invalid amount of operands in libfloor sub-group simd-shuffle");
    return nullptr;
  }

  // get shuffle type
  Op opcode = spv::OpNop;
  if (MangledName.consume_front("simd_shuffle.")) {
    opcode = spv::OpGroupNonUniformShuffle;
  } else if (MangledName.consume_front("simd_shuffle_xor.")) {
    opcode = spv::OpGroupNonUniformShuffleUp;
  } else if (MangledName.consume_front("simd_shuffle_down.")) {
    opcode = spv::OpGroupNonUniformShuffleDown;
  } else if (MangledName.consume_front("simd_shuffle_up.")) {
    opcode = spv::OpGroupNonUniformShuffleUp;
  } else {
    assert(false && "invalid sub-group simd-shuffle");
    return nullptr;
  }

  // translate value and handle type
  auto lane_value = CI->getOperand(0);
  auto spv_lane_value = transValue(lane_value, BB);
  auto shuffle_op_type = spv_lane_value->getType();

  if (MangledName.endswith("s32") || MangledName.endswith("u32")) {
    if (!shuffle_op_type->isTypeInt()) {
      assert(false && "expected integer type");
      return nullptr;
    }
  } else if (MangledName.endswith("f32")) {
    if (!shuffle_op_type->isTypeFloat()) {
      assert(false && "expected float type");
      return nullptr;
    }
  } else {
    assert(false && "invalid data type");
    return nullptr;
  }

  // idx/delta/mask *must* be unsigned
  auto lane_idx_delta_or_mask = CI->getOperand(1);
  SPIRVValue *spv_lane_idx_delta_or_mask = nullptr;
  if (auto const_lane_idx_delta_or_mask =
          dyn_cast_or_null<ConstantInt>(lane_idx_delta_or_mask);
      const_lane_idx_delta_or_mask) {
    // if it is a constant, we can directly emit the SPIR-V constant as unsinged
    spv_lane_idx_delta_or_mask = BM->getLiteralAsConstant(
        uint32_t(const_lane_idx_delta_or_mask->getZExtValue()), false);
  } else {
    // if it is dynamic, we must bitcast to unsigned if it's not unsigned
    spv_lane_idx_delta_or_mask = transValue(lane_idx_delta_or_mask, BB);
    assert(spv_lane_idx_delta_or_mask->getType()->isTypeInt());
    const auto int_type =
        (const SPIRVTypeInt *)spv_lane_idx_delta_or_mask->getType();
    if (int_type->isSigned()) {
      spv_lane_idx_delta_or_mask =
          BM->addUnaryInst(spv::OpBitcast, int_type->getUnsigned(),
                           spv_lane_idx_delta_or_mask, BB);
    }
  }

  return BM->addGroupNonUniformShuffleInst(opcode, spv::ScopeSubgroup,
                                           spv_lane_value,
                                           spv_lane_idx_delta_or_mask, BB);
}

SPIRVValue *LLVMToSPIRVBase::add_libfloor_sub_group_op(StringRef MangledName,
                                                       CallInst *CI,
                                                       SPIRVBasicBlock *BB) {
  if (!MangledName.consume_front("floor.sub_group.")) {
    assert(false && "this is not a libfloor sub-group operation");
    return nullptr;
  }

  // extract algorithm
  if (MangledName.startswith("simd_shuffle")) {
    return add_libfloor_sub_group_simd_shuffle(MangledName, CI, BB);
  }

  if (CI->getNumOperands() == 0) {
    assert(false &&
           "invalid amount of operands in libfloor sub-group operation");
    return nullptr;
  }

  auto group_op = spv::GroupOperationMax;
  if (MangledName.consume_front("reduce.")) {
    group_op = spv::GroupOperationReduce;
  } else if (MangledName.consume_front("inclusive_scan.")) {
    group_op = spv::GroupOperationInclusiveScan;
  } else if (MangledName.consume_front("exclusive_scan.")) {
    group_op = spv::GroupOperationExclusiveScan;
  } else {
    assert(false && "invalid sub-group algorithm");
    return nullptr;
  }

  // extract op
  Op opcode = spv::OpNop;
  if (MangledName.consume_front("add.")) {
    opcode = spv::OpGroupNonUniformIAdd;
  } else if (MangledName.consume_front("min.")) {
    opcode = spv::OpGroupNonUniformSMin;
  } else if (MangledName.consume_front("max.")) {
    opcode = spv::OpGroupNonUniformSMax;
  } else {
    assert(false && "invalid sub-group op");
    return nullptr;
  }

  // translate value and handle type
  auto item_value = CI->getOperand(0);
  auto spv_item_value = transValue(item_value, BB);
  auto group_op_type = spv_item_value->getType();

  if (MangledName.endswith("s32")) {
    if (!group_op_type->isTypeInt()) {
      assert(false && "expected integer type");
      return nullptr;
    }
    const auto int_type = (const SPIRVTypeInt *)group_op_type;
    if (int_type->getBitWidth() != 32) {
      assert(false && "sub-group ops only support 32-bit types");
      return nullptr;
    }
    // depending on the op, we may need to perform type conversion
    if (opcode != spv::OpGroupNonUniformIAdd && !int_type->isSigned()) {
      spv_item_value = BM->addUnaryInst(spv::OpBitcast, int_type->getSigned(),
                                        spv_item_value, BB);
      group_op_type = spv_item_value->getType();
    }

    // NOTE: opcode is already correct here
  } else if (MangledName.endswith("u32")) {
    if (!group_op_type->isTypeInt()) {
      assert(false && "expected integer type");
      return nullptr;
    }
    const auto int_type = (const SPIRVTypeInt *)group_op_type;
    if (int_type->getBitWidth() != 32) {
      assert(false && "sub-group ops only support 32-bit types");
      return nullptr;
    }
    // depending on the op, we may need to perform type conversion
    if (opcode != spv::OpGroupNonUniformIAdd && int_type->isSigned()) {
      spv_item_value = BM->addUnaryInst(spv::OpBitcast, int_type->getUnsigned(),
                                        spv_item_value, BB);
      group_op_type = spv_item_value->getType();
    }

    switch (opcode) {
    case spv::OpGroupNonUniformIAdd:
      // stays the same
      break;
    case spv::OpGroupNonUniformSMin:
      opcode = spv::OpGroupNonUniformUMin;
      break;
    case spv::OpGroupNonUniformSMax:
      opcode = spv::OpGroupNonUniformUMax;
      break;
    default:
      break;
    }
  } else if (MangledName.endswith("f32")) {
    if (!group_op_type->isTypeFloat()) {
      assert(false && "expected float type");
      return nullptr;
    }
    if (((const SPIRVTypeFloat *)group_op_type)->getBitWidth() != 32) {
      assert(false && "sub-group ops only support 32-bit types");
      return nullptr;
    }

    switch (opcode) {
    case spv::OpGroupNonUniformIAdd:
      opcode = spv::OpGroupNonUniformFAdd;
      break;
    case spv::OpGroupNonUniformSMin:
      opcode = spv::OpGroupNonUniformFMin;
      break;
    case spv::OpGroupNonUniformSMax:
      opcode = spv::OpGroupNonUniformFMax;
      break;
    default:
      break;
    }
  } else {
    assert(false && "invalid data type");
    return nullptr;
  }

  return BM->addGroupNonUniformArithmeticInst(opcode, spv::ScopeSubgroup,
                                              group_op, spv_item_value, BB);
}

SPIRVValue *LLVMToSPIRVBase::transDirectCallInst(CallInst *CI,
                                                 SPIRVBasicBlock *BB) {
  SPIRVExtInstSetKind ExtSetKind = SPIRVEIS_Count;
  SPIRVWord ExtOp = SPIRVWORD_MAX;
  llvm::Function *F = CI->getCalledFunction();
  auto MangledName = F->getName();
  StringRef DemangledName;

  if (MangledName.startswith(SPCV_CAST) || MangledName == SAMPLER_INIT)
    return oclTransSpvcCastSampler(CI, BB);

  if (oclIsBuiltin(MangledName, DemangledName) ||
      isDecoratedSPIRVFunc(F, DemangledName)) {
    if (auto BV = transBuiltinToConstant(DemangledName, CI))
      return BV;
    if (auto BV = transBuiltinToInst(DemangledName, CI, BB))
      return BV;
  }

  SmallVector<std::string, 2> Dec;
  if (isBuiltinTransToExtInst(CI->getCalledFunction(), &ExtSetKind, &ExtOp,
                              &Dec)) {
    return addDecorations(
        BM->addExtInst(
            transType(CI->getType()), BM->getExtInstSetId(ExtSetKind), ExtOp,
            transArguments(CI, BB,
                           SPIRVEntry::createUnique(ExtSetKind, ExtOp).get()),
            BB),
        Dec);
  }

  // helper functions to force an integer value to be unsigned or signed
  const auto force_uint_value = [&BB, this](SPIRVValue *val) -> SPIRVValue * {
    auto type = val->getType();
    if (!type->isTypeInt()) {
      assert(false && "expected an integer type");
      return val;
    }
    if (((SPIRVTypeInt *)type)->isSigned()) {
      // bitcast to unsigned
      return BM->addUnaryInst(spv::OpBitcast,
                              ((SPIRVTypeInt *)type)->getUnsigned(), val, BB);
    }
    // already unsigned
    return val;
  };
  const auto force_int_value = [&BB, this](SPIRVValue *val) -> SPIRVValue * {
    auto type = val->getType();
    if (!type->isTypeInt()) {
      assert(false && "expected an integer type");
      return val;
    }
    if (!((SPIRVTypeInt *)type)->isSigned()) {
      // bitcast to signed
      return BM->addUnaryInst(spv::OpBitcast,
                              ((SPIRVTypeInt *)type)->getSigned(), val, BB);
    }
    // already signed
    return val;
  };

  // TODO: put this into an extra function + use lut
  if (MangledName.startswith("floor.")) {
    if (MangledName.startswith("floor.composite_construct.")) {
      std::vector<SPIRVWord> Constituents;
      for (const auto &elem : CI->args()) {
        Constituents.emplace_back(transValue(elem, BB)->getId());
      }
      return BM->addCompositeConstructInst(transType(CI->getType()),
                                           Constituents, BB);
    } else if (MangledName == "floor.dfdx.f32" ||
               MangledName == "floor.dfdy.f32" ||
               MangledName == "floor.fwidth.f32") {
      auto OC = spv::OpDPdx;
      if (MangledName == "floor.dfdy.f32")
        OC = spv::OpDPdy;
      if (MangledName == "floor.fwidth.f32")
        OC = spv::OpFwidth;
      assert(CI->getArgOperand(0)->getType() == CI->getType() &&
             "invalid derivative type");
      return BM->addDerivativeInst(OC, transValue(CI->getArgOperand(0), BB),
                                   BB);
    } else if (MangledName == "floor.discard_fragment") {
      // since "discard" can't be modelled as a single noreturn +
      // unreachable-after-call instruction right now, but must add an
      // "additional" unreachable instead, we need to get rid of (ignore) the
      // next unreachable (this isn't particularly nice, but we can't do this on
      // the llvm side, b/c it would badly break things)
      ignore_next_unreachable = true;
      return BM->addKillInst(BB);
    } else if (MangledName.startswith("floor.find_int_lsb.")) {
      auto arg = transValue(CI->getArgOperand(0), BB);
      if (MangledName.startswith("floor.find_int_lsb.u")) {
        // force uint eval
        arg = force_uint_value(arg);
      }
      return BM->addExtInst(
          ((SPIRVTypeInt *)arg->getType())->getSigned() /* force signed */,
          BM->getExtInstSetId(SPIRVEIS_GLSL), GLSLLIB::FindILsb, getVec(arg),
          BB);
    } else if (MangledName.startswith("floor.find_int_msb.")) {
      auto arg = transValue(CI->getArgOperand(0), BB);
      bool is_uint = false;
      if (MangledName.startswith("floor.find_int_msb.u")) {
        // force uint eval
        arg = force_uint_value(arg);
        is_uint = true;
      } else if (MangledName.startswith("floor.find_int_msb.s")) {
        // force int eval
        arg = force_int_value(arg);
      }
      return BM->addExtInst(
          ((SPIRVTypeInt *)arg->getType())->getSigned() /* force signed */,
          BM->getExtInstSetId(SPIRVEIS_GLSL),
          (is_uint ? GLSLLIB::FindUMsb : GLSLLIB::FindSMsb), getVec(arg), BB);
    } else if (MangledName.startswith("floor.bit_reverse.")) {
      auto arg = transValue(CI->getArgOperand(0), BB);
      if (MangledName.startswith("floor.bit_reverse.u")) {
        // force uint eval
        arg = force_uint_value(arg);
      }
      return BM->addBitReverseInst(arg->getType(), arg, BB);
    } else if (MangledName.startswith("floor.bit_count.")) {
      auto arg = transValue(CI->getArgOperand(0), BB);
      if (MangledName.startswith("floor.bit_count.u")) {
        // force uint eval
        arg = force_uint_value(arg);
      }
      return BM->addBitCountInst(arg->getType(), arg, BB);
    } else if (MangledName.startswith("floor.image_array_load.")) {
      auto img = CI->getOperand(0);
      const auto img_type_iter = image_type_map.find(img);
      assert(img_type_iter != image_type_map.end() && "unknown image");

      auto img_ptr_type = BM->addPointerType(spv::StorageClassUniformConstant,
                                             img_type_iter->second);

      auto img_array = transValue(CI->getOperand(0), BB);
      std::vector<SPIRVValue *> indices;
      for (uint32_t arg_idx = 1; arg_idx < CI->arg_size(); ++arg_idx) {
        indices.emplace_back(transValue(CI->getArgOperand(arg_idx), BB));
      }
      auto gep = BM->addAccessChainInst(img_ptr_type, img_array, indices, BB,
                                        true, false);
      auto ld = BM->addLoadInst(gep, {}, BB);
      // must decorate both the GEP and the load as NonUniform
      gep->addDecorate(DecorationNonUniform);
      ld->addDecorate(DecorationNonUniform);
      return ld;
    } else if (MangledName.startswith("floor.ssbo_array_gep.")) {
      auto base = transValue(CI->getOperand(0), BB);
      std::vector<SPIRVValue *> indices;
      for (uint32_t arg_idx = 1 /* first index */; arg_idx < CI->arg_size();
           ++arg_idx) {
        indices.emplace_back(transValue(CI->getArgOperand(arg_idx), BB));
      }
      auto ptr_type = CI->getType();
      assert(ptr_type->isPointerTy());
      assert(ptr_type->getPointerAddressSpace() == 0 ||
             ptr_type->getPointerAddressSpace() == SPIRAS_StorageBuffer);
      if (ptr_type->getPointerAddressSpace() != SPIRAS_StorageBuffer) {
        ptr_type = llvm::PointerType::get(ptr_type->getPointerElementType(),
                                          SPIRAS_StorageBuffer);
      }
      auto gep = BM->addAccessChainInst(transType(ptr_type), base, indices, BB,
                                        true, false);
      // must decorate the GEP as NonUniform (TODO: also decorate loads?)
      gep->addDecorate(DecorationNonUniform);
      return gep;
    } else if (MangledName == "floor.loop_merge") {
      auto merge_bb = transValue(CI->getArgOperand(0), nullptr);
      auto continue_bb = transValue(CI->getArgOperand(1), nullptr);
      auto loop_control = spv::LoopControlMaskNone;
      if (auto loop_control_val =
              dyn_cast_or_null<ConstantInt>(CI->getArgOperand(2));
          loop_control_val) {
        loop_control = (LoopControlMask)loop_control_val->getZExtValue();
      }
      return BM->addLoopMergeInst(merge_bb->getId(), continue_bb->getId(),
                                  loop_control, {}, BB);
    } else if (MangledName == "floor.selection_merge") {
      auto merge_bb =
          (SPIRVBasicBlock *)transValue(CI->getArgOperand(0), nullptr);
      // default to always flatten
      auto sel_control = spv::SelectionControlFlattenMask;
      if (auto sel_control_val =
              dyn_cast_or_null<ConstantInt>(CI->getArgOperand(1));
          sel_control_val) {
        sel_control = (SelectionControlMask)sel_control_val->getZExtValue();
      }
      return BM->addSelectionMergeInst(merge_bb->getId(), sel_control, BB);
    } else if (MangledName.startswith("floor.pack_") ||
               MangledName.startswith("floor.unpack_")) {
      static const std::unordered_map<std::string, GLSLExtOpKind> pack_lut{
          {"floor.pack_snorm_4x8", GLSLLIB::PackSnorm4x8},
          {"floor.pack_unorm_4x8", GLSLLIB::PackUnorm4x8},
          {"floor.pack_snorm_2x16", GLSLLIB::PackSnorm2x16},
          {"floor.pack_unorm_2x16", GLSLLIB::PackUnorm2x16},
          {"floor.pack_half_2x16", GLSLLIB::PackHalf2x16},
          {"floor.pack_double_2x32", GLSLLIB::PackDouble2x32},
          {"floor.unpack_snorm_4x8", GLSLLIB::UnpackSnorm4x8},
          {"floor.unpack_unorm_4x8", GLSLLIB::UnpackUnorm4x8},
          {"floor.unpack_snorm_2x16", GLSLLIB::UnpackSnorm2x16},
          {"floor.unpack_unorm_2x16", GLSLLIB::UnpackUnorm2x16},
          {"floor.unpack_half_2x16", GLSLLIB::UnpackHalf2x16},
          {"floor.unpack_double_2x32", GLSLLIB::UnpackDouble2x32},
      };
      const auto ext_op = pack_lut.find(MangledName.str());
      if (ext_op == pack_lut.end()) {
        errs() << "unknown pack/unpack call: " << MangledName << "\n";
        return nullptr;
      }

      SPIRVType *ret_type = nullptr;
      if (MangledName.startswith("floor.pack_") &&
          MangledName != "floor.pack_double_2x32") {
        // enforce 32-bit unsigned integer return type
        ret_type = BM->addIntegerType(32, false);
      } else {
        // otherwise, simply translate the return type
        ret_type = transType(CI->getType());
      }

      auto args = transValue(getArguments(CI), BB);
      if (args.size() != 1) {
        errs() << "invalid arg count for pack/unpack call: " << MangledName
               << "\n";
        return nullptr;
      }

      // enforce 32-bit unsigned integer arg type
      if (MangledName.startswith("floor.unpack_") &&
          MangledName != "floor.unpack_double_2x32") {
        if (((SPIRVTypeInt *)args[0]->getType())->isSigned()) {
          args[0] = BM->addUnaryInst(
              spv::OpBitcast, BM->addIntegerType(32, false), args[0], BB);
        }
      }

      return BM->addExtInst(ret_type, BM->getExtInstSetId(SPIRVEIS_GLSL),
                            ext_op->second, args, BB);
    } else if (MangledName.startswith("floor.bitcast.")) {
      auto args = transValue(getArguments(CI), BB);
      if (args.size() != 1) {
        errs() << "invalid arg count for bitcast call: " << MangledName << "\n";
        return nullptr;
      }
      transType(CI->getType());
      if (MangledName == "floor.bitcast.f32.i32") {
        return BM->addUnaryInst(spv::OpBitcast, BM->addIntegerType(32, true),
                                args[0], BB);
      } else if (MangledName == "floor.bitcast.f32.u32") {
        return BM->addUnaryInst(spv::OpBitcast, BM->addIntegerType(32, false),
                                args[0], BB);
      } else if (MangledName == "floor.bitcast.i32.f32") {
        return BM->addUnaryInst(spv::OpBitcast, BM->addFloatType(32), args[0],
                                BB);
      } else if (MangledName == "floor.bitcast.u32.f32") {
        return BM->addUnaryInst(spv::OpBitcast, BM->addFloatType(32), args[0],
                                BB);
      }
      // else: fallthrough
    } else if (MangledName.startswith("floor.sub_group.")) {
      return add_libfloor_sub_group_op(MangledName, CI, BB);
    } else if (MangledName == "floor.barrier.local") {
      const auto wg_scope = BM->getLiteralAsConstant(spv::ScopeWorkgroup, true);
      const auto wg_sema =
          BM->getLiteralAsConstant(spv::MemorySemanticsAcquireReleaseMask |
                                       spv::MemorySemanticsUniformMemoryMask |
                                       spv::MemorySemanticsSubgroupMemoryMask |
                                       spv::MemorySemanticsWorkgroupMemoryMask |
                                       spv::MemorySemanticsImageMemoryMask,
                                   true);
      return BM->addControlBarrierInst(wg_scope, wg_scope, wg_sema, BB);
    } else if (MangledName == "floor.barrier.global" ||
               MangledName == "floor.barrier.full" ||
               MangledName == "floor.barrier.image") {
      // TODO: do we need to differentiate between global and full?
      // NOTE: for now, an image barrier equals a full/global barrier
      const auto wg_scope = BM->getLiteralAsConstant(spv::ScopeDevice, true);
      const auto wg_sema =
          BM->getLiteralAsConstant(spv::MemorySemanticsAcquireReleaseMask |
                                       spv::MemorySemanticsUniformMemoryMask |
                                       spv::MemorySemanticsSubgroupMemoryMask |
                                       spv::MemorySemanticsWorkgroupMemoryMask |
                                       spv::MemorySemanticsImageMemoryMask,
                                   true);
      return BM->addControlBarrierInst(wg_scope, wg_scope, wg_sema, BB);
    }
    errs() << "unhandled floor func: " << MangledName << "\n";
  }

  Function *Callee = CI->getCalledFunction();
  if (Callee->isDeclaration()) {
    SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName().str()
                    << ": call to an undefined function " << *CI << '\n');
    joinFPContract(CI->getFunction(), FPContract::DISABLED);
  } else {
    FPContract CalleeFPC = getFPContract(Callee);
    joinFPContract(CI->getFunction(), CalleeFPC);
    if (CalleeFPC == FPContract::DISABLED) {
      SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName().str()
                      << ": call to a function with disabled contraction: "
                      << *CI << '\n');
    }
  }

  return BM->addCallInst(
      transFunctionDecl(Callee),
      transArguments(CI, BB, SPIRVEntry::createUnique(OpFunctionCall).get()),
      BB);
}

SPIRVValue *LLVMToSPIRVBase::transIndirectCallInst(CallInst *CI,
                                                   SPIRVBasicBlock *BB) {
  if (BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_function_pointers),
          SPIRVEC_FunctionPointers, CI)) {
    return BM->addIndirectCallInst(
        transValue(CI->getCalledOperand(), BB), transType(CI->getType()),
        transArguments(CI, BB, SPIRVEntry::createUnique(OpFunctionCall).get()),
        BB);
  }
  return nullptr;
}

SPIRVValue *LLVMToSPIRVBase::transAsmINTEL(InlineAsm *IA) {
  assert(IA);

  // TODO: intention here is to provide information about actual target
  //       but in fact spir-64 is substituted as triple when translator works
  //       eventually we need to fix it (not urgent)
  StringRef TripleStr(M->getTargetTriple());
  auto AsmTarget = static_cast<SPIRVAsmTargetINTEL *>(
      BM->getOrAddAsmTargetINTEL(TripleStr.str()));
  auto SIA = BM->addAsmINTEL(
      static_cast<SPIRVTypeFunction *>(transType(IA->getFunctionType())),
      AsmTarget, IA->getAsmString(), IA->getConstraintString());
  if (IA->hasSideEffects())
    SIA->addDecorate(DecorationSideEffectsINTEL);
  return SIA;
}

SPIRVValue *LLVMToSPIRVBase::transAsmCallINTEL(CallInst *CI,
                                               SPIRVBasicBlock *BB) {
  assert(CI);
  auto IA = cast<InlineAsm>(CI->getCalledOperand());
  return BM->addAsmCallINTELInst(
      static_cast<SPIRVAsmINTEL *>(transValue(IA, BB, false)),
      transArguments(CI, BB, SPIRVEntry::createUnique(OpAsmCallINTEL).get()),
      BB);
}

bool LLVMToSPIRVBase::transAddressingMode() {
  Triple TargetTriple(M->getTargetTriple());

  if (TargetTriple.getEnvironment() != llvm::Triple::EnvironmentType::Vulkan) {
    if (TargetTriple.isArch32Bit())
      BM->setAddressingModel(AddressingModelPhysical32);
    else
      BM->setAddressingModel(AddressingModelPhysical64);
    // Physical addressing model requires Addresses capability
    BM->addCapability(CapabilityAddresses);
    // OpenCL memory model requires Kernel capability
    BM->setMemoryModel(MemoryModelOpenCL);
  } else {
    BM->setAddressingModel(AddressingModelPhysicalStorageBuffer64);
    BM->setMemoryModel(MemoryModelVulkan);

    // always add these
    BM->addCapability(CapabilityShader);
    BM->addCapability(CapabilityVulkanMemoryModel);
    BM->addCapability(CapabilityVulkanMemoryModelDeviceScope);
    BM->addCapability(CapabilityPhysicalStorageBufferAddresses);
    BM->addCapability(CapabilityVariablePointersStorageBuffer);
    BM->addCapability(CapabilityVariablePointers);
    BM->addCapability(CapabilityUniformBufferArrayNonUniformIndexing);
    BM->addCapability(CapabilityStorageBufferArrayNonUniformIndexing);
    BM->addCapability(CapabilitySampledImageArrayNonUniformIndexing);
    BM->addCapability(CapabilityStorageImageArrayNonUniformIndexing);
    BM->addCapability(CapabilityShaderNonUniform);
    BM->addCapability(CapabilityGroupNonUniform);
  }
  return true;
}
std::vector<SPIRVValue *>
LLVMToSPIRVBase::transValue(const std::vector<Value *> &Args,
                            SPIRVBasicBlock *BB) {
  std::vector<SPIRVValue *> BArgs;
  for (auto &I : Args)
    BArgs.push_back(transValue(I, BB));
  return BArgs;
}

std::vector<SPIRVWord>
LLVMToSPIRVBase::transValue(const std::vector<Value *> &Args,
                            SPIRVBasicBlock *BB, SPIRVEntry *Entry) {
  std::vector<SPIRVWord> Operands;
  for (size_t I = 0, E = Args.size(); I != E; ++I) {
    Operands.push_back(Entry->isOperandLiteral(I)
                           ? cast<ConstantInt>(Args[I])->getZExtValue()
                           : transValue(Args[I], BB)->getId());
  }
  return Operands;
}

std::vector<SPIRVWord> LLVMToSPIRVBase::transArguments(CallInst *CI,
                                                       SPIRVBasicBlock *BB,
                                                       SPIRVEntry *Entry) {
  return transValue(getArguments(CI), BB, Entry);
}

SPIRVWord LLVMToSPIRVBase::transFunctionControlMask(Function *F) {
  SPIRVWord FCM = 0;
  SPIRSPIRVFuncCtlMaskMap::foreach (
      [&](Attribute::AttrKind Attr, SPIRVFunctionControlMaskKind Mask) {
        if (F->hasFnAttribute(Attr)) {
          if (Attr == Attribute::OptimizeNone) {
            if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_optnone))
              return;
            BM->addExtension(ExtensionID::SPV_INTEL_optnone);
            BM->addCapability(internal::CapabilityOptNoneINTEL);
          }
          FCM |= Mask;
        }
      });
  return FCM;
}

void LLVMToSPIRVBase::transGlobalAnnotation(GlobalVariable *V) {
  SPIRVDBG(dbgs() << "[transGlobalAnnotation] " << *V << '\n');

  // @llvm.global.annotations is an array that contains structs with 4 fields.
  // Get the array of structs with metadata
  // TODO: actually, now it contains 5 fields, the fifth by default is nullptr
  // or undef, but it can be defined to include variadic arguments of
  // clang::annotation attribute. Need to refactor this function to turn on this
  // translation
  ConstantArray *CA = cast<ConstantArray>(V->getOperand(0));
  for (Value *Op : CA->operands()) {
    ConstantStruct *CS = cast<ConstantStruct>(Op);
    // The first field of the struct contains a pointer to annotated variable
    Value *AnnotatedVar = CS->getOperand(0)->stripPointerCasts();
    SPIRVValue *SV = transValue(AnnotatedVar, nullptr);

    // The second field contains a pointer to a global annotation string
    GlobalVariable *GV =
        cast<GlobalVariable>(CS->getOperand(1)->stripPointerCasts());

    StringRef AnnotationString;
    getConstantStringInfo(GV, AnnotationString);
    DecorationsInfoVec Decorations =
        tryParseAnnotationString(BM, AnnotationString).MemoryAttributesVec;

    // If we didn't find any annotation decorations, let's add the whole
    // annotation string as UserSemantic Decoration
    if (Decorations.empty()) {
      SV->addDecorate(
          new SPIRVDecorateUserSemanticAttr(SV, AnnotationString.str()));
    } else {
      addAnnotationDecorations(SV, Decorations);
    }
  }
}

void LLVMToSPIRVBase::transGlobalIOPipeStorage(GlobalVariable *V, MDNode *IO) {
  SPIRVDBG(dbgs() << "[transGlobalIOPipeStorage] " << *V << '\n');
  SPIRVValue *SV = transValue(V, nullptr);
  assert(SV && "Failed to process OCL PipeStorage object");
  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_io_pipes)) {
    unsigned ID = getMDOperandAsInt(IO, 0);
    SV->addDecorate(DecorationIOPipeStorageINTEL, ID);
  }
}

bool LLVMToSPIRVBase::transGlobalVariables() {
  // add global fixed/immutable samplers array that is always present
  if (SrcLang == spv::SourceLanguageGLSL) {
    static constexpr const uint32_t fixed_sampler_count{48u};
    if (bool has_vulkan_descriptor_buffer =
            M->getNamedMetadata("floor.vulkan_descriptor_buffer");
        has_vulkan_descriptor_buffer) {
      // -> for descriptor buffer use
      auto sampler_type = BM->addPointerType(spv::StorageClassUniformConstant,
                                             BM->addSamplerType());
      for (uint32_t i = 0; i < fixed_sampler_count; ++i) {
        auto var_name = "vulkan.immutable_sampler_" + std::to_string(i);
#if defined(_DEBUG)
        using namespace vulkan_sampling;
        switch (
            sampler::FILTER_MODE(i & uint32_t(sampler::__FILTER_MODE_MASK))) {
        case sampler::FILTER_MODE::NEAREST:
          var_name += "_nearest";
          break;
        case sampler::FILTER_MODE::LINEAR:
          var_name += "_linear";
          break;
        }
        switch (
            sampler::ADDRESS_MODE(i & uint32_t(sampler::__ADDRESS_MODE_MASK))) {
        case sampler::ADDRESS_MODE::CLAMP_TO_EDGE:
          var_name += "_clamp";
          break;
        case sampler::ADDRESS_MODE::REPEAT:
          var_name += "_repeat";
          break;
        case sampler::ADDRESS_MODE::REPEAT_MIRRORED:
          var_name += "_repeat_mirrored";
          break;
        default:
          break;
        }
        switch (sampler::COMPARE_FUNCTION(
            i & uint32_t(sampler::__COMPARE_FUNCTION_MASK))) {
        case sampler::COMPARE_FUNCTION::NEVER:
          // implicitly handled as no-compare
          break;
        case sampler::COMPARE_FUNCTION::LESS:
          var_name += "_cmp(<)";
          break;
        case sampler::COMPARE_FUNCTION::EQUAL:
          var_name += "_cmp(==)";
          break;
        case sampler::COMPARE_FUNCTION::LESS_OR_EQUAL:
          var_name += "_cmp(<=)";
          break;
        case sampler::COMPARE_FUNCTION::GREATER:
          var_name += "_cmp(>)";
          break;
        case sampler::COMPARE_FUNCTION::NOT_EQUAL:
          var_name += "_cmp(!=)";
          break;
        case sampler::COMPARE_FUNCTION::GREATER_OR_EQUAL:
          var_name += "_cmp(>=)";
          break;
        case sampler::COMPARE_FUNCTION::ALWAYS:
          var_name += "_cmp(a)";
          break;
        default:
          break;
        }
#endif
        auto immutable_sampler_var =
            static_cast<SPIRVVariable *>(BM->addVariable(
                sampler_type, true, spv::internal::LinkageTypeInternal, nullptr,
                var_name, spv::StorageClassUniformConstant, nullptr));
        BM->setName(immutable_sampler_var, var_name);
        immutable_sampler_var->addDecorate(new SPIRVDecorate(
            DecorationDescriptorSet, immutable_sampler_var, 0));
        immutable_sampler_var->addDecorate(
            new SPIRVDecorate(DecorationBinding, immutable_sampler_var, i));
        immutable_samplers.emplace_back(immutable_sampler_var);
      }
    } else {
      // -> legacy
      auto samplers_type =
          BM->addPointerType(spv::StorageClassUniformConstant,
                             BM->addArrayType(BM->addSamplerType(),
                                              BM->getLiteralAsConstant(
                                                  fixed_sampler_count, false)));
      auto immutable_samplers_var =
          static_cast<SPIRVVariable *>(BM->addVariable(
              samplers_type, true, spv::internal::LinkageTypeInternal, nullptr,
              "vulkan.immutable_samplers", spv::StorageClassUniformConstant,
              nullptr));
      BM->setName(immutable_samplers_var, "vulkan.immutable_samplers");
      immutable_samplers_var->addDecorate(new SPIRVDecorate(
          DecorationDescriptorSet, immutable_samplers_var, 0));
      immutable_samplers_var->addDecorate(
          new SPIRVDecorate(DecorationBinding, immutable_samplers_var, 0));
      immutable_samplers.emplace_back(immutable_samplers_var);
    }
  }

  for (auto I = M->global_begin(), E = M->global_end(); I != E; ++I) {
    // ignore any special vulkan globals used by functions (these will be
    // handled when translating the functions)
    if ((*I).hasName() && (*I).getName().find(".vulkan") != std::string::npos)
      continue;

    // ignore any globals that need to be put into functions (map to function
    // storage class), these are handled later
    if (SPIRSPIRVAddrSpaceMap::map(static_cast<SPIRAddressSpace>(
            (*I).getType()->getAddressSpace())) == spv::StorageClassFunction)
      continue;

    // ignore external globals
    if ((*I).getLinkage() == GlobalValue::ExternalLinkage ||
        (*I).getLinkage() == GlobalValue::AvailableExternallyLinkage ||
        (*I).getLinkage() == GlobalValue::PrivateLinkage ||
        (*I).getLinkage() == GlobalValue::ExternalWeakLinkage)
      continue;

    if ((*I).getName() == "llvm.global.annotations")
      transGlobalAnnotation(&(*I));
    else if ([I]() -> bool {
               // Check if the GV is used only in var/ptr instructions. If yes -
               // skip processing of this since it's only an annotation GV.
               if (I->user_empty())
                 return false;
               for (auto *U : I->users()) {
                 Value *V = U;
                 while (isa<BitCastInst>(V) || isa<AddrSpaceCastInst>(V))
                   V = cast<CastInst>(V)->getOperand(0);
                 auto *GEP = dyn_cast_or_null<GetElementPtrInst>(V);
                 if (!GEP)
                   return false;
                 for (auto *GEPU : GEP->users()) {
                   auto *II = dyn_cast<IntrinsicInst>(GEPU);
                   if (!II)
                     return false;
                   switch (II->getIntrinsicID()) {
                   case Intrinsic::var_annotation:
                   case Intrinsic::ptr_annotation:
                     continue;
                   default:
                     return false;
                   }
                 }
               }
               return true;
             }())
      continue;
    else if ((I->getName() == "llvm.global_ctors" ||
              I->getName() == "llvm.global_dtors") &&
             !BM->isAllowedToUseExtension(
                 ExtensionID::SPV_INTEL_function_pointers)) {
      // Function pointers are required to represent structor lists; do not
      // translate the variable if function pointers are not available.
      continue;
    } else if (MDNode *IO = ((*I).getMetadata("io_pipe_id")))
      transGlobalIOPipeStorage(&(*I), IO);
    else if (!transValue(&(*I), nullptr))
      return false;
  }
  return true;
}

bool LLVMToSPIRVBase::isAnyFunctionReachableFromFunction(
    const Function *FS,
    const std::unordered_set<const Function *> Funcs) const {
  std::unordered_set<const Function *> Done;
  std::unordered_set<const Function *> ToDo;
  ToDo.insert(FS);

  while (!ToDo.empty()) {
    auto It = ToDo.begin();
    const Function *F = *It;

    if (Funcs.find(F) != Funcs.end())
      return true;

    ToDo.erase(It);
    Done.insert(F);

    const CallGraphNode *FN = (*CG)[F];
    for (unsigned I = 0; I < FN->size(); ++I) {
      const CallGraphNode *NN = (*FN)[I];
      const Function *NNF = NN->getFunction();
      if (!NNF)
        continue;
      if (Done.find(NNF) == Done.end()) {
        ToDo.insert(NNF);
      }
    }
  }

  return false;
}

void LLVMToSPIRVBase::collectInputOutputVariables(SPIRVFunction *SF,
                                                  Function *F) {
  for (auto &GV : M->globals()) {
    const auto AS = GV.getAddressSpace();
    if (AS != SPIRAS_Input && AS != SPIRAS_Output)
      continue;

    std::unordered_set<const Function *> Funcs;

    for (const auto &U : GV.uses()) {
      const Instruction *Inst = dyn_cast<Instruction>(U.getUser());
      if (!Inst)
        continue;
      Funcs.insert(Inst->getFunction());
    }

    if (isAnyFunctionReachableFromFunction(F, Funcs)) {
      SF->addVariable(ValueMap[&GV]);
    }
  }
}

void LLVMToSPIRVBase::mutateFuncArgType(
    const std::map<unsigned, Type *> &ChangedType, Function *F) {
  for (auto &I : ChangedType) {
    for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE; ++UI) {
      auto Call = dyn_cast<CallInst>(*UI);
      if (!Call)
        continue;
      auto Arg = Call->getArgOperand(I.first);
      auto OrigTy = Arg->getType();
      if (OrigTy == I.second)
        continue;
      SPIRVDBG(dbgs() << "[mutate arg type] " << *Call << ", " << *Arg << '\n');
      auto CastF = M->getOrInsertFunction(SPCV_CAST, I.second, OrigTy);
      std::vector<Value *> Args;
      Args.push_back(Arg);
      auto Cast = CallInst::Create(CastF, Args, "", Call);
      Call->replaceUsesOfWith(Arg, Cast);
      SPIRVDBG(dbgs() << "[mutate arg type] -> " << *Cast << '\n');
    }
  }
}

// Propagate contraction requirement of F up the call graph.
void LLVMToSPIRVBase::fpContractUpdateRecursive(Function *F, FPContract FPC) {
  std::queue<User *> Users;
  for (User *FU : F->users()) {
    Users.push(FU);
  }

  bool EnableLogger = FPC == FPContract::DISABLED && !Users.empty();
  if (EnableLogger) {
    SPIRVDBG(dbgs() << "[fp-contract] disabled for users of " << F->getName()
                    << '\n');
  }

  while (!Users.empty()) {
    User *U = Users.front();
    Users.pop();

    if (EnableLogger) {
      SPIRVDBG(dbgs() << "[fp-contract]   user: " << *U << '\n');
    }

    // Move from an Instruction to its Function
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Users.push(I->getFunction());
      continue;
    }

    if (Function *F = dyn_cast<Function>(U)) {
      if (!joinFPContract(F, FPC)) {
        // FP contract was not updated - no need to propagate
        // This also terminates a recursion (if any).
        if (EnableLogger) {
          SPIRVDBG(dbgs() << "[fp-contract] already disabled " << F->getName()
                          << '\n');
        }
        continue;
      }
      if (EnableLogger) {
        SPIRVDBG(dbgs() << "[fp-contract] disabled for " << F->getName()
                        << '\n');
      }
      for (User *FU : F->users()) {
        Users.push(FU);
      }
      continue;
    }

    // Unwrap a constant until we reach an Instruction.
    // This is checked after the Function, because a Function is also a
    // Constant.
    if (Constant *C = dyn_cast<Constant>(U)) {
      for (User *CU : C->users()) {
        Users.push(CU);
      }
      continue;
    }

    llvm_unreachable("Unexpected use.");
  }
}

// TODO: move this to a proper place
enum class VULKAN_STAGE : uint32_t {
  NONE = 0u,
  VERTEX = (1u << 0u),
  TESSELLATION_CONTROL = (1u << 1u),
  TESSELLATION_EVALUATION = (1u << 2u),
  GEOMETRY = (1u << 3u),
  FRAGMENT = (1u << 4u),
  KERNEL = (1u << 5u),
};
static const char *vulkan_stage_to_string(const VULKAN_STAGE &stage) {
  switch (stage) {
  case VULKAN_STAGE::VERTEX:
    return "vertex";
  case VULKAN_STAGE::TESSELLATION_CONTROL:
    return "tessellation-control";
  case VULKAN_STAGE::TESSELLATION_EVALUATION:
    return "tessellation-evaluation";
  case VULKAN_STAGE::GEOMETRY:
    return "geometry";
  case VULKAN_STAGE::FRAGMENT:
    return "fragment";
  case VULKAN_STAGE::KERNEL:
    return "kernel";
  default:
    break;
  }
  return "";
}
constexpr VULKAN_STAGE operator|(const VULKAN_STAGE &e0,
                                 const VULKAN_STAGE &e1) {
  return (VULKAN_STAGE)((typename std::underlying_type<VULKAN_STAGE>::type)e0 |
                        (typename std::underlying_type<VULKAN_STAGE>::type)e1);
}
constexpr VULKAN_STAGE &operator|=(VULKAN_STAGE &e0, const VULKAN_STAGE &e1) {
  e0 = e0 | e1;
  return e0;
}
constexpr VULKAN_STAGE operator&(const VULKAN_STAGE &e0,
                                 const VULKAN_STAGE &e1) {
  return (VULKAN_STAGE)((typename std::underlying_type<VULKAN_STAGE>::type)e0 &
                        (typename std::underlying_type<VULKAN_STAGE>::type)e1);
}
constexpr VULKAN_STAGE &operator&=(VULKAN_STAGE &e0, const VULKAN_STAGE &e1) {
  e0 = e0 & e1;
  return e0;
}

void LLVMToSPIRVBase::decorateComposite(llvm::Type *llvm_type,
                                        SPIRVType *spirv_type) {
  if (SrcLang != SourceLanguageGLSL)
    return;
  // TODO: this doesn't respect padding/alignment yet, fix it (might already
  // need to dump this info on the clang/llvm side)
  const auto &DL = M->getDataLayout();
  if (auto struct_type = dyn_cast<llvm::StructType>(llvm_type)) {
    uint32_t member_idx = 0, offset = 0;
    for (const auto &elem_type : struct_type->elements()) {
      auto spirv_elem_type =
          ((SPIRVTypeStruct *)spirv_type)->getMemberType(member_idx);

      const auto this_member_idx = member_idx++;
      const auto &member_decs = spirv_type->getMemberDecorates();
      const auto iter =
          member_decs.find({this_member_idx, spv::DecorationOffset});
      if (iter == member_decs.end()) {
        spirv_type->addMemberDecorate(this_member_idx, spv::DecorationOffset,
                                      offset);
      } else {
        // shouldn't occur as far as I can tell, but better check it to be
        // certain
        assert(iter->second->getMemberNumber() == this_member_idx &&
               iter->second->getLiteral(0) == offset &&
               "existing member decoration differs from this one");
      }
      offset += DL.getTypeStoreSize(elem_type);

      // recurse
      decorateComposite(elem_type, spirv_elem_type);
    }
  } else if (auto array_type = dyn_cast<llvm::ArrayType>(llvm_type)) {
    add_array_stride_decoration(
        spirv_type, DL.getTypeStoreSize(array_type->getElementType()));
    auto spirv_elem_type =
        (spirv_type->isTypeRuntimeArray()
             ? ((SPIRVTypeRuntimeArray *)spirv_type)->getElementType()
             : ((SPIRVTypeArray *)spirv_type)->getElementType());

    // recurse
    decorateComposite(array_type->getElementType(), spirv_elem_type);
  }
}

SPIRVVariable *LLVMToSPIRVBase::emitShaderSPIRVGlobal(
    const Function &F, SPIRVFunction *spirv_func, const GlobalVariable &GV,
    const std::string &var_name, uint32_t address_space,
    const spirv_global_io_type global_type, const std::string &md_info,
    spv::BuiltIn builtin) {
  spv::StorageClass storage_class = spv::StorageClassUniform;
  if (global_type.is_builtin) {
    storage_class = (global_type.is_input ? spv::StorageClassInput
                                          : spv::StorageClassOutput);
  } else if (global_type.is_input) {
    storage_class = spv::StorageClassInput;
  } else if (global_type.is_image) {
    storage_class = spv::StorageClassUniformConstant;
  } else if (global_type.is_uniform) {
    storage_class = spv::StorageClassUniform;
    if ((!global_type.is_constant ||
         (global_type.is_constant && !global_type.is_iub))) {
      storage_class = spv::StorageClassStorageBuffer;
    }
  } else {
    storage_class = spv::StorageClassOutput;
  }

  SPIRVType *mapped_type = nullptr;
  uint32_t fbo_location = 0;
  if (global_type.is_uniform) {
    assert(GV.getType()->isPointerTy() && "uniform must be a pointer type");
    auto elem_type = GV.getType()->getPointerElementType();

    // -> SSBOs or IUBs
    if (global_type.is_ssbo_array) {
      // -> fixed size SSBO array
      assert(elem_type->isArrayTy() && "must be an array");
      assert(elem_type->getArrayElementType()->isPointerTy() &&
             "array element type must be a pointer");
      assert(elem_type->getArrayElementType()->getPointerAddressSpace() == 0 &&
             "pointer must have no address space");

      // get array length
      const auto array_pos = md_info.find(':');
      assert(array_pos != std::string::npos);
      const auto array_length_str = md_info.substr(0, array_pos);
      const auto array_length =
          (uint64_t)strtoull(array_length_str.c_str(), nullptr, 10);

      // create an outer struct containing a runtime array of the wanted SSBO
      // elem type
      const auto ssbo_elem_type =
          elem_type->getArrayElementType()->getPointerElementType();
      const auto enclosed_type = transType(ssbo_elem_type);
      const auto enclosed_array_type = BM->addRuntimeArrayType(enclosed_type);

      std::string enclosing_type_name = "";
      llvm::raw_string_ostream type_stream(enclosing_type_name);
      ssbo_elem_type->print(type_stream, false, true);

      auto enclosing_st_type =
          BM->openStructType(1, "enclose." + enclosing_type_name);
      enclosing_st_type->setMemberType(0, enclosed_array_type);
      BM->closeStructType(enclosing_st_type, false);
      auto spirv_ssbo_elem_type = enclosing_st_type;

      auto ssbo_array_type =
          BM->addArrayType(spirv_ssbo_elem_type,
                           (SPIRVConstant *)BM->addIntegerConstant(
                               BM->addIntegerType(64u, true), array_length));
      mapped_type = BM->addPointerType(storage_class, ssbo_array_type);

      // add required deco
      enclosing_st_type->addDecorate(
          new SPIRVDecorate(DecorationBlock, enclosing_st_type));
      enclosing_st_type->addMemberDecorate(0, spv::DecorationOffset, 0);
      auto array_stride = M->getDataLayout().getTypeStoreSize(ssbo_elem_type);
      add_array_stride_decoration(enclosed_array_type, array_stride);
      decorateComposite(ssbo_elem_type, enclosed_type);
    } else if (!global_type.is_image) {
      auto spirv_elem_type = transType(elem_type);
      if (!global_type.is_iub) {
        if (!global_type.is_constant) {
          // this is a SSBO with an unknown size, switch out the top pointer
          // type with a runtime array type
          auto rtarr_type = BM->addRuntimeArrayType(spirv_elem_type);
          std::string enclosing_type_name = "enclose.";
          if (elem_type->isStructTy()) {
            enclosing_type_name += elem_type->getStructName().str();
          } else {
            std::string type_str = "";
            llvm::raw_string_ostream type_stream(type_str);
            elem_type->print(type_stream, false, true);
            enclosing_type_name += type_stream.str();
          }
          auto enclosing_type = BM->openStructType(1, enclosing_type_name);
          enclosing_type->setMemberType(0, rtarr_type);
          BM->closeStructType(enclosing_type, false);
          mapped_type = BM->addPointerType(storage_class, enclosing_type);

          // add required deco
          enclosing_type->addDecorate(
              new SPIRVDecorate(DecorationBlock, enclosing_type));
          enclosing_type->addMemberDecorate(0, spv::DecorationOffset, 0);
          auto array_stride = M->getDataLayout().getTypeStoreSize(elem_type);
          add_array_stride_decoration(rtarr_type, array_stride);
          add_array_stride_decoration(mapped_type, array_stride);
        } else {
          // we need to use the storage buffer storage class
          assert(elem_type->isStructTy() && "SSBO must be a struct");
          auto ssbo_ptr_type =
              llvm::PointerType::get(elem_type, SPIRAS_StorageBuffer);
          mapped_type = transType(ssbo_ptr_type);
          spirv_elem_type->addDecorate(
              new SPIRVDecorate(DecorationBlock, spirv_elem_type));
          add_array_stride_decoration(
              mapped_type, M->getDataLayout().getTypeStoreSize(elem_type));
        }
      } else {
        assert(elem_type->isStructTy() && "uniform type must be a struct");
        // we need to use the uniform buffer storage class
        auto uniform_ptr_type = llvm::PointerType::get(
            GV.getType()->getPointerElementType(), SPIRAS_Uniform);
        mapped_type = transType(uniform_ptr_type);
        spirv_elem_type->addDecorate(
            new SPIRVDecorate(DecorationBlock, spirv_elem_type));
      }
      decorateComposite(elem_type, spirv_elem_type);
    }
    // -> images
    else {
      const auto access_split_pos = md_info.find(':');
      const auto array_or_scalar_split_pos =
          md_info.find(':', access_split_pos + 1);
      const auto elem_count_split_pos =
          md_info.find(':', array_or_scalar_split_pos + 1);

      const auto access_type = md_info.substr(0, access_split_pos);
      const auto array_or_scalar_str =
          md_info.substr(access_split_pos + 1,
                         array_or_scalar_split_pos - access_split_pos - 1);
      const auto elem_count_str =
          md_info.substr(array_or_scalar_split_pos + 1,
                         elem_count_split_pos - array_or_scalar_split_pos - 1);
      const auto sample_type = md_info.substr(elem_count_split_pos + 1);

      const bool is_array = (array_or_scalar_str == "array");
      const bool is_write = (access_type == "write");

      if (is_write) {
        // TODO: handle storage images with format
        BM->addCapability(spv::CapabilityStorageImageWriteWithoutFormat);
      }

      //
      const auto elem_count = (uint32_t)std::stoull(elem_count_str);
      uint32_t elem_count_inner_2d = 0;
      llvm::Type *img_type = GV.getType();
      if (is_array) {
        // image array
        assert(isa<PointerType>(img_type) && "must be a pointer type");
        const auto img_array_type =
            dyn_cast<ArrayType>(img_type->getPointerElementType());
        assert(img_array_type != nullptr && "image type must be an array type");
        assert(img_array_type->getNumElements() == elem_count &&
               "invalid image array element count");

        // writable image arrays may be a 2D array
        auto img_array_elem_type = img_array_type->getArrayElementType();
        if (img_array_elem_type->isPointerTy() &&
            img_array_elem_type->getPointerElementType()->isArrayTy()) {
          auto inner_img_array_type =
              dyn_cast<ArrayType>(img_array_elem_type->getPointerElementType());
          img_type = inner_img_array_type->getArrayElementType();
          elem_count_inner_2d = inner_img_array_type->getNumElements();
        } else {
          img_type = img_array_elem_type;
        }
      }
      //
      auto SPIRVImageTy = getSPIRVImageTypeFromGLSL(
          M, img_type, sample_type.c_str(), is_write, spv::ImageFormatUnknown);
      auto transSPIRVImageTy = transSPIRVOpaqueType(SPIRVImageTy);

      // cache it
      image_type_map.emplace(&GV, transSPIRVImageTy);

      //
      auto ptr_img_type = transSPIRVImageTy;
      if (is_array) {
        if (elem_count_inner_2d == 0) {
          ptr_img_type = BM->addArrayType(
              transSPIRVImageTy, BM->getLiteralAsConstant(elem_count, false));
        } else {
          ptr_img_type = BM->addArrayType(
              BM->addArrayType(
                  transSPIRVImageTy,
                  BM->getLiteralAsConstant(elem_count_inner_2d, false)),
              BM->getLiteralAsConstant(elem_count, false));
        }
      }

      mapped_type =
          BM->addPointerType(spv::StorageClassUniformConstant, ptr_img_type);
    }
  } else if (global_type.is_fbo_color) {
    // extract location idx
    const auto location_pos = md_info.rfind(':');
    assert(location_pos != std::string::npos);
    const auto location_str = md_info.substr(location_pos + 1);
    fbo_location = (uint32_t)strtoull(location_str.c_str(), nullptr, 10);

    // extract data type
    const auto data_type_pos = md_info.rfind(':', location_pos - 1);
    assert(data_type_pos != std::string::npos);
    const auto data_type_str =
        md_info.substr(data_type_pos + 1, location_pos - data_type_pos - 1);

    // float and (signed) int can always be translated directly, unsigned int
    // needs special treatment, b/c llvm doesn't differentiate ints and uints
    if (data_type_str == "uint") {
      assert(GV.getType()->isPointerTy());
      mapped_type = BM->addPointerType(
          storage_class, addSignPreservingLLVMType(
                             GV.getType()->getPointerElementType(), false));
    } else {
      mapped_type = transType(GV.getType());
    }

  } else if (global_type.is_fbo_depth) {
    // extract depth qualifier
    const auto depth_qual_pos = md_info.rfind(':');
    assert(depth_qual_pos != std::string::npos);
    const auto depth_qual = md_info.substr(depth_qual_pos + 1);

    // add execution mode for "less" and "greater"
    if (depth_qual == "less") {
      spirv_func->addExecutionMode(
          new SPIRVExecutionMode(spirv_func, ExecutionModeDepthLess));
    } else if (depth_qual == "greater") {
      spirv_func->addExecutionMode(
          new SPIRVExecutionMode(spirv_func, ExecutionModeDepthGreater));
    }
    // else: "any"/default, keep as-is

    mapped_type = transType(GV.getType());

  } else {
    mapped_type = transType(GV.getType());
  }

  auto BVar = static_cast<SPIRVVariable *>(
      BM->addVariable(mapped_type, false, spv::internal::LinkageTypeInternal,
                      nullptr, GV.getName().str(), storage_class, nullptr));
  BM->setName(BVar, GV.getName().str());
  mapValue((const Value *)&GV, BVar);

  if (global_type.is_builtin) {
    BVar->setBuiltin(builtin);
    if (builtin == spv::BuiltInBaryCoordKHR) {
      BM->addExtension(ExtensionID::SPV_KHR_fragment_shader_barycentric);
    }
  }

  BM->addEntryPointIO(spirv_func->getId(), BVar);

  // set non-readable/-writable deco on SSBOs
  if (global_type.is_uniform && !global_type.is_image) {
    if (global_type.is_read_only || global_type.is_constant) {
      BVar->addDecorate(new SPIRVDecorate(DecorationNonWritable, BVar));
      // IUB is always uniform
      if (global_type.is_constant && global_type.is_iub) {
        BVar->addDecorate(new SPIRVDecorate(DecorationUniform, BVar));
      }
    } else if (global_type.is_write_only) {
      BVar->addDecorate(new SPIRVDecorate(DecorationNonReadable, BVar));
    }
  }

  // handle decoration
  if (global_type.is_fbo_color) {
    BVar->addDecorate(
        new SPIRVDecorate(DecorationLocation, BVar, fbo_location));
  } else if ((storage_class == spv::StorageClassOutput ||
              storage_class == spv::StorageClassInput) &&
             global_type.set_location) {
    BVar->addDecorate(
        new SPIRVDecorate(DecorationLocation, BVar, global_type.location));
  }

  // automatically add the "flat" decoration on types that need it
  // NOTE: vulkan requires that this is only set on input variables
  // NOTE: for fragment shaders, also do this for builtin input
  if (storage_class == spv::StorageClassInput && !global_type.is_fbo_color &&
      !global_type.is_fbo_depth &&
      (!global_type.is_builtin ||
       (global_type.is_builtin &&
        F.getCallingConv() == llvm::CallingConv::FLOOR_FRAGMENT))) {
    // I/O should always be a pointer type
    if (GV.getType()->isPointerTy()) {
      auto elem_type = GV.getType()->getPointerElementType();
      auto elem_vec_type = dyn_cast_or_null<FixedVectorType>(elem_type);
      if (elem_type->isIntegerTy() ||
          (elem_vec_type && elem_vec_type->getElementType()->isIntegerTy())) {
        BVar->addDecorate(new SPIRVDecorate(DecorationFlat, BVar));
      }
    }
  }

  return BVar;
}

std::pair<GlobalVariable *, SPIRVVariable *> LLVMToSPIRVBase::emitShaderGlobal(
    const Function &F, SPIRVFunction *spirv_func, const std::string &var_name,
    llvm::Type *llvm_type, uint32_t address_space,
    const spirv_global_io_type global_type, const std::string &md_info,
    spv::BuiltIn builtin) {
  // check if already emitted
  if (global_type.is_builtin) {
    assert(builtin != spv::BuiltIn::BuiltInMax);
    auto gv_iter = builtin_gv_cache.find(builtin);
    if (gv_iter != builtin_gv_cache.end()) {
      return gv_iter->second;
    }
  }

  std::string name_type = ".";
  if (global_type.is_builtin) {
    name_type = (global_type.is_input ? ".vulkan_builtin_input."
                                      : ".vulkan_builtin_output.");
  } else if (global_type.is_input) {
    name_type = ".vulkan_input.";
  } else if (global_type.is_uniform) {
    name_type = ".vulkan_uniform.";
  }

  auto GV = new GlobalVariable(
      *M, llvm_type, false, GlobalVariable::ExternalWeakLinkage, nullptr,
      F.getName().str() + name_type + var_name, nullptr,
      GlobalValue::NotThreadLocal, address_space);

  // also add the SPIR-V global
  auto spirv_var =
      emitShaderSPIRVGlobal(F, spirv_func, *GV, var_name, address_space,
                            global_type, md_info, builtin);

  // add to cache
  if (global_type.is_builtin) {
    builtin_gv_cache.emplace(builtin, std::pair{GV, spirv_var});
  }

  return {GV, spirv_var};
}

// helper function to figure out if a SSBO argument is only being written to
// TODO/NOTE: since WriteOnly is a fairly new attribute, the FunctionAttrs pass
// can't handle it yet (like it does for readonly/readnone) -> once it can infer
// the WriteOnly attribute, use that instead
static bool is_write_only_arg(Function &F, Argument &arg) {
  // since Vulkan/SPIR-V is very restrictive on pointer usage, that makes this
  // rather simple. however, we still bail out if we find something that we
  // can't handle.
  const std::function<bool(Value *)> user_recurse =
      [&user_recurse](Value *val) {
        for (User *user : val->users()) {
          // is read from -> bail
          if (isa<LoadInst>(user)) {
            return false;
          }
          // is written to -> continue
          else if (isa<StoreInst>(user)) {
            continue;
          }
          // recurse for GEPs
          else if (isa<GetElementPtrInst>(user)) {
            // bail if GEP is used for loads
            if (!user_recurse(user)) {
              return false;
            }
          }
          // calls are somewhat tricky
          else if (CallInst *CI = dyn_cast<CallInst>(user)) {
            // does read -> bail
            if (!CI->onlyWritesMemory()) {
              return false;
            }
            // we don't know what the call is doing exactly, but if it does
            // return a pointer, assume it's us
            if (CI->getType()->isPointerTy()) {
              if (!user_recurse(user)) {
                return false;
              }
            }
          }
          // NOTE: Vulkan/SPIR-V doesn't allow pointer usage in
          // select/phi/bitcast, so we're good here
          // unknown usage -> assume it's being read
          else {
            return false;
          }
        }
        // didn't find any loads -> write-only
        return true;
      };
  return user_recurse(&arg);
}

void LLVMToSPIRVBase::transFunction(Function *F) {
  // again, ignore any floor.* functions
  if (F->getName().startswith("floor."))
    return;

  // ignore any non-entry-point functions
  if (F->getCallingConv() == CallingConv::FLOOR_FUNC)
    return;

  SPIRVFunction *BF = transFunctionDecl(F);

  // we're only interested in shader entry points here
  // TODO: cleanup + move to functions
  if (SrcLang == SourceLanguageGLSL &&
      (F->getCallingConv() == llvm::CallingConv::FLOOR_KERNEL ||
       F->getCallingConv() == llvm::CallingConv::FLOOR_VERTEX ||
       F->getCallingConv() == llvm::CallingConv::FLOOR_FRAGMENT ||
       F->getCallingConv() == llvm::CallingConv::FLOOR_TESS_CONTROL ||
       F->getCallingConv() == llvm::CallingConv::FLOOR_TESS_EVAL)) {
    VULKAN_STAGE stage;
    switch (F->getCallingConv()) {
    case llvm::CallingConv::FLOOR_VERTEX:
      stage = VULKAN_STAGE::VERTEX;
      break;
    case llvm::CallingConv::FLOOR_FRAGMENT:
      stage = VULKAN_STAGE::FRAGMENT;
      break;
    case llvm::CallingConv::FLOOR_TESS_CONTROL:
      stage = VULKAN_STAGE::TESSELLATION_CONTROL;
      break;
    case llvm::CallingConv::FLOOR_TESS_EVAL:
      stage = VULKAN_STAGE::TESSELLATION_EVALUATION;
      break;
    case llvm::CallingConv::FLOOR_KERNEL:
      stage = VULKAN_STAGE::KERNEL;
      break;
    default:
      return;
    }

    // always add this
    if (stage == VULKAN_STAGE::FRAGMENT) {
      BF->addExecutionMode(
          new SPIRVExecutionMode(BF, ExecutionModeOriginUpperLeft));
    }

    const std::string func_name = F->getName().str();
    std::vector<std::string> md_data_input, md_data_output;
    auto vulkan_io_md = M->getNamedMetadata("vulkan.stage_io");
    assert(vulkan_io_md != nullptr && "vulkan.io metadata doesn't exist");
    for (const auto &op : vulkan_io_md->operands()) {
      assert(op->getNumOperands() > 0 &&
             "invalid op count in vulkan.io metadata");
      if (auto md_func_name = dyn_cast<llvm::MDString>(op->getOperand(0))) {
        if (md_func_name->getString() == func_name) {
          // found our function, dump metadata strings to an easier to use
          // vector<string>
          bool at_input = false, at_output = false;
          for (uint32_t i = 1; i < op->getNumOperands(); ++i) {
            const auto md_op_str =
                dyn_cast<llvm::MDString>(op->getOperand(i))->getString();

            if (md_op_str == "stage_input") {
              at_input = true;
              at_output = false;
              continue;
            } else if (md_op_str == "stage_output") {
              at_input = false;
              at_output = true;
              continue;
            }

            if (at_input)
              md_data_input.emplace_back(md_op_str.str());
            else if (at_output)
              md_data_output.emplace_back(md_op_str.str());
          }
          break;
        }
      }
    }

    const auto get_builtin =
        [](const std::string &str) -> std::pair<spv::BuiltIn, bool> {
      static const std::unordered_map<std::string, spv::BuiltIn> builtin_lut{
          {"position", spv::BuiltInPosition},
          {"point_size", spv::BuiltInPointSize},
          {"clip_distance", spv::BuiltInClipDistance},
          {"cull_distance", spv::BuiltInCullDistance},
          //{ "vertex_id", spv::BuiltInVertexId }, // unsupported in vulkan
          //{ "instance_id", spv::BuiltInInstanceId }, // unsupported in vulkan
          {"primitive_id", spv::BuiltInPrimitiveId},
          {"invocation_id", spv::BuiltInInvocationId},
          {"layer", spv::BuiltInLayer},
          {"viewport_index", spv::BuiltInViewportIndex},
          {"tess_level_outer", spv::BuiltInTessLevelOuter},
          {"tess_level_inner", spv::BuiltInTessLevelInner},
          {"tess_coord", spv::BuiltInTessCoord},
          {"patch_vertices", spv::BuiltInPatchVertices},
          {"frag_coord", spv::BuiltInFragCoord},
          {"point_coord", spv::BuiltInPointCoord},
          {"front_facing", spv::BuiltInFrontFacing},
          {"sample_id", spv::BuiltInSampleId},
          {"sample_position", spv::BuiltInSamplePosition},
          {"sample_mask", spv::BuiltInSampleMask},
          {"frag_depth", spv::BuiltInFragDepth},
          {"helper_invocation", spv::BuiltInHelperInvocation},
          {"num_workgroups", spv::BuiltInNumWorkgroups},
          //{ "workgroup_size", spv::BuiltInWorkgroupSize }, // NOTE: must be a
          // constant or spec constant
          {"workgroup_id", spv::BuiltInWorkgroupId},
          //{"local_invocation_id", spv::BuiltInLocalInvocationId},
          //{"global_invocation_id", spv::BuiltInGlobalInvocationId},
          // OpenCL-only:
          //{ "local_invocation_index", spv::BuiltInLocalInvocationIndex },
          //{ "work_dim", spv::BuiltInWorkDim },
          //{ "global_size", spv::BuiltInGlobalSize },
          //{ "enqueued_workgroup_size", spv::BuiltInEnqueuedWorkgroupSize },
          //{ "global_offset", spv::BuiltInGlobalOffset },
          //{ "global_linear_id", spv::BuiltInGlobalLinearId },
          //{ "subgroup_size", spv::BuiltInSubgroupSize },
          //{ "subgroup_max_size", spv::BuiltInSubgroupMaxSize },
          //{ "num_subgroups", spv::BuiltInNumSubgroups },
          //{ "num_enqueued_subgroups", spv::BuiltInNumEnqueuedSubgroups },
          //{ "subgroup_id", spv::BuiltInSubgroupId },
          //{ "subgroup_local_invocation_id",
          // spv::BuiltInSubgroupLocalInvocationId },
          {"vertex_index", spv::BuiltInVertexIndex},
          {"instance_index", spv::BuiltInInstanceIndex},
          {"view_index", spv::BuiltInViewIndex},
          {"barycentric_coord", spv::BuiltInBaryCoordKHR},
          {"sub_group_id", spv::BuiltInSubgroupId},
          {"sub_group_local_id", spv::BuiltInSubgroupLocalInvocationId},
          {"sub_group_size", spv::BuiltInSubgroupSize},
          {"num_sub_groups", spv::BuiltInNumSubgroups},
      };
      const auto iter = builtin_lut.find(str);
      if (iter == builtin_lut.end()) {
        return {spv::BuiltInPosition, false};
      }
      return {iter->second, true};
    };
    const auto is_builtin_valid_in_stage = [](const spv::BuiltIn &builtin,
                                              const VULKAN_STAGE &stage,
                                              const bool is_input) {
      // NOTE: the non-listed/commented ones are unsupported in vulkan
      static const std::unordered_map<spv::BuiltIn, VULKAN_STAGE>
          builtin_validity_input_lut{
              {spv::BuiltInPosition, (VULKAN_STAGE::TESSELLATION_CONTROL |
                                      VULKAN_STAGE::TESSELLATION_EVALUATION |
                                      VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInPointSize, (VULKAN_STAGE::TESSELLATION_CONTROL |
                                       VULKAN_STAGE::TESSELLATION_EVALUATION |
                                       VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInClipDistance,
               (VULKAN_STAGE::FRAGMENT | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInCullDistance,
               (VULKAN_STAGE::FRAGMENT | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              //{ spv::BuiltInVertexId, VULKAN_STAGE::NONE },
              //{ spv::BuiltInInstanceId, VULKAN_STAGE::NONE },
              {spv::BuiltInPrimitiveId,
               VULKAN_STAGE::GEOMETRY | VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInInvocationId,
               (VULKAN_STAGE::TESSELLATION_CONTROL | VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInLayer, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInViewportIndex, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInTessLevelOuter,
               VULKAN_STAGE::TESSELLATION_EVALUATION},
              {spv::BuiltInTessLevelInner,
               VULKAN_STAGE::TESSELLATION_EVALUATION},
              {spv::BuiltInTessCoord, VULKAN_STAGE::TESSELLATION_EVALUATION},
              {spv::BuiltInPatchVertices,
               (VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION)},
              {spv::BuiltInFragCoord, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInPointCoord, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInFrontFacing, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInSampleId, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInSamplePosition, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInSampleMask, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInFragDepth, VULKAN_STAGE::NONE},
              {spv::BuiltInHelperInvocation, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInNumWorkgroups, VULKAN_STAGE::KERNEL},
              {spv::BuiltInWorkgroupSize,
               VULKAN_STAGE::NONE}, // NOTE: must be a constant or spec constant
              {spv::BuiltInWorkgroupId, VULKAN_STAGE::KERNEL},
              {spv::BuiltInLocalInvocationId, VULKAN_STAGE::KERNEL},
              {spv::BuiltInGlobalInvocationId, VULKAN_STAGE::KERNEL},
              //{ spv::BuiltInLocalInvocationIndex, VULKAN_STAGE::NONE },
              //{ spv::BuiltInWorkDim, VULKAN_STAGE::NONE },
              //{ spv::BuiltInGlobalSize, VULKAN_STAGE::NONE },
              //{ spv::BuiltInEnqueuedWorkgroupSize, VULKAN_STAGE::NONE },
              //{ spv::BuiltInGlobalOffset, VULKAN_STAGE::NONE },
              //{ spv::BuiltInGlobalLinearId, VULKAN_STAGE::NONE },
              //{ spv::BuiltInSubgroupMaxSize, VULKAN_STAGE::NONE },
              //{ spv::BuiltInNumEnqueuedSubgroups, VULKAN_STAGE::NONE },
              {spv::BuiltInVertexIndex, VULKAN_STAGE::VERTEX},
              {spv::BuiltInInstanceIndex, VULKAN_STAGE::VERTEX},
              {spv::BuiltInViewIndex,
               (VULKAN_STAGE::VERTEX | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION | VULKAN_STAGE::GEOMETRY |
                VULKAN_STAGE::FRAGMENT)},
              {spv::BuiltInBaryCoordKHR, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInSubgroupId, VULKAN_STAGE::KERNEL},
              {spv::BuiltInSubgroupLocalInvocationId, VULKAN_STAGE::KERNEL},
              {spv::BuiltInSubgroupSize, VULKAN_STAGE::KERNEL},
              {spv::BuiltInNumSubgroups, VULKAN_STAGE::KERNEL},
          };
      static const std::unordered_map<spv::BuiltIn, VULKAN_STAGE>
          builtin_validity_output_lut{
              {spv::BuiltInPosition,
               (VULKAN_STAGE::VERTEX | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInPointSize,
               (VULKAN_STAGE::VERTEX | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInClipDistance,
               (VULKAN_STAGE::VERTEX | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInCullDistance,
               (VULKAN_STAGE::VERTEX | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              //{ spv::BuiltInVertexId, VULKAN_STAGE::NONE },
              //{ spv::BuiltInInstanceId, VULKAN_STAGE::NONE },
              {spv::BuiltInPrimitiveId,
               (VULKAN_STAGE::FRAGMENT | VULKAN_STAGE::TESSELLATION_CONTROL |
                VULKAN_STAGE::TESSELLATION_EVALUATION |
                VULKAN_STAGE::GEOMETRY)},
              {spv::BuiltInInvocationId, VULKAN_STAGE::NONE},
              {spv::BuiltInLayer, VULKAN_STAGE::GEOMETRY},
              {spv::BuiltInViewportIndex, VULKAN_STAGE::GEOMETRY},
              {spv::BuiltInTessLevelOuter, VULKAN_STAGE::TESSELLATION_CONTROL},
              {spv::BuiltInTessLevelInner, VULKAN_STAGE::TESSELLATION_CONTROL},
              {spv::BuiltInTessCoord, VULKAN_STAGE::NONE},
              {spv::BuiltInPatchVertices, VULKAN_STAGE::NONE},
              {spv::BuiltInFragCoord, VULKAN_STAGE::NONE},
              {spv::BuiltInPointCoord, VULKAN_STAGE::NONE},
              {spv::BuiltInFrontFacing, VULKAN_STAGE::NONE},
              {spv::BuiltInSampleId, VULKAN_STAGE::NONE},
              {spv::BuiltInSamplePosition, VULKAN_STAGE::NONE},
              {spv::BuiltInSampleMask, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInFragDepth, VULKAN_STAGE::FRAGMENT},
              {spv::BuiltInHelperInvocation, VULKAN_STAGE::NONE},
              {spv::BuiltInNumWorkgroups, VULKAN_STAGE::NONE},
              {spv::BuiltInWorkgroupSize, VULKAN_STAGE::NONE},
              {spv::BuiltInWorkgroupId, VULKAN_STAGE::NONE},
              {spv::BuiltInLocalInvocationId, VULKAN_STAGE::NONE},
              {spv::BuiltInGlobalInvocationId, VULKAN_STAGE::NONE},
              //{ spv::BuiltInLocalInvocationIndex, VULKAN_STAGE::NONE },
              //{ spv::BuiltInWorkDim, VULKAN_STAGE::NONE },
              //{ spv::BuiltInGlobalSize, VULKAN_STAGE::NONE },
              //{ spv::BuiltInEnqueuedWorkgroupSize, VULKAN_STAGE::NONE },
              //{ spv::BuiltInGlobalOffset, VULKAN_STAGE::NONE },
              //{ spv::BuiltInGlobalLinearId, VULKAN_STAGE::NONE },
              //{ spv::BuiltInSubgroupMaxSize, VULKAN_STAGE::NONE },
              //{ spv::BuiltInNumEnqueuedSubgroups, VULKAN_STAGE::NONE },
              {spv::BuiltInVertexIndex, VULKAN_STAGE::NONE},
              {spv::BuiltInInstanceIndex, VULKAN_STAGE::NONE},
              {spv::BuiltInViewIndex, VULKAN_STAGE::NONE},
          };

      if (is_input) {
        const auto iter = builtin_validity_input_lut.find(builtin);
        if (iter == builtin_validity_input_lut.end()) {
          return false;
        }
        return (iter->second & stage) != VULKAN_STAGE::NONE;
      } else {
        const auto iter = builtin_validity_output_lut.find(builtin);
        if (iter == builtin_validity_output_lut.end()) {
          return false;
        }
        return (iter->second & stage) != VULKAN_STAGE::NONE;
      }
    };

    // handle parameters (input or globals)
    // TODO: proper input/output location handling: 1x - 4x 32-bit values can be
    // packed into a single location, anything larger needs to be distributed
    // over multiple locations
    uint32_t input_arg_idx = 0, uniform_arg_idx = 0, arg_buffer_arg_idx = 0;
    uint32_t input_location = 0;
    uint32_t desc_set = 0;
    uint32_t arg_buffer_desc_set_offset = 0;
    uint32_t arg_buffer_desc_set = 0;
    switch (stage) { // put each stage into a different set
    case VULKAN_STAGE::KERNEL:
      desc_set = 1;
      arg_buffer_desc_set_offset = 2; // [2, 15]
      break;
    case VULKAN_STAGE::VERTEX:
      desc_set = 1;
      arg_buffer_desc_set_offset = 5; // [5, 8]
      break;
    case VULKAN_STAGE::FRAGMENT:
      desc_set = 2;
      arg_buffer_desc_set_offset = 9; // [9, 12]
      break;
    case VULKAN_STAGE::GEOMETRY:
    case VULKAN_STAGE::TESSELLATION_CONTROL:
      // will never support geometry shaders + tessellation at the same time
      desc_set = 3;
      arg_buffer_desc_set_offset = 13; // [13, 15]
      break;
    case VULKAN_STAGE::TESSELLATION_EVALUATION:
      desc_set = 4;
      break;
    default:
      llvm_unreachable("invalid stage");
    }
    arg_buffer_desc_set = arg_buffer_desc_set_offset;
    for (Argument &arg : F->args()) {
      llvm::Type *arg_type = arg.getType();
      const auto arg_name = arg.getName();

      const auto &arg_md_in = md_data_input[input_arg_idx];
      const auto md_prefix_split_pos = arg_md_in.find(':');
      std::string md_prefix = (md_prefix_split_pos != std::string::npos
                                   ? arg_md_in.substr(0, md_prefix_split_pos)
                                   : "");

      std::string md_info = (md_prefix_split_pos != std::string::npos
                                 ? arg_md_in.substr(md_prefix_split_pos + 1)
                                 : arg_md_in);

      // descriptor set + binding for this arg
      uint32_t *arg_desc_set = &desc_set;
      uint32_t *arg_idx = &uniform_arg_idx;

      // argument buffer pre-handling
      if (md_prefix == "argbuf") {
        // extract argument buffer index
        const auto arg_buf_idx_split_pos = md_info.find(':');
        const auto actual_md_prefix_split_pos =
            md_info.find(':', arg_buf_idx_split_pos + 1);
        if (arg_buf_idx_split_pos == std::string::npos ||
            actual_md_prefix_split_pos == std::string::npos) {
          errs() << "invalid argument buffer metadata: " << arg_md_in << "\n";
          errs().flush();
          assert(false && "invalid argument buffer metadata");
          return;
        }
        const std::string arg_buf_idx_str =
            md_info.substr(0, arg_buf_idx_split_pos);
        const auto arg_buf_idx = std::stoull(arg_buf_idx_str);

        // handle argument buffer index
        assert(arg_buf_idx + arg_buffer_desc_set_offset < 16);
        const auto this_arg_buffer_desc_set =
            arg_buf_idx + arg_buffer_desc_set_offset;
        if (arg_buffer_desc_set != this_arg_buffer_desc_set) {
          // -> next arg buffer, reset arg index
          arg_buffer_arg_idx = 0;
        }
        arg_buffer_desc_set = this_arg_buffer_desc_set;

        // set argument buffer descriptor set + binding for this arg
        arg_desc_set = &arg_buffer_desc_set;
        arg_idx = &arg_buffer_arg_idx;

        // advance in metadata to get the proper/normal metadata prefix and info
        md_prefix = md_info.substr(arg_buf_idx_split_pos + 1,
                                   actual_md_prefix_split_pos -
                                       arg_buf_idx_split_pos - 1);
        md_info = md_info.substr(actual_md_prefix_split_pos + 1);
      }

      if (arg_type->isPointerTy() &&
          arg_type->getPointerAddressSpace() != SPIRAS_Input) {
        llvm::Type *elem_type = arg_type->getPointerElementType();

        // -> globals
        SPIRVVariable *uniform_var = nullptr;
        const auto ptr_as = arg_type->getPointerAddressSpace();
        if (arg.onlyReadsMemory() &&
            (arg.hasAttribute(Attribute::Dereferenceable) ||
             arg.hasAttribute(Attribute::DereferenceableOrNull))) {
          // -> uniform, use static/fixed SSBO
          // NOTE: this could be made a Block variable, but that would have
          // insane alignment/offset requirements, so always make it a SSBO,
          // which has less restrictions
          // (TODO: could also make this a push constant later on)
          spirv_global_io_type global_type;
          global_type.is_constant = true;
          global_type.is_uniform = true;
          global_type.is_read_only = true;
          auto storage_class = SPIRAS_StorageBuffer;
          if (md_prefix == "iub") {
            global_type.is_iub = true;
            storage_class = SPIRAS_Uniform;
          } else {
            assert(md_prefix == "ssbo");
          }
          GlobalVariable *GV = nullptr;
          std::tie(GV, uniform_var) =
              emitShaderGlobal(*F, BF, arg_name.str(), elem_type, storage_class,
                               global_type, md_info);
          arg.replaceAllUsesWith(GV, true);
        } else if (ptr_as == SPIRAS_Uniform || ptr_as == SPIRAS_StorageBuffer) {
          // all image types are opaque/unsized
          if (!elem_type->isSized()) {
            // -> image
            assert(ptr_as == SPIRAS_Uniform);
            spirv_global_io_type global_type;
            global_type.is_image = true;
            global_type.is_constant = true;
            global_type.is_uniform = true;
            GlobalVariable *GV = nullptr;
            std::tie(GV, uniform_var) =
                emitShaderGlobal(*F, BF, arg_name.str(), elem_type,
                                 SPIRAS_Uniform, global_type, md_info);
            arg.replaceAllUsesWith(GV);
          } else {
            // -> SSBO
            spirv_global_io_type global_type;
            global_type.is_uniform = true;
            global_type.is_ssbo_array = (md_prefix == "ssbo_array");
            global_type.is_read_only = arg.onlyReadsMemory();
            global_type.is_write_only =
                (!global_type.is_read_only ? is_write_only_arg(*F, arg)
                                           : false);
            const auto storage_class = SPIRAS_StorageBuffer;
            GlobalVariable *GV = nullptr;
            std::tie(GV, uniform_var) =
                emitShaderGlobal(*F, BF, arg_name.str(), elem_type,
                                 storage_class, global_type, md_info);
            // any GEPs can be directly replaced with the new GV, all others can
            // no longer access it directly, but must go through a new "GEP #0"
            // access
            while (!arg.user_empty()) {
              auto user = *arg.user_begin();
              // simple GEP ptr replacement
              if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(user)) {
                GEP->setOperand(0, GV);
                continue;
              } else if (auto ssbo_array_gep = dyn_cast<CallInst>(user);
                         ssbo_array_gep &&
                         ssbo_array_gep->getCalledFunction()
                             ->getName()
                             .startswith("floor.ssbo_array_gep.")) {
                ssbo_array_gep->setOperand(0, GV);
                continue;
              }
              // direct access
              if (Instruction *instr = dyn_cast<Instruction>(user)) {
                // create GEP to the first element
                llvm::Value *idx_list[]{
                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(*Ctx), 0),
                };
                auto zero_gep = GetElementPtrInst::CreateInBounds(
                    GV->getType()->getScalarType()->getPointerElementType(), GV,
                    idx_list, "", instr);
                zero_gep->setDebugLoc(instr->getDebugLoc());

                // replace all uses in the instruction
                for (uint32_t op_idx = 0; op_idx < instr->getNumOperands();
                     ++op_idx) {
                  auto op = instr->getOperand(op_idx);
                  if (op == &arg) {
                    instr->setOperand(op_idx, zero_gep);
                  }
                }
                continue;
              }
              assert(false && "should not be here - unknown arg user");
            }
          }
        } else if (arg_type->getPointerAddressSpace() == SPIRAS_Private &&
                   arg_type->getPointerElementType()->isArrayTy()) {
          auto array_type = cast<ArrayType>(arg_type->getPointerElementType());

          auto array_elem_type = array_type->getArrayElementType();
          assert(array_elem_type->isPointerTy() &&
                 "expected array element to be a pointer type");
          auto array_elem_ptr_type = array_elem_type->getPointerElementType();
          llvm::Type *image_or_buffer_type = nullptr;
          bool is_image = false;
          if (array_elem_ptr_type->isStructTy() &&
              !array_elem_ptr_type->isSized()) {
            // -> 1D array of images
            image_or_buffer_type = array_elem_ptr_type;
            is_image = true;
          } else if (array_elem_ptr_type->isArrayTy()) {
            // -> 2D array of images
            auto inner_arr_elem_type =
                array_elem_ptr_type->getArrayElementType();
            assert(inner_arr_elem_type->isPointerTy() &&
                   "expected inner array element to be a pointer type");
            image_or_buffer_type = inner_arr_elem_type->getPointerElementType();
            is_image = true;
          } else if (array_elem_type->getPointerAddressSpace() != 0 &&
                     array_elem_ptr_type->isSized()) {
            image_or_buffer_type = array_elem_type;
          } else {
            assert(false && "unexpected array element type");
          }

          if (is_image) {
            assert(image_or_buffer_type->isStructTy() &&
                   "expected image struct type");
            auto st_img_type = cast<StructType>(image_or_buffer_type);
            assert(st_img_type->getStructName().startswith("opencl.image") &&
                   "expected image type");
          }

          // -> image/buffer array
          spirv_global_io_type global_type;
          global_type.is_image = is_image;
          global_type.is_constant = true;
          global_type.is_uniform = true;
          GlobalVariable *GV = nullptr;
          std::tie(GV, uniform_var) =
              emitShaderGlobal(*F, BF, arg_name.str(), array_type,
                               SPIRAS_Uniform, global_type, md_info);
          // replace with address space change (private to uniform)
          arg.replaceAllUsesWith(GV, true);
        } else if (arg_type->getPointerAddressSpace() == SPIRAS_Local) {
          // -> local memory (TODO: implement this)
          llvm_unreachable("local memory parameters are not yet implemented");
        } else if (arg_type->getPointerAddressSpace() == SPIRAS_Generic) {
          // -> unknown generic
          llvm_unreachable("generic parameters are not supported");
        } else {
          llvm_unreachable("unknown parameter address space");
        }

        //
        uniform_var->addDecorate(new SPIRVDecorate(DecorationDescriptorSet,
                                                   uniform_var, *arg_desc_set));
        uniform_var->addDecorate(
            new SPIRVDecorate(DecorationBinding, uniform_var, *arg_idx));
        ++*arg_idx;
      } else {
        if (md_prefix == "builtin") {
          // -> special input variable
          // transform function parameter to in-function alloca + input
          // annotation
          const auto builtin = get_builtin(md_info);
          if (!builtin.second) {
            errs() << "unknown builtin: " << md_info << "\n";
          }

          const auto is_valid =
              is_builtin_valid_in_stage(builtin.first, stage, true /* input */);
          if (is_valid) {
            llvm::Type *elem_type = arg_type->getPointerElementType();
            spirv_global_io_type global_type;
            global_type.is_input = true;
            global_type.is_builtin = true;
            auto [repl_var, _] = emitShaderGlobal(
                *F, BF, arg_name.str(), elem_type, SPIRAS_Input, global_type,
                md_info, builtin.first);
            arg.replaceAllUsesWith(repl_var);
          } else {
            // TODO: should catch this earlier
            if (arg.getNumUses() > 0) {
              errs() << "input builtin \"" << md_info
                     << "\" can not be used in stage \""
                     << vulkan_stage_to_string(stage) << "\"\n";
            }
          }
        } else if (md_prefix == "stage" || md_prefix == "") {
          // -> stage input
          spirv_global_io_type global_type;
          global_type.is_input = true;
          global_type.is_read_only = true;
          if (md_prefix != "stage") {
            // only emit this input if it is an actual input (not a builtin)
            global_type.set_location = true;
            global_type.location = input_location++;

            auto [repl_var, _] =
                emitShaderGlobal(*F, BF, arg_name.str(), arg_type, SPIRAS_Input,
                                 global_type, md_info);
            // only emit load if there actually is a user
            if (arg.getNumUses() > 0) {
              LoadInst *load_repl_var = new LoadInst(
                  arg_type, repl_var, arg_name, false, &*(F->front().begin()));
              arg.replaceAllUsesWith(load_repl_var);
            }
          } else if (md_prefix == "stage" && md_info == "position") {
            if (arg.getNumUses() != 0) {
              if (!is_builtin_valid_in_stage(spv::BuiltInFragCoord, stage,
                                             true /* input */)) {
                errs() << "frag coord can not be used in stage \""
                       << vulkan_stage_to_string(stage) << "\"\n";
              } else {
                // -> replace uses with builtin frag coord
                llvm::Type *elem_type =
                    FixedVectorType::get(Type::getFloatTy(M->getContext()), 4u);
                spirv_global_io_type global_type;
                global_type.is_input = true;
                global_type.is_builtin = true;
                auto [repl_var, _] = emitShaderGlobal(
                    *F, BF, "vulkan.frag_coord", elem_type, SPIRAS_Input,
                    global_type, "frag_coord", spv::BuiltInFragCoord);
                LoadInst *load_repl_var =
                    new LoadInst(arg_type, repl_var, "frag_coord", false,
                                 &*(F->front().begin()));
                arg.replaceAllUsesWith(load_repl_var);
              }
            }
          } else {
            // builtin input -> must be ignored
            if (arg.getNumUses() != 0) {
              errs() << "stage input should not have any users (must use "
                        "built-ins)\n";
              assert(
                  false &&
                  "stage input should not have any users (must use built-ins)");
            }
          }
        } else {
          assert(false && "unknown or unhandled input");
        }
      }

      // used by all arg types
      ++input_arg_idx;
    }

    // handle return value / output
    // TODO: more metadata + handling
    // NOTE: inputs, builtins and uniforms are handled on the SPIRVLib side
    // above, outputs are however already handled on the LLVM side (VulkanFinal
    // pass) and thus have no SPIRVVariable mapping yet and have not been added
    // to the entry point i/o set yet
    // -> create SPIRVVariable for outputs + add them to the entry point i/o set
    // in here
    const std::string output_var_name_stub = func_name + ".vulkan_output.";
    uint32_t output_arg_idx = 0, output_location = 0;
    for (const auto &GV : M->globals()) {
      if (GV.hasName() && GV.getName().find(output_var_name_stub) == 0) {
        auto output_name = GV.getName().split(".vulkan_output.").second;

        assert(output_arg_idx < md_data_output.size() &&
               "invalid/incomplete output metadata");
        const auto md_prefix_split_pos =
            md_data_output[output_arg_idx].find(':');
        const std::string md_prefix =
            (md_prefix_split_pos != std::string::npos
                 ? md_data_output[output_arg_idx].substr(0, md_prefix_split_pos)
                 : "");
        const std::string md_info =
            (md_prefix_split_pos != std::string::npos
                 ? md_data_output[output_arg_idx].substr(md_prefix_split_pos +
                                                         1)
                 : md_data_output[output_arg_idx]);

        if (md_prefix != "") {
          // -> fbo color
          if (md_prefix == "stage" &&
              md_info.find("fbo_output:") != std::string::npos) {
            spirv_global_io_type global_type;
            global_type.is_write_only = true;
            global_type.is_fbo_color = true;
            emitShaderSPIRVGlobal(*F, BF, GV, output_name.str(), SPIRAS_Output,
                                  global_type, md_info);
          }
          // -> fbo depth
          else if (md_prefix == "stage" &&
                   md_info.find("fbo_depth:") != std::string::npos) {
            spirv_global_io_type global_type;
            global_type.is_write_only = true;
            global_type.is_fbo_depth = true;
            global_type.is_builtin = true;
            emitShaderSPIRVGlobal(*F, BF, GV, output_name.str(), SPIRAS_Output,
                                  global_type, md_info, spv::BuiltInFragDepth);
            // since we explicitly write depth, flag the function as
            // "DepthReplacing"
            BF->addExecutionMode(
                new SPIRVExecutionMode(BF, ExecutionModeDepthReplacing));
          }
          // -> builtin
          else if (md_prefix == "builtin" || md_prefix == "stage") {
            const auto builtin = get_builtin(md_info);
            if (!builtin.second) {
              errs() << "unknown builtin: " << md_info << "\n";
            }

            const auto is_valid = is_builtin_valid_in_stage(
                builtin.first, stage, false /* output */);
            if (is_valid) {
              spirv_global_io_type global_type;
              global_type.is_builtin = true;
              global_type.is_write_only = true;
              emitShaderSPIRVGlobal(*F, BF, GV, output_name.str(),
                                    SPIRAS_Output, global_type, md_info,
                                    builtin.first);
            } else {
              // TODO: should catch this earlier
              errs() << "output builtin \"" << md_info
                     << "\" can not be used in stage \""
                     << vulkan_stage_to_string(stage) << "\"\n";
            }
          }
          // -> unknown or unhandled yet
          else {
            assert(false && "unknown or unhandled output");
          }
        }
        // -> normal output
        else {
          spirv_global_io_type global_type;
          global_type.is_write_only = true;
          global_type.set_location = true;
          global_type.location = output_location++;
          emitShaderSPIRVGlobal(*F, BF, GV, output_name.str(), SPIRAS_Output,
                                global_type, md_info);
        }
        ++output_arg_idx;
      }
    }

    // add immutable samples to the interface
    if (!immutable_samplers.empty()) {
      for (auto &var : immutable_samplers) {
        BM->addEntryPointIO(BF->getId(), var);
      }
    }

    // Create all basic blocks before creating any instruction.
    for (Function::iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI) {
      transValue(&*FI, nullptr);
    }

    // set compute shader constant work-group size
    if (F->getCallingConv() == llvm::CallingConv::FLOOR_KERNEL) {
      if (MDNode *WGSize = F->getMetadata(kSPIR2MD::WGSize); WGSize) {
        // -> constant/required user-specified local/work-group size
        uint32_t constant_wg_size_vals[3]{1, 1, 1};
        decodeMDNode(WGSize, constant_wg_size_vals[0], constant_wg_size_vals[1],
                     constant_wg_size_vals[2]);
        BF->addExecutionMode(new SPIRVExecutionMode(
            BF, spv::ExecutionModeLocalSize, constant_wg_size_vals[0],
            constant_wg_size_vals[1], constant_wg_size_vals[2]));
      } else {
        // -> any local/work-group size (allow specialization)
        const auto global_name = func_name + ".vulkan_constant.local_size";
        auto gv_wg_size = M->getNamedGlobal(global_name);

        // NOTE: 128 is the minimum value that has to be supported for x
        uint32_t default_wg_size_vals[3]{128, 1, 1};

        auto uint_type = BM->addIntegerType(32, false);
        auto uint3_type = BM->addVectorType(uint_type, 3);
        std::vector<SPIRVValue *> wg_size_vals{
            BM->addSpecIntegerConstant(uint_type, default_wg_size_vals[0]),
            BM->addSpecIntegerConstant(uint_type, default_wg_size_vals[1]),
            BM->addSpecIntegerConstant(uint_type, default_wg_size_vals[2]),
        };
        auto wg_size = BM->addSpecCompositeConstant(uint3_type, wg_size_vals);
        BM->setName(wg_size, global_name);

        // set work-group size (x, y, z) spec ids to 1, 2 and 3
        // NOTE: we're starting this at 1 instead of 0, b/c of nvidia driver
        // bugs
        for (uint32_t i = 0; i < 3; ++i) {
          wg_size_vals[i]->addDecorate(spv::DecorationSpecId, i + 1);
        }
        BF->addExecutionMode(new SPIRVExecutionModeId(
            BF, spv::ExecutionModeLocalSizeId, wg_size_vals[0]->getId(),
            wg_size_vals[1]->getId(), wg_size_vals[2]->getId()));

        // preempt loads of the "<i32 x 3>*" work-group size constant
        // -> this has to be a constant composite in SPIR-V, not a variable
        // -> replace (map) all loads with the constant
        std::vector<User *> users;
        for (auto user : gv_wg_size->users()) {
          users.emplace_back(user);
        }

        if (!users.empty()) {
          // bitcast uint3 -> int3 for all users
          // NOTE/TODO: ideally, this should stay a uint3, but this would incur
          // type mismatch problems later on (would need to do int type
          // inference over the whole function to fix this properly)
          auto int_type = BM->addIntegerType(32, true);
          auto int3_type = BM->addVectorType(int_type, 3);
          auto entry_bb = (SPIRVBasicBlock *)transValue(&*F->begin(), nullptr);
          auto wg_size_int3 =
              BM->addUnaryInst(spv::OpBitcast, int3_type, wg_size, entry_bb);

          for (auto user : users) {
            if (const auto instr = dyn_cast<LoadInst>(user)) {
              mapValue((const Value *)instr, wg_size_int3);
            }
          }
        }
      }
    }
  } else if (SrcLang != SourceLanguageGLSL) {
    // Creating all basic blocks before creating any instruction.
    for (auto &FI : *F) {
      transValue(&FI, nullptr);
    }
  }

  // handle global constant variables, these need to be lowered to
  // function-scope (duplicate per function)
  // SPIR-V 1.4+: also handle local buffers (add to interface)
  // NOTE: needs to be done after basic blocks have been created (to add
  // OpVariables), but before being used when adding the instructions
  std::unordered_set<GlobalVariable *> added_globals;
  for (auto &GV : M->globals()) {
    // don't want to handle the globals that we added in here
    if (added_globals.count(&GV) > 0)
      continue;

    // ignore globals with unknown/unhandled linkage
    if (GV.getLinkage() == GlobalValue::AvailableExternallyLinkage ||
        GV.getLinkage() == GlobalValue::PrivateLinkage)
      continue;

    const auto gv_as = GV.getType()->getAddressSpace();
    if (SPIRSPIRVAddrSpaceMap::map(static_cast<SPIRAddressSpace>(gv_as)) ==
        spv::StorageClassFunction) {
      bool is_used_in_function = false;
      libfloor_utils::for_all_instruction_users(
          GV, [&is_used_in_function, &F](Instruction &instr) {
            if (instr.getParent()->getParent() == F) {
              is_used_in_function = true;
            }
          });
      if (!is_used_in_function) {
        continue;
      }

      // duplicate the global + replace all uses of it in this function with the
      // duplicate
      auto dup = new GlobalVariable(
          *M, GV.getType()->getPointerElementType(), GV.isConstant(),
          GlobalVariable::ExternalWeakLinkage,
          (GV.hasInitializer() ? GV.getInitializer() : nullptr),
          GV.getName() + "." + F->getName(), nullptr,
          GlobalValue::NotThreadLocal, GV.getType()->getAddressSpace());
      added_globals.emplace(dup);

      // need to copy all uses beforehand due iter invalidation
      std::vector<Use *> uses;
      uses.reserve(GV.getNumUses());
      for (auto &use : GV.uses()) {
        uses.emplace_back(&use);
      }

      for (auto &use : uses) {
        if (const auto instr = dyn_cast<Instruction>(use->getUser())) {
          if (instr->getParent()->getParent() == F) {
            use->set(dup);
          }
        }
      }

      // translate value/duplicate at the beginning of the entry BB of this
      // function
      auto BB = (SPIRVBasicBlock *)transValue(&F->getEntryBlock(), nullptr);
      transValue(dup, BB);
    } else if (SPIRSPIRVAddrSpaceMap::map(static_cast<SPIRAddressSpace>(
                   gv_as)) == spv::StorageClassWorkgroup) {
      BM->addEntryPointIO(BF->getId(),
                          (SPIRVVariable *)transValue(&GV, nullptr));
    }
  }

  // create all instructions
  for (auto &FI : *F) {
    SPIRVBasicBlock *BB =
        static_cast<SPIRVBasicBlock *>(transValue(&FI, nullptr));
    for (auto &BI : FI) {
      transValue(&BI, BB, false);
    }
  }
  // Enable FP contraction unless proven otherwise
  joinFPContract(F, FPContract::ENABLED);
  fpContractUpdateRecursive(F, getFPContract(F));

  if (isEntryPoint(F) && SrcLang != SourceLanguageGLSL /* already handled */) {
    collectInputOutputVariables(BF, F);
  }
}

bool LLVMToSPIRVBase::transVulkanVersion() {
  SrcLang = std::get<0>(getSPIRVSource(M));
  if (SrcLang != SourceLanguageGLSL) {
    return true;
  }

  const llvm::NamedMDNode *VulkanVersion =
      M->getNamedMetadata("vulkan.version");
  if (VulkanVersion == nullptr || VulkanVersion->getNumOperands() != 1) {
    return false;
  }

  const MDNode *vulkan_version_md = VulkanVersion->getOperand(0);
  if (vulkan_version_md->getNumOperands() < 2) {
    return false;
  }

  uint64_t version_major = 0, version_minor = 0;

  const MDOperand &version_major_op = vulkan_version_md->getOperand(0);
  if (const ConstantAsMetadata *version_major_md =
          dyn_cast_or_null<ConstantAsMetadata>(version_major_op.get())) {
    if (const ConstantInt *version_major_int =
            dyn_cast_or_null<ConstantInt>(version_major_md->getValue())) {
      version_major = version_major_int->getZExtValue();
    } else {
      return false;
    }
  } else {
    return false;
  }

  const MDOperand &version_minor_op = vulkan_version_md->getOperand(1);
  if (const ConstantAsMetadata *version_minor_md =
          dyn_cast_or_null<ConstantAsMetadata>(version_minor_op.get())) {
    if (const ConstantInt *version_minor_int =
            dyn_cast_or_null<ConstantInt>(version_minor_md->getValue())) {
      version_minor = version_minor_int->getZExtValue();
    } else {
      return false;
    }
  } else {
    return false;
  }

  switch (version_major) {
  case 1:
    switch (version_minor) {
    case 3:
      BM->setSPIRVVersion(static_cast<uint32_t>(VersionNumber::SPIRV_1_6));
      break;
    default:
      return false;
    }
    break;
  default:
    return false;
  }

  return true;
}

bool isEmptyLLVMModule(Module *M) {
  return M->empty() &&      // No functions
         M->global_empty(); // No global variables
}

bool LLVMToSPIRVBase::translate() {
  BM->setGeneratorVer(KTranslatorVer);

  if (isEmptyLLVMModule(M))
    BM->addCapability(CapabilityLinkage);

  // Transform SPV-IR builtin calls to builtin variables.
  if (!transWorkItemBuiltinCallsToVariables())
    return false;

  if (!transVulkanVersion())
    return false;
  if (!transSourceLanguage())
    return false;
  if (!transExtension())
    return false;
  if (!transBuiltinSet())
    return false;
  if (!transAddressingMode())
    return false;
  if (!transGlobalVariables())
    return false;

  for (auto &F : *M) {
    auto FT = F.getFunctionType();
    std::map<unsigned, Type *> ChangedType;
    oclGetMutatedArgumentTypesByBuiltin(FT, ChangedType, &F);
    mutateFuncArgType(ChangedType, &F);
  }

  // SPIR-V logical layout requires all function declarations go before
  // function definitions.
  std::vector<Function *> Decls, Defs;
  for (auto &F : *M) {
    if (isBuiltinTransToInst(&F) || isBuiltinTransToExtInst(&F) ||
        F.getName().startswith(SPCV_CAST) ||
        F.getName().startswith(LLVM_MEMCPY) ||
        F.getName().startswith(SAMPLER_INIT))
      continue;
    if (F.isDeclaration())
      Decls.push_back(&F);
    else
      Defs.push_back(&F);
  }
  for (auto I : Decls)
    transFunctionDecl(I);
  for (auto I : Defs)
    transFunction(I);

  if (!transMetadata())
    return false;
  if (!transExecutionMode())
    return false;

#if 0 // for testing purposes: mark everything as NonUniform
	for (auto& func : BM->getFunctions()) {
		for (auto& BB : func->getBasicBlocks()) {
			for (auto& instr : BB->getInstructions()) {
				if (instr->hasType() && !instr->getType()->isTypeVoid()) {
					instr->addDecorate(DecorationNonUniform);
				}
			}
		}
	}
	for (auto& var : BM->getVariables()) {
		if (var->hasType() && !var->getType()->isTypeVoid()) {
			var->addDecorate(DecorationNonUniform);
		}
	}
#endif

  BM->resolveUnknownStructFields();
  DbgTran->transDebugMetadata();
  return true;
}

llvm::IntegerType *LLVMToSPIRVBase::getSizetType(unsigned AS) {
  return IntegerType::getIntNTy(M->getContext(),
                                M->getDataLayout().getPointerSizeInBits(AS));
}

void LLVMToSPIRVBase::oclGetMutatedArgumentTypesByBuiltin(
    llvm::FunctionType *FT, std::map<unsigned, Type *> &ChangedType,
    Function *F) {
  StringRef Demangled;
  if (!oclIsBuiltin(F->getName(), Demangled))
    return;
  if (Demangled.find(kSPIRVName::SampledImage) == std::string::npos)
    return;
  if (FT->getParamType(1)->isIntegerTy())
    ChangedType[1] = getSamplerType(F->getParent());
}

SPIRVValue *LLVMToSPIRVBase::transBuiltinToConstant(StringRef DemangledName,
                                                    CallInst *CI) {
  Op OC = getSPIRVFuncOC(DemangledName);
  if (!isSpecConstantOpCode(OC))
    return nullptr;
  if (OC == spv::OpSpecConstantComposite) {
    return BM->addSpecConstantComposite(transType(CI->getType()),
                                        transValue(getArguments(CI), nullptr));
  }
  Value *V = CI->getArgOperand(1);
  Type *Ty = CI->getType();
  assert(((Ty == V->getType()) ||
          // If bool is stored into memory, then clang will emit it as i8,
          // however for other usages of bool (like return type of a function),
          // it is emitted as i1.
          // Therefore, situation when we encounter
          // i1 _Z20__spirv_SpecConstant(i32, i8) is valid
          (Ty->isIntegerTy(1) && V->getType()->isIntegerTy(8))) &&
         "Type mismatch!");
  uint64_t Val = 0;
  if (Ty->isIntegerTy())
    Val = cast<ConstantInt>(V)->getZExtValue();
  else if (Ty->isFloatingPointTy())
    Val = cast<ConstantFP>(V)->getValueAPF().bitcastToAPInt().getZExtValue();
  else
    return nullptr;
  SPIRVValue *SC = BM->addSpecConstant(transType(Ty), Val);
  return SC;
}

SPIRVInstruction *LLVMToSPIRVBase::transBuiltinToInst(StringRef DemangledName,
                                                      CallInst *CI,
                                                      SPIRVBasicBlock *BB) {
  SmallVector<std::string, 2> Dec;
  Op OC = OpNop;
  SPIRVInstruction *Inst = nullptr;

  // special handling for Vulkan/SPIR-V image read/write
  if (SrcLang == spv::SourceLanguageGLSL &&
      (DemangledName.find(kOCLBuiltinName::ReadImage) == 0 ||
       DemangledName.find(kOCLBuiltinName::WriteImage) == 0 ||
       DemangledName.find(std::string(kSPIRVName::Prefix) +
                          kSPIRVName::ImageQuerySize) ==
           0 /* matches both LOD and non-LOD */)) {
    auto inst_op = transVulkanImageFunction(CI, BB, DemangledName.str());
    assert(inst_op.first != nullptr && inst_op.second != OpNop &&
           "failed to translate image read/write function");
    Inst = inst_op.first;
    OC = inst_op.second;
  } else {
    OC = getSPIRVFuncOC(DemangledName, &Dec);

    if (OC == OpNop)
      return nullptr;

    if (OpReadPipeBlockingINTEL <= OC && OC <= OpWritePipeBlockingINTEL &&
        !BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_blocking_pipes))
      return nullptr;

    if (OpFixedSqrtINTEL <= OC && OC <= OpFixedExpINTEL)
      BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_arbitrary_precision_fixed_point),
          SPIRVEC_InvalidInstruction,
          CI->getCalledOperand()->getName().str() +
              "\nFixed point instructions can't be translated correctly "
              "without "
              "enabled SPV_INTEL_arbitrary_precision_fixed_point extension!\n");

    if ((OpArbitraryFloatSinCosPiINTEL <= OC &&
         OC <= OpArbitraryFloatCastToIntINTEL) ||
        (OpArbitraryFloatAddINTEL <= OC && OC <= OpArbitraryFloatPowNINTEL))
      BM->getErrorLog().checkError(
          BM->isAllowedToUseExtension(
              ExtensionID::SPV_INTEL_arbitrary_precision_floating_point),
          SPIRVEC_InvalidInstruction,
          CI->getCalledOperand()->getName().str() +
              "\nFloating point instructions can't be translated correctly "
              "without enabled SPV_INTEL_arbitrary_precision_floating_point "
              "extension!\n");

    Inst = transBuiltinToInstWithoutDecoration(OC, CI, BB);
  }

  addDecorations(Inst, Dec);
  return Inst;
}

bool LLVMToSPIRVBase::transExecutionMode() {
  if (auto NMD = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::ExecutionMode)) {
    while (!NMD.atEnd()) {
      unsigned EMode = ~0U;
      Function *F = nullptr;
      auto N = NMD.nextOp(); /* execution mode MDNode */
      N.get(F).get(EMode);

      SPIRVFunction *BF = static_cast<SPIRVFunction *>(getTranslatedValue(F));
      assert(BF && "Invalid kernel function");
      if (!BF)
        return false;

      auto AddSingleArgExecutionMode = [&](ExecutionMode EMode) {
        uint32_t Arg;
        N.get(Arg);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(BF, EMode, Arg)));
      };

      switch (EMode) {
      case spv::ExecutionModeContractionOff:
        if (SrcLang != spv::SourceLanguageGLSL) {
          BF->addExecutionMode(BM->add(
              new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
        }
        break;
      case spv::ExecutionModeInitializer:
      case spv::ExecutionModeFinalizer:
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1)) {
          BF->addExecutionMode(BM->add(
              new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
        } else {
          getErrorLog().checkError(false, SPIRVEC_Requires1_1,
                                   "Initializer/Finalizer Execution Mode");
          return false;
        }
        break;
      case spv::ExecutionModeLocalSize:
      case spv::ExecutionModeLocalSizeHint: {
        unsigned X, Y, Z;
        N.get(X).get(Y).get(Z);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            BF, static_cast<ExecutionMode>(EMode), X, Y, Z)));
      } break;
      case spv::ExecutionModeMaxWorkgroupSizeINTEL: {
        if (BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes)) {
          unsigned X, Y, Z;
          N.get(X).get(Y).get(Z);
          BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
              BF, static_cast<ExecutionMode>(EMode), X, Y, Z)));
          BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
          BM->addCapability(CapabilityKernelAttributesINTEL);
        }
      } break;
      case spv::ExecutionModeNoGlobalOffsetINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes))
          break;
        BF->addExecutionMode(BM->add(
            new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
        BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
        BM->addCapability(CapabilityKernelAttributesINTEL);
      } break;
      case spv::ExecutionModeVecTypeHint:
      case spv::ExecutionModeSubgroupSize:
      case spv::ExecutionModeSubgroupsPerWorkgroup:
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        break;
      case spv::ExecutionModeNumSIMDWorkitemsINTEL:
      case spv::ExecutionModeSchedulerTargetFmaxMhzINTEL:
      case spv::ExecutionModeMaxWorkDimINTEL:
      case spv::internal::ExecutionModeStreamingInterfaceINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_kernel_attributes))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        BM->addExtension(ExtensionID::SPV_INTEL_kernel_attributes);
        BM->addCapability(CapabilityFPGAKernelAttributesINTEL);
      } break;
      case spv::ExecutionModeSharedLocalMemorySizeINTEL: {
        if (!BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
      } break;

      case spv::ExecutionModeDenormPreserve:
      case spv::ExecutionModeDenormFlushToZero:
      case spv::ExecutionModeSignedZeroInfNanPreserve:
      case spv::ExecutionModeRoundingModeRTE:
      case spv::ExecutionModeRoundingModeRTZ: {
        if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_4)) {
          BM->setMinSPIRVVersion(
              static_cast<SPIRVWord>(VersionNumber::SPIRV_1_4));
          AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        } else if (BM->isAllowedToUseExtension(
                       ExtensionID::SPV_KHR_float_controls)) {
          BM->addExtension(ExtensionID::SPV_KHR_float_controls);
          AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
        }
      } break;
      case spv::ExecutionModeRoundingModeRTPINTEL:
      case spv::ExecutionModeRoundingModeRTNINTEL:
      case spv::ExecutionModeFloatingPointModeALTINTEL:
      case spv::ExecutionModeFloatingPointModeIEEEINTEL: {
        if (!BM->isAllowedToUseExtension(
                ExtensionID::SPV_INTEL_float_controls2))
          break;
        AddSingleArgExecutionMode(static_cast<ExecutionMode>(EMode));
      } break;
      case spv::internal::ExecutionModeFastCompositeKernelINTEL: {
        if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fast_composite))
          BF->addExecutionMode(BM->add(
              new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
      } break;
      default:
        llvm_unreachable("invalid execution mode");
      }
    }
  }

  transFPContract();

  return true;
}

void LLVMToSPIRVBase::transFPContract() {
  FPContractMode Mode = BM->getFPContractMode();

  for (Function &F : *M) {
    SPIRVValue *TranslatedF = getTranslatedValue(&F);
    if (!TranslatedF) {
      continue;
    }
    SPIRVFunction *BF = static_cast<SPIRVFunction *>(TranslatedF);

    bool IsEntryPoint =
        BF->getModule()->isEntryPoint(spv::ExecutionModelKernel, BF->getId());
    if (!IsEntryPoint)
      continue;

    FPContract FPC = getFPContract(&F);
    assert(FPC != FPContract::UNDEF);

    bool DisableContraction = false;
    switch (Mode) {
    case FPContractMode::Fast:
      DisableContraction = false;
      break;
    case FPContractMode::On:
      DisableContraction = FPC == FPContract::DISABLED;
      break;
    case FPContractMode::Off:
      DisableContraction = true;
      break;
    }

    if (DisableContraction && SrcLang != spv::SourceLanguageGLSL) {
      BF->addExecutionMode(BF->getModule()->add(
          new SPIRVExecutionMode(BF, spv::ExecutionModeContractionOff)));
    }
  }
}

bool LLVMToSPIRVBase::transMetadata() {
  if (!transOCLMetadata())
    return false;

  auto Model = getMemoryModel(*M);
  if (Model != SPIRVMemoryModelKind::MemoryModelMax)
    BM->setMemoryModel(static_cast<SPIRVMemoryModelKind>(Model));

  return true;
}

// Work around to translate kernel_arg_type and kernel_arg_type_qual metadata
static void transKernelArgTypeMD(SPIRVModule *BM, Function *F, MDNode *MD,
                                 std::string MDName) {
  std::string KernelArgTypesMDStr =
      std::string(MDName) + "." + F->getName().str() + ".";
  for (const auto &TyOp : MD->operands())
    KernelArgTypesMDStr += cast<MDString>(TyOp)->getString().str() + ",";
  BM->getString(KernelArgTypesMDStr);
}

bool LLVMToSPIRVBase::transOCLMetadata() {
  // TODO: do shaders need to be handled in here?
  if (SrcLang == spv::SourceLanguageGLSL) {
    return true;
  }

  for (auto &F : *M) {
    if (F.getCallingConv() != CallingConv::FLOOR_KERNEL)
      continue;

    SPIRVFunction *BF = static_cast<SPIRVFunction *>(getTranslatedValue(&F));
    assert(BF && "Kernel function should be translated first");

    // Create 'OpString' as a workaround to store information about
    // *orignal* (typedef'ed, unsigned integers) type names of kernel arguments.
    // OpString "kernel_arg_type.%kernel_name%.typename0,typename1,..."
    if (auto *KernelArgType = F.getMetadata(SPIR_MD_KERNEL_ARG_TYPE))
      if (BM->shouldPreserveOCLKernelArgTypeMetadataThroughString())
        transKernelArgTypeMD(BM, &F, KernelArgType, SPIR_MD_KERNEL_ARG_TYPE);

    if (auto *KernelArgTypeQual = F.getMetadata(SPIR_MD_KERNEL_ARG_TYPE_QUAL)) {
      foreachKernelArgMD(
          KernelArgTypeQual, BF,
          [](const std::string &Str, SPIRVFunctionParameter *BA) {
            if (Str.find("volatile") != std::string::npos)
              BA->addDecorate(new SPIRVDecorate(DecorationVolatile, BA));
            if (Str.find("restrict") != std::string::npos)
              BA->addDecorate(
                  new SPIRVDecorate(DecorationFuncParamAttr, BA,
                                    FunctionParameterAttributeNoAlias));
          });
      // Create 'OpString' as a workaround to store information about
      // constant qualifiers of pointer kernel arguments. Store empty string
      // for a non constant parameter.
      // OpString "kernel_arg_type_qual.%kernel_name%.qual0,qual1,..."
      if (BM->shouldPreserveOCLKernelArgTypeMetadataThroughString())
        transKernelArgTypeMD(BM, &F, KernelArgTypeQual,
                             SPIR_MD_KERNEL_ARG_TYPE_QUAL);
    }
    if (auto *KernelArgName = F.getMetadata(SPIR_MD_KERNEL_ARG_NAME)) {
      foreachKernelArgMD(
          KernelArgName, BF,
          [=](const std::string &Str, SPIRVFunctionParameter *BA) {
            BM->setName(BA, Str);
          });
    }
  }
  return true;
}

bool LLVMToSPIRVBase::transSourceLanguage() {
  auto Src = getSPIRVSource(M);
  SrcLang = std::get<0>(Src);
  SrcLangVer = std::get<1>(Src);
  if (SrcLang == SourceLanguageGLSL) {
    // "GLSL" is compiled as OpenCL 2.0 -> switch out the version number
    SrcLangVer = 450;
  }
  BM->setSourceLanguage(static_cast<SourceLanguage>(SrcLang), SrcLangVer);
  return true;
}

bool LLVMToSPIRVBase::transExtension() {
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::Extension)) {
    while (!N.atEnd()) {
      std::string S;
      N.nextOp().get(S);
      assert(!S.empty() && "Invalid extension");
      BM->getExtension().insert(S);
    }
  }
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::SourceExtension)) {
    while (!N.atEnd()) {
      std::string S;
      N.nextOp().get(S);
      assert(!S.empty() && "Invalid extension");
      BM->getSourceExtension().insert(S);
    }
  }
  for (auto &I :
       map<SPIRVCapabilityKind>(rmap<OclExt::Kind>(BM->getExtension())))
    BM->addCapability(I);

  return true;
}

void LLVMToSPIRVBase::dumpUsers(Value *V) {
  SPIRVDBG(dbgs() << "Users of " << *V << " :\n");
  for (auto UI = V->user_begin(), UE = V->user_end(); UI != UE; ++UI)
    SPIRVDBG(dbgs() << "  " << **UI << '\n');
}

Op LLVMToSPIRVBase::transBoolOpCode(SPIRVValue *Opn, Op OC) {
  if (!Opn->getType()->isTypeVectorOrScalarBool())
    return OC;
  IntBoolOpMap::find(OC, &OC);
  return OC;
}

std::pair<SPIRVInstruction *, Op>
LLVMToSPIRVBase::transVulkanImageFunction(CallInst *CI, SPIRVBasicBlock *BB,
                                          const std::string &DemangledName) {
  // NOTE: argument validity checking has already been done in VulkanImage
  //
  // read(image, sampler_idx, coord_with_layer,
  //      lod_type, [lod_arg_0], [lod_arg_1],
  //      bool is_offset, [offset],
  //      [sample_idx],
  //      [compare_val])
  //
  // write(image, coord_with_layer, data,
  //       // NOTE: only explicit lod or no lod
  //       lod_type, [lod_arg_0])
  //
  // query(image, lod)
  //

  // TODO: try to "cache" these things (image loads and sampler stuff)

  auto args = getArguments(CI);
  auto img_arg = args[0];
  auto spirv_img = getSPIRVValue(img_arg);
  assert(spirv_img != nullptr && "invalid image");

  SPIRVValue *loaded_img = nullptr;
  if (!isa<CallInst>(img_arg)) {
    // load the image (image function argument)
    loaded_img = BM->addLoadInst(spirv_img, {}, BB);
  } else {
    // special call, image has already been loaded (e.g. image array load)
    loaded_img = spirv_img;
  }
  auto spirv_img_type = (SPIRVTypeImage *)loaded_img->getType();

  std::vector<SPIRVWord> image_operands;
  uint32_t operands_mask = spv::ImageOperandsMaskNone;

  if (DemangledName.find("read_image") == 0) {
    // retrieve the sampler idx + load the sampler
    auto sampler_idx_arg = dyn_cast<ConstantInt>(args[1]);
    assert(sampler_idx_arg != nullptr && "sampler must be a constant int");
    const vulkan_sampling::sampler sampler_val{
        (uint32_t)sampler_idx_arg->getZExtValue()};
    const bool is_fetch =
        ((sampler_val.value &
          vulkan_sampling::sampler::COORD_MODE::__COORD_MODE_MASK) ==
         vulkan_sampling::sampler::COORD_MODE::PIXEL);

    // only load the sampler + create the sampled image if necessary
    SPIRVValue *loaded_sampler = nullptr;
    SPIRVValue *img = loaded_img;
    if (!is_fetch) {
      const bool has_vulkan_descriptor_buffer =
          (M->getNamedMetadata("floor.vulkan_descriptor_buffer") != nullptr);
      if (has_vulkan_descriptor_buffer) {
        // -> for descriptor buffer use
        loaded_sampler =
            BM->addLoadInst(immutable_samplers[sampler_val.value], {}, BB);
      } else {
        // -> legacy
        std::vector<SPIRVValue *> indices{
            BM->getLiteralAsConstant(sampler_val.value, false)};

        auto sampler_ptr = BM->addAccessChainInst(
            BM->addPointerType(spv::StorageClassUniformConstant,
                               BM->addSamplerType()),
            immutable_samplers[0], indices, BB, true);
        loaded_sampler = BM->addLoadInst(sampler_ptr, {}, BB);
      }

      // create the sampled image
      std::vector<SPIRVWord> sampled_img_ops{
          loaded_img->getId(),
          loaded_sampler->getId(),
      };
      img = BM->addInstTemplate(OpSampledImage, sampled_img_ops, BB,
                                BM->addSampledImageType(spirv_img_type));
    }

    std::vector<SPIRVWord> read_operands{img->getId()};
    spv::Op read_opcode = spv::OpNop;

    // coords
    auto coords_arg = transValue(args[2], BB);
    read_operands.emplace_back(coords_arg->getId());

    // lod type and args
    auto lod_type_arg = dyn_cast<ConstantInt>(args[3]);
    uint32_t arg_idx = 4; // from here on: arg count and indices are variable
    assert(lod_type_arg != nullptr && "lod type must be a constant int");
    auto lod_type = (vulkan_sampling::LOD_TYPE)lod_type_arg->getZExtValue();
    assert(lod_type <= vulkan_sampling::LOD_TYPE::__MAX_LOD_TYPE &&
           "invalid lod type");

    const bool is_fragment_shader =
        (CI->getParent()->getParent()->getCallingConv() ==
         CallingConv::FLOOR_FRAGMENT);
    if (!is_fragment_shader &&
        (lod_type == vulkan_sampling::LOD_TYPE::IMPLICIT_LOD ||
         lod_type == vulkan_sampling::LOD_TYPE::IMPLICIT_LOD_WITH_BIAS)) {
      // implicit LOD is only allowed in fragment shaders -> fix it
      auto explicit_lod_arg =
          ConstantInt::get(IntegerType::get(M->getContext(), 32), 0);
      if (lod_type == vulkan_sampling::LOD_TYPE::IMPLICIT_LOD_WITH_BIAS) {
        // replace bias arg with dummy 0
        args[arg_idx] = explicit_lod_arg;
      } else {
        // insert dummy 0
        args.insert(args.begin() + arg_idx, explicit_lod_arg);
      }
      lod_type = vulkan_sampling::LOD_TYPE::EXPLICIT_LOD;
    }

    switch (lod_type) {
    case vulkan_sampling::LOD_TYPE::NO_LOD:
      read_opcode = spv::OpImageFetch;
      break;
    case vulkan_sampling::LOD_TYPE::IMPLICIT_LOD:
      read_opcode = spv::OpImageSampleImplicitLod;
      break;
    case vulkan_sampling::LOD_TYPE::IMPLICIT_LOD_WITH_BIAS:
      read_opcode = spv::OpImageSampleImplicitLod;
      operands_mask |= spv::ImageOperandsBiasMask;
      image_operands.emplace_back(transValue(args[arg_idx++], BB)->getId());
      break;
    case vulkan_sampling::LOD_TYPE::EXPLICIT_LOD:
      read_opcode =
          (!is_fetch ? spv::OpImageSampleExplicitLod : spv::OpImageFetch);
      operands_mask |= spv::ImageOperandsLodMask;
      image_operands.emplace_back(transValue(args[arg_idx++], BB)->getId());
      break;
    case vulkan_sampling::LOD_TYPE::GRADIENT:
      read_opcode =
          (!is_fetch ? spv::OpImageSampleExplicitLod : spv::OpImageFetch);
      operands_mask |= spv::ImageOperandsGradMask;
      image_operands.emplace_back(transValue(args[arg_idx++], BB)->getId());
      image_operands.emplace_back(transValue(args[arg_idx++], BB)->getId());
      break;
    default:
      llvm_unreachable("invalid lod type");
    }

    // offset
    auto is_offset_arg = dyn_cast<ConstantInt>(args[arg_idx++]);
    assert(is_offset_arg != nullptr && "is_offset flag must be a constant int");
    if (!is_offset_arg->isZero()) {
      auto offset_arg = args[arg_idx++];
      assert(isa<Constant>(offset_arg) && "offset must be constant");
      operands_mask |= spv::ImageOperandsConstOffsetMask;
      image_operands.emplace_back(transValue(offset_arg, BB)->getId());
    }

    // sample idx
    if (DemangledName.find("msaa") != std::string::npos) {
      auto sample_idx_arg = args[arg_idx++];
      operands_mask |= spv::ImageOperandsSampleMask;
      image_operands.emplace_back(transValue(sample_idx_arg, BB)->getId());
    }

    // depth compare
    bool is_depth_compare = false;
    if (arg_idx < args.size()) {
      is_depth_compare = true;
      auto compare_arg = args[arg_idx++];

      // must switch out the opcode
      assert((read_opcode == spv::OpImageSampleImplicitLod ||
              read_opcode == spv::OpImageSampleExplicitLod) &&
             "invalid read opcode");
      read_opcode = (read_opcode == spv::OpImageSampleImplicitLod
                         ? spv::OpImageSampleDrefImplicitLod
                         : spv::OpImageSampleDrefExplicitLod);
      read_operands.emplace_back(transValue(compare_arg, BB)->getId());
    }

    // sanity check
    assert(arg_idx == args.size() && "unhandled args");

    // create the image read
    if (operands_mask != spv::ImageOperandsMaskNone) {
      // must only be emitted if mask != None + operands are ordered
      read_operands.emplace_back(operands_mask);
      for (const auto &id : image_operands) {
        read_operands.emplace_back(id);
      }
    }

    SPIRVType *scalar_ret_type = nullptr;
    bool use_relaxed_precision = false;
    if (DemangledName.find("read_imageui") == 0) {
      scalar_ret_type = BM->addIntegerType(32, false);
    } else if (DemangledName.find("read_imagei") == 0) {
      scalar_ret_type = BM->addIntegerType(32, true);
    } else if (DemangledName.find("read_imagef") == 0) {
      scalar_ret_type = BM->addFloatType(32);
    } else if (DemangledName.find("read_imageh") == 0) {
      // Vulkan can't handle half precision sampling
      // -> use 32-bit float instead + flag everything as "RelaxedPrecision"
      scalar_ret_type = BM->addFloatType(32);
      use_relaxed_precision = true;
    } else {
      assert(false && "invalid image read function");
    }

    SPIRVType *ret_type = scalar_ret_type; // only depth compare is scalar
    if (!is_depth_compare) {
      ret_type = BM->addVectorType(scalar_ret_type, 4);
    }

    auto read_sample =
        BM->addInstTemplate(read_opcode, read_operands, BB, ret_type);
    if (use_relaxed_precision) {
      // spec says we should decorate both the instruction and the image
      read_sample->addDecorate(DecorationRelaxedPrecision);
      img->addDecorate(DecorationRelaxedPrecision);
    }
    return {read_sample, read_sample->getOpCode()};
  } else if (DemangledName.find("write_image") == 0) {
    std::vector<SPIRVWord> write_operands{loaded_img->getId()};

    // coords
    auto coords_arg = transValue(args[1], BB);
    write_operands.emplace_back(coords_arg->getId());

    // data
    // TODO: proper uint/int data type?
    auto data_arg = transValue(args[2], BB);
    write_operands.emplace_back(data_arg->getId());

    // lod type and args
    auto lod_type_arg = dyn_cast<ConstantInt>(args[3]);
    uint32_t arg_idx = 4; // from here on: arg count and indices are variable
    assert(lod_type_arg != nullptr && "lod type must be a constant int");
    auto lod_type = (vulkan_sampling::LOD_TYPE)lod_type_arg->getZExtValue();
    assert(lod_type <= vulkan_sampling::LOD_TYPE::__MAX_LOD_TYPE &&
           "invalid lod type");
    switch (lod_type) {
    case vulkan_sampling::LOD_TYPE::NO_LOD:
      // nop
      break;
    case vulkan_sampling::LOD_TYPE::EXPLICIT_LOD:
      // NOTE: SPIR-V supports this, but Vulkan doesn't
      //       -> this is dealt with elsewhere
      // operands_mask |= spv::ImageOperandsLodMask;
      // image_operands.emplace_back(transValue(args[arg_idx++], BB)->getId());
      ++arg_idx;
      break;
    case vulkan_sampling::LOD_TYPE::IMPLICIT_LOD:
    case vulkan_sampling::LOD_TYPE::IMPLICIT_LOD_WITH_BIAS:
    case vulkan_sampling::LOD_TYPE::GRADIENT:
    default:
      llvm_unreachable("invalid lod type");
    }

    // sample idx
    if (DemangledName.find("msaa") != std::string::npos) {
      // TODO: !
      // auto sample_idx_arg = args[arg_idx++];
      // operands_mask |= spv::ImageOperandsSampleMask;
      // image_operands.emplace_back(transValue(sample_idx_arg, BB)->getId());
    }

    // sanity check
    assert(arg_idx == args.size() && "unhandled args");

    // create the image write
    if (operands_mask != spv::ImageOperandsMaskNone) {
      // must only be emitted if mask != None + operands are ordered
      write_operands.emplace_back(operands_mask);
      for (const auto &id : image_operands) {
        write_operands.emplace_back(id);
      }
    }
    auto write_sample =
        BM->addInstTemplate(spv::OpImageWrite, write_operands, BB, nullptr);

    // Vulkan can't handle half precision writing
    // -> flag everything as "RelaxedPrecision"
    if (DemangledName.find("write_imageh") != std::string::npos) {
      // we can only decorate the image here, since there is no return id
      loaded_img->addDecorate(DecorationRelaxedPrecision);
      assert(data_arg->getType()->isTypeVectorOrScalarFloat(32) &&
             "must be 32-bit float");
    }

    return {write_sample, write_sample->getOpCode()};
  } else if (DemangledName.find(kSPIRVName::ImageQuerySize) !=
             std::string::npos) {
    std::vector<SPIRVWord> query_operands{loaded_img->getId()};

    // query requires cap
    BM->addCapability(spv::CapabilityImageQuery);

    // buffer and MSAA images must use the non-LOD variant
    const auto non_lod_variant =
        (DemangledName.find("msaa") != std::string::npos ||
         DemangledName.find("buffer") != std::string::npos);

    auto query_op = spv::OpImageQuerySize;
    if (!non_lod_variant) {
      auto lod_arg = transValue(args[1], BB);
      query_operands.emplace_back(lod_arg->getId());
      query_op = spv::OpImageQuerySizeLod;
    }

    // the return type is already correct on the LLVM side, just translate it
    SPIRVType *ret_type = transType(CI->getType());

    auto img_query =
        BM->addInstTemplate(query_op, query_operands, BB, ret_type);
    return {img_query, img_query->getOpCode()};
  }

  return {nullptr, OpNop};
}

SPIRVInstruction *
LLVMToSPIRVBase::transBuiltinToInstWithoutDecoration(Op OC, CallInst *CI,
                                                     SPIRVBasicBlock *BB) {
  assert(!(SrcLang == spv::SourceLanguageGLSL && isImageOpCode(OC)) &&
         "should not be here");

  if (isGroupOpCode(OC))
    BM->addCapability(CapabilityGroups);
  switch (OC) {
  case OpControlBarrier: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addControlBarrierInst(BArgs[0], BArgs[1], BArgs[2], BB);
  } break;
  case OpGroupAsyncCopy: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addAsyncGroupCopy(BArgs[0], BArgs[1], BArgs[2], BArgs[3],
                                 BArgs[4], BArgs[5], BB);
  } break;
  case OpSampledImage: {
    // Clang can generate SPIRV-friendly call for OpSampledImage instruction,
    // i.e. __spirv_SampledImage... But it can't generate correct return type
    // for this call, because there is no support for type corresponding to
    // OpTypeSampledImage. So, in this case, we create the required type here.
    Value *Image = CI->getArgOperand(0);
    Type *ImageTy = Image->getType();
    if (isOCLImageType(ImageTy))
      ImageTy = getSPIRVImageTypeFromOCL(M, ImageTy);
    Type *SampledImgTy = getSPIRVTypeByChangeBaseTypeName(
        M, ImageTy, kSPIRVTypeName::Image, kSPIRVTypeName::SampledImg);
    Value *Sampler = CI->getArgOperand(1);
    return BM->addSampledImageInst(transType(SampledImgTy),
                                   transValue(Image, BB),
                                   transValue(Sampler, BB), BB);
  }
  case OpFixedSqrtINTEL:
  case OpFixedRecipINTEL:
  case OpFixedRsqrtINTEL:
  case OpFixedSinINTEL:
  case OpFixedCosINTEL:
  case OpFixedSinCosINTEL:
  case OpFixedSinPiINTEL:
  case OpFixedCosPiINTEL:
  case OpFixedSinCosPiINTEL:
  case OpFixedLogINTEL:
  case OpFixedExpINTEL: {
    // LLVM fixed point functions return value:
    // iN (arbitrary precision integer of N bits length)
    // Arguments:
    // A(iN), S(i1), I(i32), rI(i32), Quantization(i32), Overflow(i32)
    // where A - integer input of any width.

    // SPIR-V fixed point instruction contains:
    // <id>ResTy Res<id> In<id> \
    // Literal S Literal I Literal rI Literal Q Literal O

    Type *ResTy = CI->getType();

    auto OpItr = CI->value_op_begin();
    auto OpEnd = OpItr + CI->arg_size();

    // If the return type of an instruction is wider than 64-bit, then this
    // instruction will return via 'sret' argument added into the arguments
    // list. Here we reverse this, removing 'sret' argument and restoring
    // the original return type.
    if (CI->hasStructRetAttr()) {
      assert(ResTy->isVoidTy() && "Return type is not void");
      ResTy = cast<PointerType>(OpItr->getType())->getElementType();
      OpItr++;
    }

    // Input - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *Input = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy())
      Input = BM->addLoadInst(Input, {}, BB);
    OpItr++;

    std::vector<SPIRVWord> Literals;
    std::transform(OpItr, OpEnd, std::back_inserter(Literals), [](auto *O) {
      return cast<llvm::ConstantInt>(O)->getZExtValue();
    });

    auto *APIntInst =
        BM->addFixedPointIntelInst(OC, transType(ResTy), Input, Literals, BB);
    if (!CI->hasStructRetAttr())
      return APIntInst;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), APIntInst, {},
                            BB);
  }
  case OpArbitraryFloatCastINTEL:
  case OpArbitraryFloatCastFromIntINTEL:
  case OpArbitraryFloatCastToIntINTEL:
  case OpArbitraryFloatRecipINTEL:
  case OpArbitraryFloatRSqrtINTEL:
  case OpArbitraryFloatCbrtINTEL:
  case OpArbitraryFloatSqrtINTEL:
  case OpArbitraryFloatLogINTEL:
  case OpArbitraryFloatLog2INTEL:
  case OpArbitraryFloatLog10INTEL:
  case OpArbitraryFloatLog1pINTEL:
  case OpArbitraryFloatExpINTEL:
  case OpArbitraryFloatExp2INTEL:
  case OpArbitraryFloatExp10INTEL:
  case OpArbitraryFloatExpm1INTEL:
  case OpArbitraryFloatSinINTEL:
  case OpArbitraryFloatCosINTEL:
  case OpArbitraryFloatSinCosINTEL:
  case OpArbitraryFloatSinPiINTEL:
  case OpArbitraryFloatCosPiINTEL:
  case OpArbitraryFloatSinCosPiINTEL:
  case OpArbitraryFloatASinINTEL:
  case OpArbitraryFloatASinPiINTEL:
  case OpArbitraryFloatACosINTEL:
  case OpArbitraryFloatACosPiINTEL:
  case OpArbitraryFloatATanINTEL:
  case OpArbitraryFloatATanPiINTEL: {
    // Format of instruction CastFromInt:
    //   LLVM arbitrary floating point functions return value type:
    //       iN (arbitrary precision integer of N bits length)
    //   Arguments: A(iN), Mout(i32), FromSign(bool), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal Mout Literal FromSign
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of instruction CastToInt:
    //   LLVM arbitrary floating point functions return value: iN
    //   Arguments: A(iN), MA(i32), ToSign(bool), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA Literal ToSign
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of other instructions:
    //   LLVM arbitrary floating point functions return value: iN
    //   Arguments: A(iN), MA(i32), Mout(i32), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA Literal Mout
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    Type *ResTy = CI->getType();

    auto OpItr = CI->value_op_begin();
    auto OpEnd = OpItr + CI->arg_size();

    // If the return type of an instruction is wider than 64-bit, then this
    // instruction will return via 'sret' argument added into the arguments
    // list. Here we reverse this, removing 'sret' argument and restoring
    // the original return type.
    if (CI->hasStructRetAttr()) {
      assert(ResTy->isVoidTy() && "Return type is not void");
      ResTy = cast<PointerType>(OpItr->getType())->getElementType();
      OpItr++;
    }

    // Input - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *Input = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy())
      Input = BM->addLoadInst(Input, {}, BB);
    OpItr++;

    std::vector<SPIRVWord> Literals;
    std::transform(OpItr, OpEnd, std::back_inserter(Literals), [](auto *O) {
      return cast<llvm::ConstantInt>(O)->getZExtValue();
    });

    auto *APIntInst = BM->addArbFloatPointIntelInst(OC, transType(ResTy), Input,
                                                    nullptr, Literals, BB);
    if (!CI->hasStructRetAttr())
      return APIntInst;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), APIntInst, {},
                            BB);
  }
  case OpArbitraryFloatAddINTEL:
  case OpArbitraryFloatSubINTEL:
  case OpArbitraryFloatMulINTEL:
  case OpArbitraryFloatDivINTEL:
  case OpArbitraryFloatGTINTEL:
  case OpArbitraryFloatGEINTEL:
  case OpArbitraryFloatLTINTEL:
  case OpArbitraryFloatLEINTEL:
  case OpArbitraryFloatEQINTEL:
  case OpArbitraryFloatHypotINTEL:
  case OpArbitraryFloatATan2INTEL:
  case OpArbitraryFloatPowINTEL:
  case OpArbitraryFloatPowRINTEL:
  case OpArbitraryFloatPowNINTEL: {
    // Format of instructions Add, Sub, Mul, Div, Hypot, ATan2, Pow, PowR:
    //   LLVM arbitrary floating point functions return value:
    //       iN (arbitrary precision integer of N bits length)
    //   Arguments: A(iN), MA(i32), B(iN), MB(i32), Mout(i32),
    //              EnableSubnormals(i32), RoundingMode(i32),
    //              RoundingAccuracy(i32)
    //   where A, B and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal MB Literal Mout
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of instruction PowN:
    //   LLVM arbitrary floating point functions return value: iN
    //   Arguments: A(iN), MA(i32), B(iN), Mout(i32), EnableSubnormals(i32),
    //              RoundingMode(i32), RoundingAccuracy(i32)
    //   where A, B and return values are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal Mout
    //       Literal EnableSubnormals Literal RoundingMode
    //       Literal RoundingAccuracy

    // Format of instructions GT, GE, LT, LE, EQ:
    //   LLVM arbitrary floating point functions return value: Bool
    //   Arguments: A(iN), MA(i32), B(iN), MB(i32)
    //   where A and B are of arbitrary precision integer type.
    //   SPIR-V arbitrary floating point instruction layout:
    //   <id>ResTy Res<id> A<id> Literal MA B<id> Literal MB

    Type *ResTy = CI->getType();

    auto OpItr = CI->value_op_begin();
    auto OpEnd = OpItr + CI->arg_size();

    // If the return type of an instruction is wider than 64-bit, then this
    // instruction will return via 'sret' argument added into the arguments
    // list. Here we reverse this, removing 'sret' argument and restoring
    // the original return type.
    if (CI->hasStructRetAttr()) {
      assert(ResTy->isVoidTy() && "Return type is not void");
      ResTy = cast<PointerType>(OpItr->getType())->getElementType();
      OpItr++;
    }

    // InA - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *InA = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy())
      InA = BM->addLoadInst(InA, {}, BB);
    OpItr++;

    std::vector<SPIRVWord> Literals;
    Literals.push_back(cast<llvm::ConstantInt>(*OpItr++)->getZExtValue());

    // InB - integer input of any width or 'byval' pointer to this integer
    SPIRVValue *InB = transValue(*OpItr, BB);
    if (OpItr->getType()->isPointerTy()) {
      std::vector<SPIRVWord> Mem;
      InB = BM->addLoadInst(InB, Mem, BB);
    }
    OpItr++;

    std::transform(OpItr, OpEnd, std::back_inserter(Literals), [](auto *O) {
      return cast<llvm::ConstantInt>(O)->getZExtValue();
    });

    auto *APIntInst = BM->addArbFloatPointIntelInst(OC, transType(ResTy), InA,
                                                    InB, Literals, BB);
    if (!CI->hasStructRetAttr())
      return APIntInst;
    return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), APIntInst, {},
                            BB);
  }
  default: {
    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit) {
      return BM->addUnaryInst(OC, transType(CI->getType()),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else if (isCmpOpCode(OC) || isUnaryPredicateOpCode(OC)) {
      auto ResultTy = CI->getType();
      Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
      auto IsVector = ResultTy->isVectorTy();
      if (IsVector)
        BoolTy = FixedVectorType::get(
            BoolTy, cast<FixedVectorType>(ResultTy)->getNumElements());
      auto BBT = transType(BoolTy);
      SPIRVInstruction *Res;
      if (isCmpOpCode(OC)) {
        assert(CI && CI->arg_size() == 2 && "Invalid call inst");
        Res = BM->addCmpInst(OC, BBT, transValue(CI->getArgOperand(0), BB),
                             transValue(CI->getArgOperand(1), BB), BB);
      } else {
        assert(CI && CI->arg_size() == 1 && "Invalid call inst");
        Res =
            BM->addUnaryInst(OC, BBT, transValue(CI->getArgOperand(0), BB), BB);
      }
      // OpenCL C and OpenCL C++ built-ins may have different return type
      if (ResultTy == BoolTy)
        return Res;
      assert(IsVector || (!IsVector && ResultTy->isIntegerTy(32)));
      auto Zero = transValue(Constant::getNullValue(ResultTy), BB);
      auto One = transValue(
          IsVector ? Constant::getAllOnesValue(ResultTy) : getInt32(M, 1), BB);
      return BM->addSelectInst(Res, One, Zero, BB);
    } else if (isBinaryOpCode(OC)) {
      assert(CI && CI->arg_size() == 2 && "Invalid call inst");
      return BM->addBinaryInst(OC, transType(CI->getType()),
                               transValue(CI->getArgOperand(0), BB),
                               transValue(CI->getArgOperand(1), BB), BB);
    } else if (CI->arg_size() == 1 && !CI->getType()->isVoidTy() &&
               !hasExecScope(OC) && !isAtomicOpCode(OC)) {
      return BM->addUnaryInst(OC, transType(CI->getType()),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else {
      auto Args = getArguments(CI);
      SPIRVType *SPRetTy = nullptr;
      Type *RetTy = CI->getType();
      auto F = CI->getCalledFunction();
      if (!RetTy->isVoidTy()) {
        SPRetTy = transType(RetTy);
      } else if (Args.size() > 0 && F->arg_begin()->hasStructRetAttr()) {
        SPRetTy = transType(F->arg_begin()->getType()->getPointerElementType());
        Args.erase(Args.begin());
      }
      auto *SPI = SPIRVInstTemplateBase::create(OC);
      std::vector<SPIRVWord> SPArgs;
      for (size_t I = 0, E = Args.size(); I != E; ++I) {
        assert((!isFunctionPointerType(Args[I]->getType()) ||
                isa<Function>(Args[I])) &&
               "Invalid function pointer argument");
        SPArgs.push_back(SPI->isOperandLiteral(I)
                             ? cast<ConstantInt>(Args[I])->getZExtValue()
                             : transValue(Args[I], BB)->getId());
      }
      BM->addInstTemplate(SPI, SPArgs, BB, SPRetTy);
      if (!SPRetTy || !SPRetTy->isTypeStruct())
        return SPI;
      std::vector<SPIRVWord> Mem;
      SPIRVDBG(spvdbgs() << *SPI << '\n');
      return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), SPI, Mem,
                              BB);
    }
  }
  }
  return nullptr;
}

SPIRV::SPIRVLinkageTypeKind
LLVMToSPIRVBase::transLinkageType(const GlobalValue *GV) {
  if (GV->isDeclarationForLinker())
    return SPIRVLinkageTypeKind::LinkageTypeImport;
  if (GV->hasInternalLinkage() || GV->hasPrivateLinkage())
    return spv::internal::LinkageTypeInternal;
  if (GV->hasLinkOnceODRLinkage())
    if (BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_linkonce_odr))
      return SPIRVLinkageTypeKind::LinkageTypeLinkOnceODR;
  return SPIRVLinkageTypeKind::LinkageTypeExport;
}

LLVMToSPIRVBase::FPContract LLVMToSPIRVBase::getFPContract(Function *F) {
  auto It = FPContractMap.find(F);
  if (It == FPContractMap.end()) {
    return FPContract::UNDEF;
  }
  return It->second;
}

bool LLVMToSPIRVBase::joinFPContract(Function *F, FPContract C) {
  FPContract &Existing = FPContractMap[F];
  switch (Existing) {
  case FPContract::UNDEF:
    if (C != FPContract::UNDEF) {
      Existing = C;
      return true;
    }
    return false;
  case FPContract::ENABLED:
    if (C == FPContract::DISABLED) {
      Existing = C;
      return true;
    }
    return false;
  case FPContract::DISABLED:
    return false;
  }
  llvm_unreachable("Unhandled FPContract value.");
}

void LLVMToSPIRVBase::add_array_stride_decoration(SPIRVType *type,
                                                  const uint32_t stride) {
  if (base_array_strides.count(type) > 0) {
    return;
  }
  base_array_strides.emplace(type, stride);
  type->addDecorate(spv::DecorationArrayStride, stride);
}

} // namespace SPIRV

char LLVMToSPIRVLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(LLVMToSPIRVLegacy, "llvmtospv",
                      "Translate LLVM to SPIR-V", false, false)
INITIALIZE_PASS_DEPENDENCY(OCLTypeToSPIRVLegacy)
INITIALIZE_PASS_END(LLVMToSPIRVLegacy, "llvmtospv", "Translate LLVM to SPIR-V",
                    false, false)

ModulePass *llvm::createLLVMToSPIRVLegacy(SPIRVModule *SMod) {
  return new LLVMToSPIRVLegacy(SMod);
}

void addPassesForSPIRV(legacy::PassManager &PassMgr,
                       const SPIRV::TranslatorOpts &Opts) {
  if (Opts.isSPIRVMemToRegEnabled())
    PassMgr.add(createPromoteMemoryToRegisterPass());
  PassMgr.add(createPreprocessMetadataLegacy());
  PassMgr.add(createSPIRVLowerSPIRBlocksLegacy());
  PassMgr.add(createOCLTypeToSPIRVLegacy());
  PassMgr.add(createSPIRVLowerOCLBlocksLegacy());
  PassMgr.add(createOCLToSPIRVLegacy());
  PassMgr.add(createLLVMToSPIRVTransformations());
  PassMgr.add(createSPIRVRegularizeLLVMLegacy());
  PassMgr.add(createSPIRVLowerConstExprLegacy());
  PassMgr.add(createSPIRVLowerBoolLegacy());
  PassMgr.add(createSPIRVLowerMemmoveLegacy());
  PassMgr.add(createSPIRVLowerSaddWithOverflowLegacy());
  PassMgr.add(createSPIRVLowerBitCastToNonStandardTypeLegacy(Opts));
}

bool isValidLLVMModule(Module *M, SPIRVErrorLog &ErrorLog) {
  if (!M)
    return false;

  if (isEmptyLLVMModule(M))
    return true;

  Triple TT(M->getTargetTriple());
  if (!ErrorLog.checkError(isSupportedTriple(TT), SPIRVEC_InvalidTargetTriple,
                           "Actual target triple is " + M->getTargetTriple()))
    return false;

  return true;
}

bool llvm::writeSpirv(Module *M, spv_ostream &OS, std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
#if 0 // NOPE
  // To preserve old behavior of the translator, let's enable all extensions
  // by default in this API
  DefaultOpts.enableAllExtensions();
#endif
  return llvm::writeSpirv(M, DefaultOpts, OS, ErrMsg);
}

bool llvm::writeSpirv(Module *M, const SPIRV::TranslatorOpts &Opts,
                      spv_ostream &OS, std::string &ErrMsg) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule(Opts));
  if (!isValidLLVMModule(M, BM->getErrorLog()))
    return false;

  legacy::PassManager PassMgr;
  addPassesForSPIRV(PassMgr, Opts);
#if 0 // absolutely DO NOT do this
  // Run loop simplify pass in order to avoid duplicate OpLoopMerge
  // instruction. It can happen in case of continue operand in the loop.
  if (hasLoopMetadata(M))
    PassMgr.add(createLoopSimplifyPass());
#endif
  PassMgr.add(createLLVMToSPIRVLegacy(BM.get()));
  PassMgr.run(*M);

  if (BM->getError(ErrMsg) != SPIRVEC_Success)
    return false;
  OS << *BM;
  return true;
}

bool llvm::regularizeLlvmForSpirv(Module *M, std::string &ErrMsg) {
  SPIRV::TranslatorOpts DefaultOpts;
#if 0 // NOPE
  // To preserve old behavior of the translator, let's enable all extensions
  // by default in this API
  DefaultOpts.enableAllExtensions();
#endif
  return llvm::regularizeLlvmForSpirv(M, ErrMsg, DefaultOpts);
}

bool llvm::regularizeLlvmForSpirv(Module *M, std::string &ErrMsg,
                                  const SPIRV::TranslatorOpts &Opts) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule());
  if (!isValidLLVMModule(M, BM->getErrorLog()))
    return false;

  legacy::PassManager PassMgr;
  addPassesForSPIRV(PassMgr, Opts);
  PassMgr.run(*M);
  return true;
}
