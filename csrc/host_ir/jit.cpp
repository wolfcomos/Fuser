// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bfs.h>
#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <val_graph_visitor.h>

#include <instrumentation.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <chrono>
#include <queue>
#include <unordered_map>
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <ATen/ATen.h>
#include <c10/core/MemoryFormat.h> // for c10::optional
#include <host_ir/jit.h>
#include <host_ir/host_ir.h>
#include <functional>
#include <utility>

namespace nvfuser {

/*
input: input buffer, input buffer length, options, 
output: at::Tensor result
*/
using allocate_fn = std::function<void*(int64_t*, int64_t, void*)>;

/*
input: cache id
output: KernelArgumentHolder
*/ 
struct KernelArgumentHolderPair {
  void* args;      // KernelArgumentHolder* for inputs
  void* outputs;   // KernelArgumentHolder* for outputs
};

using launch_kernel_fn = std::function<KernelArgumentHolderPair(int64_t, at::Tensor**, at::Tensor**)>;

// PIMPL implementation for HostIrJit
struct HostIrJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, allocate_fn> allocate_funcs_;
  std::unordered_map<const hir::LaunchKernel*, launch_kernel_fn> launch_kernel_funcs_;
};

// Helper function to exit on error on LLVM JIT initialization
template <typename T>
T ExitOnErr(llvm::Expected<T>&& E) {
  if (!E) {
    NVF_ERROR(
        false,
        "LLVM JIT Initialization Error: ",
        llvm::toString(E.takeError()));
    exit(1);
  }
  return std::move(*E);
}

inline void ExitOnErr(llvm::Error&& Err) {
  if (Err) {
    NVF_ERROR(
        false,
        "LLVM JIT Initialization Error: " + llvm::toString(std::move(Err)));
    exit(1);
  }
}

// Generate a function for LaunchKernel node
void generateLaunchKernelFunc(
    const hir::LaunchKernel* launch_kernel,
    llvm::Module* mod) {
  llvm::LLVMContext& context = mod->getContext();
  llvm::IRBuilder<> builder(context);

  std::string func_name = "launch_kernel_" +
          std::to_string(reinterpret_cast<uintptr_t>(launch_kernel));
  
  // Since we registered the wrapper functions with these exact names,
  // we can look them up directly without mangling
  std::string constructor_name = "KernelArgumentHolder::KernelArgumentHolder";
  std::string set_cache_name = "KernelArgumentHolder::setCacheId";
  std::string set_device_name = "KernelArgumentHolder::setDeviceIndex";
  std::string push_name = "KernelArgumentHolder::push";
  
  // Look up functions using the registered names
  llvm::Function* constructor_func = mod->getFunction(constructor_name);
  if (!constructor_func) {
    // Create function declaration for constructor
    llvm::FunctionType* ctor_type = llvm::FunctionType::get(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // return KernelArgumentHolder*
      {}, // no parameters for default constructor
      false
    );
    constructor_func = llvm::Function::Create(
      ctor_type, llvm::Function::ExternalLinkage, constructor_name, mod
    );
  }

  llvm::Function* push_func = mod->getFunction(push_name);
  if (!push_func) {
    // Create function declaration for member function
    std::vector<llvm::Type*> param_types = {
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // this pointer
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))    // at::Tensor object (passed as pointer for simplicity)
    };
    llvm::FunctionType* push_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), param_types, false
    );
    push_func = llvm::Function::Create(
      push_type, llvm::Function::ExternalLinkage, push_name, mod
    );
  }
  
  llvm::Function* set_cache_func = mod->getFunction(set_cache_name);
  if (!set_cache_func) {
    // Create function declaration for member function
    std::vector<llvm::Type*> param_types = {
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // this pointer
      llvm::Type::getInt64Ty(context)    // size_t parameter
    };
    llvm::FunctionType* set_cache_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), param_types, false
    );
    set_cache_func = llvm::Function::Create(
      set_cache_type, llvm::Function::ExternalLinkage, set_cache_name, mod
    );
  }
  
  llvm::Function* set_device_func = mod->getFunction(set_device_name);
  if (!set_device_func) {
    // Create function declaration for setDeviceIndex
    std::vector<llvm::Type*> param_types = {
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)) // this pointer only
    };
    llvm::FunctionType* set_device_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), param_types, false
    );
    set_device_func = llvm::Function::Create(
      set_device_type, llvm::Function::ExternalLinkage, set_device_name, mod
    );
  }
  
  // Create the main function
  // Parameters: cache_id, input_tensors_ptr, output_tensors_ptr
  std::vector<llvm::Type*> param_types = {
    llvm::Type::getInt64Ty(context),  // cache_id
    llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))), // input_tensors_ptr (at::Tensor**)
    llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)))  // output_tensors_ptr (at::Tensor**)
  };
  
  // Create struct type for return value
  std::vector<llvm::Type*> struct_elements = {
    llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // args pointer
    llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))  // outputs pointer
  };
  llvm::StructType* return_struct_type = llvm::StructType::create(context, struct_elements, "KernelArgumentHolderPair");
  
  llvm::FunctionType* main_func_type = llvm::FunctionType::get(
    return_struct_type, // return KernelArgumentHolderPair
    param_types,
    false
  );
  
  llvm::Function* main_func = llvm::Function::Create(
    main_func_type, llvm::Function::ExternalLinkage, func_name, mod
  );
  
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", main_func);
  builder.SetInsertPoint(entry);
  
  // Get function arguments
  llvm::Value* cache_id_arg = main_func->getArg(0);
  llvm::Value* input_tensors_ptr = main_func->getArg(1);
  llvm::Value* output_tensors_ptr = main_func->getArg(2);
  
  // Create KernelArgumentHolder args (for inputs)
  llvm::Value* args_ptr = builder.CreateCall(constructor_func, {});
  
  // Set cache ID if not monostate (cache_id != -1)
  llvm::Value* monostate_check = builder.CreateICmpNE(cache_id_arg, llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), -1));
  llvm::BasicBlock* set_cache_block = llvm::BasicBlock::Create(context, "set_cache", main_func);
  llvm::BasicBlock* skip_cache_block = llvm::BasicBlock::Create(context, "skip_cache", main_func);
  builder.CreateCondBr(monostate_check, set_cache_block, skip_cache_block);
  
  builder.SetInsertPoint(set_cache_block);
  builder.CreateCall(set_cache_func, {args_ptr, cache_id_arg});
  builder.CreateBr(skip_cache_block);
  
  builder.SetInsertPoint(skip_cache_block);
  
  // Push all input tensors to args
  for (size_t i = 0; i < launch_kernel->inputs().size(); ++i) {
    llvm::Value* tensor_ptr = builder.CreateGEP(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      input_tensors_ptr, 
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), i)
    );
    llvm::Value* input_tensor = builder.CreateLoad(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      tensor_ptr
    );
    
    // Push tensor to args - pass the tensor pointer to our wrapper
    builder.CreateCall(push_func, {args_ptr, input_tensor});
  }
  
  // Create KernelArgumentHolder outputs (for outputs)
  llvm::Value* outputs_ptr = builder.CreateCall(constructor_func, {});
  
  // Push all output tensors to outputs
  for (size_t i = 0; i < launch_kernel->outputs().size(); ++i) {
    llvm::Value* tensor_ptr = builder.CreateGEP(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      output_tensors_ptr, 
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), i)
    );
    llvm::Value* output_tensor = builder.CreateLoad(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      tensor_ptr
    );
    
    // Push tensor to outputs - pass the tensor pointer to our wrapper
    builder.CreateCall(push_func, {outputs_ptr, output_tensor});
  }
  
  // Set device index on args
  builder.CreateCall(set_device_func, {args_ptr});
  
  // Create the return struct
  llvm::Value* return_struct = llvm::UndefValue::get(return_struct_type);
  return_struct = builder.CreateInsertValue(return_struct, args_ptr, 0);
  return_struct = builder.CreateInsertValue(return_struct, outputs_ptr, 1);
  
  // Return the struct containing both args and outputs
  builder.CreateRet(return_struct);

  // Verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(!llvm::verifyModule(*mod, &error_stream), "LLVM module verification failed: " + error);

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    // Print the LLVM IR module
    llvm::outs() << "=== LLVM IR ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

// Generate a function for allocate node
void generateAllocateFunc(
    const kir::Allocate* allocate,
    llvm::Module* mod) {
  llvm::LLVMContext& context = mod->getContext();
  llvm::IRBuilder<> builder(context);

  std::string func_name = "create_tensor_from_sizes_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
  // Define function signature: at::Tensor func(const std::vector<int64_t>&
  // strides)
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();

  // Create function type: at::Tensor (*)(const std::vector<int64_t>&)
  // For simplicity, we'll use void* for at::Tensor return type
  llvm::PointerType* void_ptr_type =
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  llvm::FunctionType* func_type = llvm::FunctionType::get(
      void_ptr_type, {int64_ptr_type, int64_type, void_ptr_type}, false);

  // Create the function with the custom name
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, func_name, mod);

  // Create basic block
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_block);

  // Get function arguments
  llvm::Value* sizes_arg = func->getArg(0); // int64_t* (buffer)
  llvm::Value* ndim_arg = func->getArg(1); // int64_t   (length)
  llvm::Value* options_arg = func->getArg(2); // void*    (TensorOptions*)

  // Get the at::empty function pointer (registered in the JIT)
  llvm::Function* at_empty_func = mod->getFunction("at::empty");
  if (at_empty_func == nullptr) {
    llvm::FunctionType* at_empty_func_type = llvm::FunctionType::get(
        void_ptr_type, {int64_ptr_type, int64_type, void_ptr_type}, false);
    at_empty_func = llvm::Function::Create(
        at_empty_func_type, llvm::Function::ExternalLinkage, "at::empty", mod);
  }

  // Call at::empty
  llvm::Value* result =
      builder.CreateCall(at_empty_func, {sizes_arg, ndim_arg, options_arg});

  // Return the result
  builder.CreateRet(result);

  // Verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(!llvm::verifyModule(*mod, &error_stream), "LLVM module verification failed: " + error);

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    // Print the LLVM IR module
    llvm::outs() << "=== LLVM IR ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

void compile(const hir::HostIrContainer* container, llvm::orc::LLJIT* jit, std::unordered_map<const kir::Allocate*, allocate_fn>& allocate_funcs_, std::unordered_map<const hir::LaunchKernel*, launch_kernel_fn>& launch_kernel_funcs_) {
  if (allocate_funcs_.size() > 0) {
    return;
  }
  if (container == nullptr) {
    NVF_ERROR(false, "container is nullptr during host ir JIT compilation");
    return;
  }
  FUSER_PERF_SCOPE("HostIrJit::compile");
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>(
      "host_ir_container_" +
          std::to_string(reinterpret_cast<uintptr_t>(container)),
      *ctx);
  std::unordered_map<const kir::Allocate*, std::string> allocate_func_names;
  std::unordered_map<const hir::LaunchKernel*, std::string> launch_kernel_func_names;
  llvm::orc::JITDylib& dest_dynamic_lib = jit->getMainJITDylib();
  llvm::orc::MangleAndInterner mangler(
      dest_dynamic_lib.getExecutionSession(), jit->getDataLayout());
  for (auto expr : container->topLevelExprs()) {
    if (auto allocate = dynamic_cast<const kir::Allocate*>(expr)) {
      // Generate a unique function name for this allocate
      generateAllocateFunc(allocate, mod.get());
      // Store the mapping from allocate to function name
      allocate_func_names[allocate] = "create_tensor_from_sizes_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
    }
    else if (auto launch_kernel = dynamic_cast<const hir::LaunchKernel*>(expr)) {
      std::cout << "Generating launch kernel function" << std::endl;
      generateLaunchKernelFunc(launch_kernel, mod.get());
      // Store the mapping from launch_kernel to function name
      launch_kernel_func_names[launch_kernel] = "launch_kernel_" +
          std::to_string(reinterpret_cast<uintptr_t>(launch_kernel));
    }
  }

  // Add the module to the JIT
  ExitOnErr(jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));

  // Look up all functions and store their pointers
  for (const auto& [allocate, func_name] : allocate_func_names) {
    auto func_addr = ExitOnErr(jit->lookup(func_name));
    // Lookup and reinterpret the function pointer to store in the map
    allocate_funcs_[allocate] = allocate_fn(reinterpret_cast<void*(*)(int64_t*, int64_t, void*)>(func_addr.getValue()));
  }
  
  // Look up all launch kernel functions and store their pointers
  for (const auto& [launch_kernel, func_name] : launch_kernel_func_names) {
    auto func_addr = ExitOnErr(jit->lookup(func_name));
    // Lookup and reinterpret the function pointer to store in the map
    launch_kernel_funcs_[launch_kernel] = launch_kernel_fn(reinterpret_cast<KernelArgumentHolderPair(*)(int64_t, at::Tensor**, at::Tensor**)>(func_addr.getValue()));
  }
}

// Constructor implementation
HostIrJit::HostIrJit(hir::HostIrContainer* container, int num_threads) : pimpl_(new LlvmJitImpl) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  pimpl_->jit = ExitOnErr(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());
  llvm::orc::JITDylib& dest_dynamic_lib = pimpl_->jit->getMainJITDylib();
  llvm::orc::MangleAndInterner mangler(
      dest_dynamic_lib.getExecutionSession(), pimpl_->jit->getDataLayout());
  dest_dynamic_lib.addGenerator(
      ExitOnErr(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          pimpl_->jit->getDataLayout().getGlobalPrefix())));

  // Create wrapper function pointers to at::empty_strided and at::empty
  void* empty_strided_func_ptr = reinterpret_cast<void*>(
      +[](int64_t* sizes, int64_t ndim, int64_t* strides, void* options) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, ndim);
        at::TensorOptions opts = options
            ? *reinterpret_cast<at::TensorOptions*>(options)
            : at::TensorOptions();
        return new at::Tensor(
            at::empty_strided(aten_sizes, aten_strides, opts));
      });

  void* empty_func_ptr =
      reinterpret_cast<void*>(+[](int64_t* sizes, int64_t ndim, void* options) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::TensorOptions opts = options
            ? *reinterpret_cast<at::TensorOptions*>(options)
            : at::TensorOptions();
        return new at::Tensor(at::empty(aten_sizes, opts));
      });

  // Register at::empty_strided and at::empty functions in LLVM
  auto empty_strided_addr =
      llvm::orc::ExecutorAddr::fromPtr(empty_strided_func_ptr);
  auto empty_addr = llvm::orc::ExecutorAddr::fromPtr(empty_func_ptr);
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler("at::empty_strided")] = llvm::orc::ExecutorSymbolDef(
      empty_strided_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("at::empty")] =
      llvm::orc::ExecutorSymbolDef(empty_addr, llvm::JITSymbolFlags::Exported);

  // Register KernelArgumentHolder functions
  void* kernel_argument_holder_constructor_func_ptr = reinterpret_cast<void*>(
      +[]() -> KernelArgumentHolder* {
        return new KernelArgumentHolder();
      });

  void* kernel_argument_holder_set_cache_id_func_ptr = reinterpret_cast<void*>(
      +[](KernelArgumentHolder* self, size_t id) {
        self->setCacheId(id);
      });

  void* kernel_argument_holder_set_device_index_func_ptr = reinterpret_cast<void*>(
      +[](KernelArgumentHolder* self) {
        self->setDeviceIndex();
      });

  void* kernel_argument_holder_push_func_ptr = reinterpret_cast<void*>(
      +[](KernelArgumentHolder* self, at::Tensor* tensor_ptr) {
        // std::cout << "Wrapper function called with tensor_ptr: " << tensor_ptr << std::endl;
        if (tensor_ptr == nullptr) {
          // std::cout << "ERROR: tensor_ptr is null!" << std::endl;
          return;
        }
        // std::cout << "About to call self->push(*tensor_ptr)" << std::endl;
        self->push(*tensor_ptr);
        // std::cout << "Successfully called push" << std::endl;
      });

  auto kernel_argument_holder_constructor_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_constructor_func_ptr);
  auto kernel_argument_holder_set_cache_id_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_set_cache_id_func_ptr);
  auto kernel_argument_holder_set_device_index_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_set_device_index_func_ptr);
  auto kernel_argument_holder_push_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_push_func_ptr);
  
  symbolMap[mangler("KernelArgumentHolder::KernelArgumentHolder")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_constructor_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::setCacheId")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_set_cache_id_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::setDeviceIndex")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_set_device_index_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::push")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_push_addr, llvm::JITSymbolFlags::Exported);

  ExitOnErr(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));
  compile(container, pimpl_->jit.get(), pimpl_->allocate_funcs_, pimpl_->launch_kernel_funcs_);
}

HostIrJit::~HostIrJit() = default;

at::Tensor HostIrJit::allocate(
    const kir::Allocate* allocate,
    const std::vector<int64_t>& input_sizes) {
  if (pimpl_->allocate_funcs_.find(allocate) == pimpl_->allocate_funcs_.end()) {
    NVF_ERROR(false, "allocate function not found for ", allocate);
  }
  auto func_ptr = pimpl_->allocate_funcs_[allocate];
  at::TensorOptions opts = at::TensorOptions().device(at::kCUDA);
  void* result = func_ptr(
      const_cast<int64_t*>(input_sizes.data()),
      input_sizes.size(),
      reinterpret_cast<void*>(&opts));
  return *reinterpret_cast<at::Tensor*>(result);
}

HostIrJit::LaunchKernelResult HostIrJit::launchKernel(
    const hir::LaunchKernel* launch_kernel,
    int64_t cache_id,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  if (pimpl_->launch_kernel_funcs_.find(launch_kernel) == pimpl_->launch_kernel_funcs_.end()) {
    NVF_ERROR(false, "launch kernel function not found for ", launch_kernel);
  }
  
  auto func_ptr = pimpl_->launch_kernel_funcs_[launch_kernel];
  
  // std::cout << "Calling LLVM function with:" << std::endl;
  // std::cout << "  cache_id: " << cache_id << std::endl;
  // std::cout << "  inputs.size(): " << inputs.size() << std::endl;
  // std::cout << "  outputs.size(): " << outputs.size() << std::endl;
  
  // Convert const std::vector<at::Tensor>& to at::Tensor** arrays
  std::vector<at::Tensor*> input_ptrs;
  input_ptrs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    input_ptrs.push_back(const_cast<at::Tensor*>(&tensor));
  }
  
  std::vector<at::Tensor*> output_ptrs;
  output_ptrs.reserve(outputs.size());
  for (const auto& tensor : outputs) {
    output_ptrs.push_back(const_cast<at::Tensor*>(&tensor));
  }
  
  // Get raw pointer arrays
  at::Tensor** input_array = input_ptrs.data();
  at::Tensor** output_array = output_ptrs.data();
  
  KernelArgumentHolderPair result = func_ptr(cache_id, input_array, output_array);
  
  KernelArgumentHolder args = *reinterpret_cast<KernelArgumentHolder*>(result.args);
  KernelArgumentHolder outputs_holder = *reinterpret_cast<KernelArgumentHolder*>(result.outputs);
  
  return LaunchKernelResult{args, outputs_holder};
}

} // namespace nvfuser
