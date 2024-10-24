#include "KernelCodeGen.h"
#include "log.h"

namespace KernelCodeGen {

Log KCGLog::level = Log::Release;

mlir::ModuleOp& KernelCodeGenerator::optimize(ComputeDAG& graph_) {
  graph = graph_;
  mlir::Operation *cloned = graph.module->clone();
  auto module = mlir::dyn_cast<mlir::ModuleOp>(cloned);

  saveBestModule(module);

  for (auto& opt : opts) {
    backupModule(module);
    if (*opt == FMHAOptimizer()) {
      for (auto& fmhaConfig : fmhaConfigs) {
        FMHAOptimizer::fmhaConfig = fmhaConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (*opt == MatmulOptimizer()) {
      for (auto& matmulConfig : matmulConfigs) {
        MatmulOptimizer::matmulConfig = matmulConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (*opt == BinaryOptimizer()) {
      for (auto& binaryConfig : binaryConfigs) {
        BinaryOptimizer::binaryConfig = binaryConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (*opt == ElementWiseOptimizer()) {
      for (auto& elementWiseConfig : elementWiseConfigs) {
        ElementWiseOptimizer::elementWiseConfig = elementWiseConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (*opt == LayerNormOptimizer()) {
      for (auto& layerNormConfig : layerNormConfigs) {
        LayerNormOptimizer::layerNormConfig = layerNormConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (*opt == GatherOptimizer()) {
      for (auto& gatherConfig : gatherConfigs) {
        GatherOptimizer::gatherConfig = gatherConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          // bestModule->dump();
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (*opt == BatchMatmulOptimizer()) {
      for (auto& batchMatmulConfig : batchMatmulConfigs) {
        BatchMatmulOptimizer::batchMatmulConfig = batchMatmulConfig;
        resetModule(module);
        if (opt->applicable(module)) {
          opt->applyOptimzer(module, builder);
          // bestModule->dump();
          auto curLatency = evaluate(module);
          if (curLatency < minLatency) {
            minLatency = curLatency;
            saveBestModule(module);
          }
        }
      }
    } else if (opt->applicable(module)) {
      opt->applyOptimzer(module, builder);
      auto curLatency = evaluate(module);
      if (curLatency < minLatency) {
        minLatency = curLatency;
        saveBestModule(module);
      }
    }
  }
  return bestModule;
}

bool KernelCodeGenerator::lowering(mlir::ModuleOp &module) {
  mlir::PassManager pm(&context);
  pm.addPass(createLowerToLLVMPass());
  // pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  // pm.addPass(mlir::arith::populateArithToLLVMConversionPatterns());
  // pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  // pm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
  // pm.addPass(mlir::createConvertFuncToLLVMPass());
  if (mlir::failed(pm.run(module)))
    return false;
  return true;  
}
}
