#pragma once

#include "Frontend/Operators.h"
#include "Optimizer/Optimizer.h"
#include "Backend/CUDA.h"
#include "log.h"

// #include "ComputeDAG.h"
// #include "GraphTune.h"
// #include "AutoConfig.h"
// #include "Optimizer.h"
// #include "Element_Collecter.h"
// #include "Element_Parser.h"
// #include "CodeGen.h"

#include <string>
#include <sstream>
#include <fstream>
#include <initializer_list>
#include <climits>
#include <cfloat>
#include "llvm/ADT/StringRef.h"

namespace KernelCodeGen {

class KernelCodeGenerator {
public:
  KernelCodeGenerator(const std::string& platform_ = {"CUDA"}) : builder(&context), graph(builder), platform(std::move(platform_)) {
    initMLIRContext();
    // opts.push_back(std::move(std::make_unique<MatmulOptimizer>()));
    // opts.push_back(std::move(std::make_unique<BinaryOptimizer>()));
    // opts.push_back(std::move(std::make_unique<ElementWiseOptimizer>()));
    // opts.push_back(std::move(std::make_unique<LayerNormOptimizer>()));
    // opts.push_back(std::move(std::make_unique<GatherOptimizer>()));
    // opts.push_back(std::move(std::make_unique<FMHAOptimizer>()));
    matmulConfigs = {
      { {"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 8}, {"GROUP_SIZE_M", 8}, 
        {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
    };
    binaryConfigs = {
      {{"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}}
    };
    elementWiseConfigs = {
      {{"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}}
    };
    layerNormConfigs = {
      {{"BLOCK_SIZE", 2048}, {"THREAD_SIZE", 4}, {"VECTORIZE_WIDTH", 4}}
    };
    gatherConfigs = {
      {{"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}}
    };
    fmhaConfigs = {
      {{"BLOCK_SIZE", 128}, {"HdxBr", 128 * 64}, {"BrxBc", 128 * 64}, {"WarpX_O", 2}, {"Slice", 8},
       {"BrTileS", 8}, {"BcTileS", 8}, {"BrTileO", 8}, {"HdTileO", 8}, {"Width", 4}, {"WARP_SIZE", 32}}
    };
    batchMatmulConfigs = {
      { {"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 8}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, 
      {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
    };
  }
  KernelCodeGenerator() = delete;

  void initMLIRContext() {
    // context.getOrLoadDialect<mlir::compute_dag::ComputeDAGDialect>();
    // context.getOrLoadDialect<mlir::schedule::ScheduleDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    mlir::registerAllPasses();
  }

  ComputeDAG& createGraph(const std::string& graphName) {
    minLatency = FLT_MAX;
    graph.module = mlir::ModuleOp::create(builder.getUnknownLoc(), std::optional<::llvm::StringRef>(std::move(graphName)));
    graph.builder.setInsertionPointToEnd(graph.module.getBody());
    return graph;
  }

  ComputeDAG& getGraph() {
    return graph;
  }

  void dump(mlir::ModuleOp& module, const std::string& info = "") {
    if (KCGLog::level == Log::Release) return;
    llvm::errs() << "----------------------------------------------------------\n";
    llvm::errs() << "           " << info << "\n";
    llvm::errs() << "----------------------------------------------------------\n";
    module->dump();
    if (mlir::failed(mlir::verify(module))) {
      module->emitError("module verification error");
      assert(false);
    }
  }

  void save(const std::string& str, const std::string& file) {
    if (file == "terminal") {
      llvm::outs() << str;
      return;
    }
    std::ofstream fileWriter;
    fileWriter.open(file.c_str());
    std::stringstream stringStream;
    if (fileWriter.is_open()) {
      fileWriter << str;
      fileWriter.close();
    } else {
      llvm::errs() << "Can't open file \"" << file << "\"\n";
      return;
    }
    return;
  }

  void resetModule(mlir::ModuleOp& module) {
    mlir::Operation *cloned = backupModule_->clone();
    module = mlir::dyn_cast<mlir::ModuleOp>(cloned);   
  }

  void backupModule(mlir::ModuleOp& module) {
    mlir::Operation *cloned = module->clone();
    backupModule_ = mlir::dyn_cast<mlir::ModuleOp>(cloned);
  }

  void saveBestModule(mlir::ModuleOp& module) {
    mlir::Operation *cloned = module->clone();
    bestModule = mlir::dyn_cast<mlir::ModuleOp>(cloned);    
  }

  mlir::ModuleOp& optimize(ComputeDAG& graph_);

  float evaluate(mlir::ModuleOp& module) {
    return 0.0f;
  }

  std::string codegen(mlir::ModuleOp module) {
    if (platform == "CUDA") {
      return std::move(CUDAGen(module));
    }
  }

  void setLogMode(Log level) {
    KCGLog::level = level;
  }
public:
  std::vector<std::unique_ptr<Optimizer>> opts;

private:
  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::ModuleOp backupModule_;
  mlir::ModuleOp bestModule;
  ComputeDAG graph;
  std::string platform;
  float minLatency = FLT_MAX;
  std::vector<std::map<std::string, int>> matmulConfigs;
  std::vector<std::map<std::string, int>> fmhaConfigs;
  std::vector<std::map<std::string, int>> binaryConfigs;
  std::vector<std::map<std::string, int>> elementWiseConfigs;
  std::vector<std::map<std::string, int>> gatherConfigs;
  std::vector<std::map<std::string, int>> layerNormConfigs;
  std::vector<std::map<std::string, int>> batchMatmulConfigs;
};

}
