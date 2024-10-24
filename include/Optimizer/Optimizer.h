#pragma once

#include "Optimizer/Analyzer.h"
#include "Optimizer/Rewriter.h"
#include "Frontend/Operators.h"

#include "IR/IR.h"

#include <unordered_map>

struct BatchMatmulDescriptor {
  int m;
  int n;
  int k;
  bool transA;
  bool transB;
  std::vector<int> batch;
  void log() {
    llvm::errs() << "m = " << m << " n = " << n << " k = " << k << "\n";
    llvm::errs() << "batch = ";
    for (auto b : batch) llvm::errs() << b << ",";
    llvm::errs() << "\n";
    llvm::errs() << "trans A = " << transA << " transB = " << transB << "\n";
  }
};

namespace KernelCodeGen {

struct Optimizer {
  virtual bool applicable(mlir::ModuleOp& module) = 0;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) = 0;
  bool operator==(const Optimizer& other) {
    return name == other.name;
  }
  std::string name;
};

struct MatmulOptimizer : Optimizer {

  MatmulOptimizer() {
    this->name = std::move(std::string("Matmul"));
  }

  // bool isMatmulPattern(mlir::affine::AffineForOp forOp);

  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;

  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder);

  void clear() {
    matmuls.clear();
    matmulLoops.clear();
    matmulBuffers.clear();
  }

  // using the outermost loop represent a matmul.
  // std::set<mlir::affine::AffineForOp, CompareLoop> matmuls;
  std::set<mlir::func::FuncOp, CompareFunc> matmuls;


  // Map: from outermost loop to all loops in the matmul(loopM->[loopM, loopN, loopK]).
  // std::map<mlir::affine::AffineForOp, std::vector<mlir::affine::AffineForOp>, CompareLoop> matmulLoops;
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> matmulLoops;


  // Memory: A, B, C
  struct MemoryBuffer {
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
    // MemoryBuffer(mlir::Value A_, mlir::Value B_, mlir::Value C_) : A(A_), B(B_), C(C_) {}
  };

  // loopM->[A, B, C]
  // std::map<mlir::affine::AffineForOp, MemoryBuffer, CompareLoop> matmulBuffers;
  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> matmulBuffers;

  static std::map<std::string, int> matmulConfig;
};

struct BinaryOptimizer : Optimizer {
  BinaryOptimizer() {
    this->name = std::move(std::string("Binary"));
  }
  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;

  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras={}, 
                                const int needDims=0, const int oneDimNums=0);
  llvm::SmallVector<mlir::Value> getMinLoadOperands(int dim, llvm::SmallVector<mlir::Value> operands, const mlir::Value cst=nullptr);

  enum class BinaryType {
    constant = 0,
    allOne = 1,
    allEqual = 2,   // (2, 20, 256), (2, 20, 256)
    hasOneOrder = 3,  // (2, 20, 256), (1, 20, 256)   (2, 20, 256), (1, 256)
    noOneOrder = 4,  // (2, 20, 256), (20, 256)
    hasOneUnorder = 5    // (2, 20, 256), (20, 1)   (2, 20, 256), (1, 20, 1)
  };

  struct BinaryOpData {
    BinaryType type;
    int oneDimNums = 0;
    int needDimNums = 0;
  };

  void getBinaryOpData(mlir::Value max_load, mlir::Value min_load, BinaryOpData &data);
  
  void clear() {
    binaryBuffers.clear();
    binarys.clear();
    binaryLoops.clear();
  }

  struct MemoryBuffer {
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
  };

  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> binaryBuffers;
  std::set<mlir::func::FuncOp, CompareFunc> binarys;
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> binaryLoops;
  static std::map<std::string, int> binaryConfig;
};

struct ElementWiseOptimizer : Optimizer {
  ElementWiseOptimizer() {
    this->name = std::move(std::string("ElementWise"));
  }
  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;
  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras={});
  void clear() {
    elementWiseBuffers.clear();
    elementWises.clear();
    elementWiseLoops.clear();
  }

  struct MemoryBuffer {
    mlir::Value input;
    mlir::Value output;
  };

  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> elementWiseBuffers;
  std::set<mlir::func::FuncOp, CompareFunc> elementWises;
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> elementWiseLoops;
  static std::map<std::string, int> elementWiseConfig;
};

struct LayerNormOptimizer : Optimizer {
  LayerNormOptimizer() {
    this->name = std::move(std::string("LayerNorm"));
  }
  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;
  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras={});
  mlir::affine::AffineParallelOp combineParallel(std::vector<mlir::affine::AffineParallelOp> pals);
  mlir::affine::AffineForOp write(mlir::affine::AffineForOp forOp, std::vector<mlir::Value> buffers);
  mlir::affine::AffineForOp extractOpsFromLoop(mlir::affine::AffineForOp forOp, std::vector<mlir::Value> buffers);
  std::vector<mlir::affine::AffineForOp> read(mlir::affine::AffineForOp forOp, std::vector<mlir::Value> buffers);
  std::vector<mlir::Operation*> reduceUnrollOptimize(mlir::affine::AffineForOp forOp, mlir::affine::AffineParallelOp pal);
  void elementWiseUnrollOptimize(mlir::affine::AffineForOp forOp, mlir::affine::AffineParallelOp pal);

  void clear() {
    layerNormBuffers.clear();
    layerNorms.clear();
    layerNormLoops.clear();
  }

  struct MemoryBuffer {
    mlir::Value input;
    mlir::Value scale;
    mlir::Value bias;
    mlir::Value output;
  };

  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> layerNormBuffers;
  std::set<mlir::func::FuncOp, CompareFunc> layerNorms;
  std::map<mlir::func::FuncOp, std::vector<std::vector<mlir::affine::AffineForOp>>, CompareFunc> layerNormLoops;
  static std::map<std::string, int> layerNormConfig;
};

struct GatherOptimizer : Optimizer {
  GatherOptimizer() {
    this->name = std::move(std::string("Gather"));
  }
  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;
  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras={});
  void oneIndexLoad(mlir::affine::AffineForOp forOp, mlir::affine::AffineParallelOp pal);

  void clear() {
    gatherBuffers.clear();
    gathers.clear();
    gatherLoops.clear();
  }

  struct MemoryBuffer {
    mlir::Value input;
    mlir::Value indices;
    mlir::Value output;
  };

  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> gatherBuffers;
  std::set<mlir::func::FuncOp, CompareFunc> gathers;
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> gatherLoops;
  static std::map<std::string, int> gatherConfig;
};

struct FMHAOptimizer : Optimizer {

  FMHAOptimizer() {
    this->name = std::move(std::string("FMHA"));
  }

  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;

  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder);

  void softmaxIR(mlir::OpBuilder& builder, mlir::Value tileS, mlir::Value rowMax, mlir::Value smMax, mlir::Value rowSum, 
  mlir::Value smSum, mlir::Value smFac, mlir::Value zero, mlir::Value flt_min, mlir::Value tid);

  void clear() {
    uniqueFuncCalls.clear();
    call2callsMap.clear();
    call2bufferMap.clear();
  }

  ///< Avoid dumplicted cases.
  std::set<mlir::func::CallOp, CompareFuncCall> uniqueFuncCalls;


  // Map: from the first batched matmul call to {softmax, second batched matmul}
  std::map<mlir::func::CallOp, std::vector<mlir::func::CallOp>, CompareFuncCall> call2callsMap;

  // Memory: 
  struct MemoryBuffer {
    mlir::Value Q;
    mlir::Value K;
    mlir::Value S;
    mlir::Value V;
    mlir::Value O;
    BatchMatmulDescriptor matmul1;
    BatchMatmulDescriptor matmul2;
  };

  std::map<mlir::func::CallOp, MemoryBuffer, CompareFuncCall> call2bufferMap;

  static std::map<std::string, int> fmhaConfig;
};

struct BatchMatmulOptimizer : Optimizer {

  BatchMatmulOptimizer() {
    this->name = std::move(std::string("BatchMatmul"));
  }

  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) override;

  // mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const int64_t batchNum=0);
  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder);

  std::vector<mlir::Value> threadLevelOneToTwo(mlir::affine::AffineParallelOp pal);
  void clear() {
    batchMatmulBuffers.clear();
    batchMatmuls.clear();
    batchMatmulLoops.clear();
  }

  struct MemoryBuffer {
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
    BatchMatmulDescriptor matmul;
  };

  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> batchMatmulBuffers;
  std::set<mlir::func::FuncOp, CompareFunc> batchMatmuls;
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> batchMatmulLoops;
  static std::map<std::string, int> batchMatmulConfig;
  
};

}