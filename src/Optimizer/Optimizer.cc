#include "Optimizer/Optimizer.h"
#include "log.h"
#include <cfloat>

#define DUMP(module)                    \
{                                       \
  if (KCGLog::level == Log::Debug) {    \
    module.dump();                      \
  }                                     \
}

inline std::string toStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"float16"};
  if(type.isa<mlir::Float32Type>()) return {"float32"};
  if(type.isa<mlir::Float64Type>()) return {"float64"};
  if(auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    if (int_type.getWidth() == 1) return {"bool"};
    else if (int_type.getWidth() == 8) return {"int"};
    else if (int_type.getWidth() == 32) return {"int32_t"};
    else if (int_type.getWidth() == 64) return {"int64_t"};
  }
  if(type.isa<mlir::IndexType>()) return {"index"};
  return nullptr;
}

namespace KernelCodeGen {

std::map<std::string, int> MatmulOptimizer::matmulConfig;
std::map<std::string, int> FMHAOptimizer::fmhaConfig;
std::map<std::string, int> BinaryOptimizer::binaryConfig;
std::map<std::string, int> ElementWiseOptimizer::elementWiseConfig;
std::map<std::string, int> LayerNormOptimizer::layerNormConfig;
std::map<std::string, int> GatherOptimizer::gatherConfig;
std::map<std::string, int> BatchMatmulOptimizer::batchMatmulConfig;

struct LoadOrStoreOp {
  enum MemRSKind {
    LOAD = 0,
    STORE = 1,
  };

  LoadOrStoreOp() = default;
  LoadOrStoreOp(mlir::affine::AffineLoadOp loadOp_) : loadOp(loadOp_), kind(LOAD) {}
  LoadOrStoreOp(mlir::affine::AffineStoreOp storeOp_) : storeOp(storeOp_), kind(STORE) {}
  LoadOrStoreOp(const LoadOrStoreOp&  other) {
    kind = other.kind;
    kind == LoadOrStoreOp::LOAD ? loadOp = other.loadOp : storeOp = other.storeOp;
  }
  // LoadOrStoreOp& LoadOrStoreOp(LoadOrStoreOp&&  other) {
  //   kind = other.kind;
  //   kind == LoadOrStoreOp::LOAD ? loadOp = other.loadOp : storeOp = other.storeOp;
  //   return *this;
  // }
  mlir::affine::AffineForOp getParentLoop() {
    auto* parent = kind == LoadOrStoreOp::LOAD ? loadOp->getParentOp() :
                                                storeOp->getParentOp();
    auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(parent);
    return forOp;
  }

  mlir::Operation::operand_range getIndexes() {
    return kind == LoadOrStoreOp::LOAD ? loadOp.getIndices(): 
                                         storeOp.getIndices();
  }

  mlir::Value getMemory() {
    return kind == LoadOrStoreOp::LOAD ? loadOp.getMemref() : 
                                         storeOp.getMemref();
  }

  mlir::affine::AffineLoadOp loadOp;
  mlir::affine::AffineStoreOp storeOp;
  MemRSKind kind;
};

int getLoopIndex(const std::vector<mlir::affine::AffineForOp>& loops, mlir::affine::AffineForOp forOp) {
  int index = -1;
  for (auto loop : loops) {
    index += 1;
    if (loop == forOp) return index;
  }
  return -1;
}

int searchInductionVar(const std::vector<mlir::affine::AffineForOp>& loops, mlir::Value val) {
  auto ivArg = val.dyn_cast<mlir::BlockArgument>();
  if (!ivArg || !ivArg.getOwner())
    return -1;
  auto *containingInst = ivArg.getOwner()->getParent()->getParentOp();
  auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(containingInst);
  if (!forOp || forOp.getInductionVar() != val) return -1;

  return getLoopIndex(loops, forOp);
} 


// bool MatmulOptimizer::isMatmulPattern(mlir::affine::AffineForOp rootOp) {
//   std::vector<mlir::affine::AffineForOp> loops;
//   rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
//     loops.push_back(forOp);
//   });
//   ///TODO: descirpe [ Check 1 ]. (M, N, K) 3 nested loops.
//   if (loops.size() != 3) return false;
//   // Reduction loop.
//   if (loops[2].getIterOperands().size() == 0) return false;

//   std::map<mlir::affine::AffineForOp, std::vector<LoadOrStoreOp>, CompareLoop> scopeLdSt;

//   bool result = true;

//   auto collectLoadStoreOps = [&](LoadOrStoreOp& op) {
//     auto parentOp = op.getParentLoop();
//     auto index = getLoopIndex(loops, parentOp);
//     if (index == -1) {
//       result = false;
//       return;
//     }
//     if (scopeLdSt.count(loops[index]) == 0) {
//       scopeLdSt[loops[index]] = std::move(std::vector<LoadOrStoreOp>{op});
//     } else {
//       auto ldstVector = scopeLdSt.find(loops[index]);
//       ldstVector->second.push_back(op);
//     }
//   };

//   rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
//     auto op = LoadOrStoreOp(loadOp);
//     collectLoadStoreOps(op);
//   });
//   if (!result) return false;
//   rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp storeOp) {
//     auto op = LoadOrStoreOp(storeOp);
//     collectLoadStoreOps(op);
//   });
//   if (!result) return false;

//   MemoryBuffer buf;

//   ///TODO: [ Check 2 ]
//   auto MNLoopScopeCheck = [&]() {
//     // need to store C[i][j] in the MN scope.
//     // but we can't expect it equals to 1, 
//     //  as there maybe kernel fusion bringing additional memory access.
//     if (scopeLdSt.count(loops[1]) == 0) {
//       result = false;
//       return;
//     }
//     bool storeCij = false;

//     auto mnLdStVector = scopeLdSt.find(loops[1])->second;
//     for (auto ldst : mnLdStVector) {
//       auto&& indexes = ldst.getIndexes();
//       std::vector<int> offsets;
//       // all index to memory buffer must depend on M，N Loops.
//       for (auto index : indexes) {
//         int offset = searchInductionVar({loops[0], loops[1]}, index);
//         if (offset == -1) {
//           result = false;
//           return;
//         }
//         offsets.push_back(offset);
//       }
      
//       if (!storeCij && ldst.kind == LoadOrStoreOp::STORE) {
//         if (offsets == std::vector<int> {0, 1}) {
//           storeCij = true;
//           buf.C = ldst.getMemory();
//         }
//       }
//     }

//     if (!storeCij) {
//       result = false;
//     }
//   };

//   ///TODO: [ Check 3 ]
//   auto KLoopScopeCheck = [&]() {
//     //at least: read A[i][k], read B[k][j]
//     if (scopeLdSt.count(loops[2])  == 0 ||
//         scopeLdSt.find(loops[2])->second.size() < 2) {
//       result = false;
//       return;
//     }
//     bool readAik = false;
//     bool readBkj = false;
//     // bool writeCij = false;
//     auto mnLdStVector = scopeLdSt.find(loops[2])->second;
//     for (auto ldst : mnLdStVector) {
//       auto&& indexes = ldst.getIndexes();
//       std::vector<int> offsets;
//       for (auto index : indexes) {
//         // all index to memory buffer must depend on M，N, K Loops.
//         int offset = searchInductionVar({loops[0], loops[1], loops[2]}, index);
//         if (offset == -1) {
//           result = false;
//           return;
//         }
//         offsets.push_back(offset);
//       }

//       if (!readAik && ldst.kind == LoadOrStoreOp::LOAD) {
//         if (offsets == std::vector<int> {0, 2}) {
//           readAik = true;
//           buf.A = ldst.getMemory();
//         }
//       }

//       if (!readBkj && ldst.kind == LoadOrStoreOp::LOAD) {
//         if (offsets == std::vector<int> {2, 1}) {
//           readBkj = true;
//           buf.B = ldst.getMemory();
//         }
//       }
      
//       // if (!writeCij && ldst.kind == LoadOrStoreOp::STORE) {
//       //   if (offsets == std::vector<int> {0, 1}) {
//       //     writeCij = true;
//       //     assert(buf.C == ldst.getMemory());
//       //   }
//       // }
//     }

//     // if (!(readAik && readBkj && writeCij)) {
//     //   result = false;
//     // }
//     if (!(readAik && readBkj)) {
//       result = false;
//     }
//   };

//   MNLoopScopeCheck();
//   KLoopScopeCheck();

//   if (result) {
//     matmuls.insert(rootOp);
//     matmulLoops[rootOp] = loops;
//     matmulBuffers[rootOp] = buf;
//   }
//   return result;
// }

bool MatmulOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& matmulFuncs = Analyzer::collectFunctions(module, "Matmul");
  bool res = matmulFuncs.size() != 0 ? true : false;

  for (auto& matmulFunc : matmulFuncs) {
    if (matmuls.count(matmulFunc) != 0 || matmulLoops.count(matmulFunc) != 0
      || matmulBuffers.count(matmulFunc) != 0) {
      llvm::errs() << "Duplicated Matmul in module\n";
    }
    auto funcName = matmulFunc.getSymName();
    if (funcName.str().find("BatchMatmul") != std::string::npos) continue;
    matmuls.insert(matmulFunc);
    auto&& loops = Analyzer::collectFuncLoops(matmulFunc);
    matmulLoops[matmulFunc] = std::move(loops);
    auto funcArgs = matmulFunc.front().getArguments();
    // matmulBuffers[matmulFunc] = MemoryBuffer(mlir::dyn_cast<mlir::Value>(funcArgs[0]), 
    //   mlir::dyn_cast<mlir::Value>(funcArgs[1]), mlir::dyn_cast<mlir::Value>(funcArgs[2]));
    MemoryBuffer ABC;
    ABC.A = funcArgs[0];
    ABC.B = funcArgs[1];
    auto &block = matmulFunc.front();
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    ABC.C = returnOp.getOperand(0);
    // matmulBuffers[matmulFunc] = MemoryBuffer(funcArgs[0].dyn_cast<mlir::Value>(), 
    //   funcArgs[1].dyn_cast<mlir::Value>(), funcArgs[2].dyn_cast<mlir::Value>());
    matmulBuffers[matmulFunc] = ABC;
  }
  return res;
}

int64_t smAReadSride(int64_t blockDim, int64_t warpSize) {
  int64_t warpNum = blockDim / warpSize;
  int64_t laneNum = warpSize;
  //warp orgnize: 2 x 4
  std::vector<int64_t> warpOrg {2, 4};
  std::vector<int64_t> threadOrg {8, 4};
  return (warpNum / warpOrg[1]) * threadOrg[0];
}

int64_t smBReadSride(int64_t blockDim, int64_t warpSize) {
  int64_t warpNum = blockDim / warpSize;
  int64_t laneNum = warpSize;
  //warp orgnize: 2 x 4
  std::vector<int64_t> warpOrg {2, 4};
  std::vector<int64_t> threadOrg {8, 4};
  return (warpNum / warpOrg[0]) * threadOrg[1];
}

mlir::AffineMap MatmulOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto dim7 = builder.getAffineDimExpr(7);
  int64_t blockDimY = matmulConfig["BLOCK_SIZE_M"] / matmulConfig["THREAD_SIZE_M"];
  int64_t blockDimX = matmulConfig["BLOCK_SIZE_N"] / matmulConfig["THREAD_SIZE_N"];
  bool vectorize = matmulConfig.count("VECTORIZE_WIDTH") != 0;
  int width = vectorize ? matmulConfig["VECTORIZE_WIDTH"] : 1;

  std::vector<int64_t> warpOrg {2, 4};  
  std::vector<int64_t> threadOrg {8, 4};

  if (mapIdentifier == "loadTileA") {
    // dims are:[dim0, dim1, dim2, dim3, dim4]
    // operands are: [threadIdx.y, threadIdx.x, blockIdx.y, k_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim4 * blockDimY * blockDimX;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width);
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width); 
    auto M_Base = dim2 * matmulConfig["BLOCK_SIZE_M"];
    auto K_Base = dim3;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(M_Offset + M_Base);
    exprs.push_back(K_Offset * width + K_Base);
    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadTileB") {
    // dims are:[dim0, dim1, dim2, dim3, dim4]
    // operands are: [threadIdx.y, threadIdx.x, k_outer, blockIdx.x, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim4 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width); 
    auto K_Base = dim2;
    auto N_Base = dim3 * matmulConfig["BLOCK_SIZE_N"];
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset + K_Base);
    exprs.push_back(N_Offset * width + N_Base);
    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileA") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, iv, ivInVector]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width);
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset * width + dim3);
    exprs.push_back(M_Offset);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileB") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.y, threadIdx.x, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset);
    exprs.push_back(N_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragA") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, k_inner, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(matmulConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(matmulConfig["WARP_SIZE"]);

    auto M_offset = laneId.floorDiv(threadOrg[1]) + threadOrg[0] * (warpId.floorDiv(warpOrg[1]) + dim3 * warpOrg[0]);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(M_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragB") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, k_inner, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(matmulConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(matmulConfig["WARP_SIZE"]);

    auto N_offset = laneId % threadOrg[1] + threadOrg[1] * (warpId % warpOrg[1] + dim3 * warpOrg[1]);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(N_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheReadA" || mapIdentifier == "cacheReadB") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheWriteC") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7]
    // operands are: [threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, iv0, iv1, iv2, iv3]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(matmulConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(matmulConfig["WARP_SIZE"]);

    auto M_offset = laneId.floorDiv(threadOrg[1]) + threadOrg[0] * (warpId.floorDiv(warpOrg[1]) + dim4.floorDiv(width) * warpOrg[0]);
    auto N_offset = laneId % threadOrg[1] + threadOrg[1] * (warpId % warpOrg[1] + dim5.floorDiv(width) * warpOrg[1]);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim2 * matmulConfig["BLOCK_SIZE_M"] + M_offset * width + dim6);
    exprs.push_back(dim3 * matmulConfig["BLOCK_SIZE_N"] + N_offset * width + dim7);
    return mlir::AffineMap::get(/*dimCount*/8, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void MatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto& matmul : matmuls) {
    matmul->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto loops = matmulLoops[matmul];
    auto loopM = loops[0], loopN = loops[1], loopK = loops[2];
    auto buffers = matmulBuffers[matmul];
    auto A = buffers.A, B = buffers.B, C = buffers.C;
    
    auto m_axes = Rewriter::split(loopM, 3, {matmulConfig["THREAD_SIZE_M"], matmulConfig["BLOCK_SIZE_M"]});
    auto n_axes = Rewriter::split(loopN, 3, {matmulConfig["THREAD_SIZE_N"], matmulConfig["BLOCK_SIZE_N"]});

    DUMP(module);

    auto m_outer = m_axes[0], m_mider = m_axes[1], m_inner = m_axes[2];
    auto n_outer = n_axes[0], n_mider = n_axes[1], n_inner = n_axes[2];


    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    DUMP(module);

    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    DUMP(module);


    std::vector<mlir::affine::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
    auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
    loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];
    DUMP(module);

    Rewriter::reorder({loopK, m_inner, n_inner});
    DUMP(module);

    auto k_axes = Rewriter::split(loopK, 2, {matmulConfig["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_inner = k_axes[1];
    DUMP(module);

    int64_t blockThreads;
    auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);

    auto ldgASize = matmulConfig["BLOCK_SIZE_K"] * matmulConfig["BLOCK_SIZE_M"] / blockThreads;
    auto ldgBSize = matmulConfig["BLOCK_SIZE_K"] * matmulConfig["BLOCK_SIZE_N"] / blockThreads;
    auto fragASize = matmulConfig["BLOCK_SIZE_M"] / smAReadSride(blockThreads, matmulConfig["WARP_SIZE"]);
    auto fragBSize = matmulConfig["BLOCK_SIZE_N"] / smBReadSride(blockThreads, matmulConfig["WARP_SIZE"]);
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();

    auto fragB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragBSize}, elementB);
    auto fragA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragASize}, elementA);

    auto tileB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgBSize}, elementB);
    auto tileA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgASize}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {matmulConfig["BLOCK_SIZE_K"], matmulConfig["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {matmulConfig["BLOCK_SIZE_K"], matmulConfig["BLOCK_SIZE_M"]}, elementA);
    DUMP(module);
    
    auto blockIdx = Rewriter::getParallelIdx(gridLevel);
    auto threadIdx = Rewriter::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder);
    auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], k_outer, Position::begin);
    auto loadTileBMap = getAffineMap("loadTileB", builder);
    auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, 
                      {threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[1]}, 
                      matmulConfig["VECTORIZE_WIDTH"], loadTileA, Position::after);
    DUMP(module);

    auto storeTileAMap = getAffineMap("storeTileA", builder);
    auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, 
                        matmulConfig["VECTORIZE_WIDTH"], loadTileB, Position::after);
    auto storeTileBMap = getAffineMap("storeTileB", builder);
    auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, 
                                matmulConfig["VECTORIZE_WIDTH"], storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);

    DUMP(module);

    auto loadFragAMap = getAffineMap("loadFragA", builder);
    auto loadFragA = Rewriter::read(smA, fragA, loadFragAMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], k_inner, Position::begin);
    auto loadFragBMap = getAffineMap("loadFragB", builder);
    auto loadFragB = Rewriter::read(smB, fragB, loadFragBMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], loadFragA, Position::after);
    DUMP(module);

    Rewriter::cache_read(k_inner, A, fragA, getAffineMap("cacheReadA", builder), {m_inner.getInductionVar()});
    Rewriter::cache_read(k_inner, B, fragB, getAffineMap("cacheReadB", builder), {n_inner.getInductionVar()});
    DUMP(module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {matmulConfig["VECTORIZE_WIDTH"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {matmulConfig["VECTORIZE_WIDTH"]});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
    DUMP(module);

    Rewriter::cache_write(m_inner_0, C, C, getAffineMap("cacheWriteC", builder), 
                          {threadIdx[0], threadIdx[1], blockIdx[0], blockIdx[1], m_inner_0.getInductionVar(),
                          n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), n_inner_1.getInductionVar()});
    DUMP(module);

    Rewriter::vectorize(n_inner_1, matmulConfig["VECTORIZE_WIDTH"]);
    DUMP(module);
    
    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    DUMP(module);

    Rewriter::detach_last_loop(k_inner);
    DUMP(module);

    Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
    Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
    Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
    Rewriter::schedule(doubleLoadTileB[1][0], doubleLoadTileA[1][0], Position::after);
    Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
    Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
    auto ifOp = doubleLoadTileA[1][1]->getParentOp();
    Rewriter::schedule(ifOp, k_inner, Position::after); 
    Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
    Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
    Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
    Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
    DUMP(module);

    Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
    Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);;
    DUMP(module);

    Rewriter::take_off_true_if(module);
    Rewriter::delete_false_if(module);
    DUMP(module);

    int64_t threshold = std::max(matmulConfig["BLOCK_SIZE_K"], std::max(matmulConfig["THREAD_SIZE_M"], matmulConfig["THREAD_SIZE_N"]));
    Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times >= std::min<int64_t>(threshold, matmulConfig["VECTORIZE_WIDTH"])) return false;
      return true;
    });
    DUMP(module);

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times > threshold) return false;
      return true;
    });
    DUMP(module);
  }
}

/*----------------------------binary---------------------------------*/

std::vector<int64_t> getCreateAffineMapArgs(std::vector<mlir::affine::AffineForOp> loops) {
  std::vector<int64_t> extras;
  extras.push_back(loops.size());
  for (int i=0; i<loops.size(); i++) {
    auto sum = 1;
    for (int j=i+1; j<loops.size(); j++) {
      sum *= loops[j].getUpperBoundMap().getSingleConstantResult();
    }
    if (i != extras[0] -1) {extras.push_back(sum);}
    else {extras.push_back(loops.back().getUpperBoundMap().getSingleConstantResult());}
  }
  return extras;
}

bool BinaryOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& binaryFuncs = Analyzer::collectFunctions(module, "Binary");
  bool res = binaryFuncs.size() != 0 ? true : false;

  for (auto& binaryFunc : binaryFuncs) {
    if (binarys.count(binaryFunc) != 0 || binaryLoops.count(binaryFunc) != 0
      || binaryBuffers.count(binaryFunc) != 0) {
      llvm::errs() << "Duplicated binary in module\n";
    }
    binarys.insert(binaryFunc);
    auto&& loops = Analyzer::collectFuncLoops(binaryFunc);
    binaryLoops[binaryFunc] = std::move(loops);
    auto funcArgs = binaryFunc.front().getArguments();

    MemoryBuffer ABC;
    ABC.A = funcArgs[0];
    if (funcArgs.size() == 2) {ABC.B = funcArgs[1];}
    else {ABC.B = nullptr;}
    auto &block = binaryFunc.front();
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    ABC.C = returnOp.getOperand(0);
    binaryBuffers[binaryFunc] = ABC;
  }
  return res;
}

mlir::AffineMap BinaryOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, 
                                              const std::vector<int64_t> &extras, const int needDimNums, const int oneDimNums) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto width = 4;  // float4 type width

  if (mapIdentifier == "MaxLoadOrStore") {
    auto oneDimExpr_y = dim0 + dim1 + dim2;
    auto oneDimExpr_x = dim3 + dim4 + dim5 * width;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    if (extras[0] == 2) {
      exprs.push_back(oneDimExpr_y);
      exprs.push_back(oneDimExpr_x);
    } else {
      auto oneDimExpr = oneDimExpr_y * extras.back() + oneDimExpr_x;  // 这个是多维映射到一维的index
      for (int i=0; i<extras[0]; i++) {  // [old_loops_len, 5120, 256, 256, 80]
        if (i == 0) {
          exprs.push_back(oneDimExpr.floorDiv(extras[i+1]));
        } else if (i != extras[0] - 1) {
          auto tmpExpr = oneDimExpr % extras[i];
          exprs.push_back(tmpExpr.floorDiv(extras[i+1]));
        } else {
          exprs.push_back(oneDimExpr % extras[i+1]);
        }
      }
    }
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "MinVectorLoad") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    if (extras[0] == 2) {
      if (oneDimNums) {
        exprs.push_back(dim0);
        auto expr = dim1 + dim2 + dim3 * width;
        exprs.push_back(expr);
        return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      }
      auto expr = dim0 + dim1 + dim2 * width;
      exprs.push_back(expr);
      return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
    }
    mlir::AffineExpr oneDimExpr_y, oneDimExpr_x;
    if (oneDimNums) {
      oneDimExpr_y = dim1 + dim2 + dim3;
      oneDimExpr_x = dim4 + dim5 + dim6 * width;
      for (int i=0; i<oneDimNums; i++) {exprs.push_back(dim0);}
    } else {
      oneDimExpr_y = dim0 + dim1 + dim2;
      oneDimExpr_x = dim3 + dim4 + dim5 * width;
    }
    auto oneDimExpr = oneDimExpr_y * extras.back() + oneDimExpr_x;

    for (int i=extras[0] - needDimNums; i<extras[0]; i++) {
      if (i == 0) {
        exprs.push_back(oneDimExpr.floorDiv(extras[i+1]));
      } else if (i != extras[0] - 1) {
        auto tmpExpr = oneDimExpr % extras[i];
        exprs.push_back(tmpExpr.floorDiv(extras[i+1]));
      } else {
        exprs.push_back(oneDimExpr % extras[i+1]);
      }
    }
    auto dimCount = 6;
    if (oneDimNums) {dimCount++;}
    // llvm::outs() << oneDimNums << "\n\n";
    // for (auto exp : exprs) { llvm::outs() << exp << "\n"; }
    // llvm::outs() << "\n";
    return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "CacheLoadOrStore") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void BinaryOptimizer::getBinaryOpData(mlir::Value max_load, mlir::Value min_load, BinaryOpData &data) {
  auto max_type = max_load.getType();
  auto max_shape = max_type.dyn_cast<mlir::MemRefType>().getShape();
  if (!min_load) {  // 常数
    data.type = BinaryType::constant;
  } else {  // 不为常数
    auto min_type = min_load.getType();
    auto min_shape = min_type.dyn_cast<mlir::MemRefType>().getShape();
    bool shapeIsOne = true;   // check全部为1
    for (auto shape : min_shape) {
      if (shape != 1) {shapeIsOne = false; break;}
    }
    if (shapeIsOne) {  // 全为1
      data.type = BinaryType::allOne;
    } else {
      if (min_shape.size() == max_shape.size()) {  // 维度相等
        bool shapeEq = true;
        for (int i=0; i<max_shape.size(); i++) {  // check 全部相等
          if (min_shape[i] != max_shape[i]) {shapeEq = false; break;}
        }
        if (shapeEq) {  // shape位全部相等
          data.type = BinaryType::allEqual;
        } else {  // shape位不全等
          bool lenEqLastEq = true, first = false;
          for (int i=0; i<max_shape.size(); i++) {
            if (min_shape[i] != max_shape[i]){
              if (first) {lenEqLastEq = false;}
              data.oneDimNums++;
            } else {
              first = true;
              data.needDimNums++;
            }
          }
          if (lenEqLastEq) {  //维度相等，shape不等，按顺序
            data.type = BinaryType::hasOneOrder;
          } else {   // 维度相等，shape不等，无顺序
            data.type = BinaryType::hasOneUnorder;
          }
        }
      } else {  // 维度不等
        bool lastEq = true, shrotLastEq = true, first = false;
        int dst = max_shape.size() - min_shape.size();
        for (int i=0; i<min_shape.size(); i++) {
          if (max_shape[i+dst] != min_shape[i]) {
            lastEq = false;
            if (first) {shrotLastEq = false;}
            data.oneDimNums++;
          } else {
            first = true;
            data.needDimNums++;
            }
        }
        if (lastEq) {   // 维度不等，shape相等
          data.type = BinaryType::noOneOrder;
        } else if (shrotLastEq){  // 维度不等，shape不等, 有序
          data.type = BinaryType::hasOneOrder;
        } else {  // 维度相等，shape不等，无序
          data.type = BinaryType::hasOneUnorder;
        }
      }
    }
  }
}

llvm::SmallVector<mlir::Value> BinaryOptimizer::getMinLoadOperands(int dim, llvm::SmallVector<mlir::Value> operands, const mlir::Value cst) {
  llvm::SmallVector<mlir::Value> minOperands;
  if (dim == 2) {
    if (cst) { minOperands = {cst, operands[3], operands[4]}; } 
    else { minOperands = {operands[3], operands[4]}; }
  } else {
    minOperands = operands;
    if (cst) { minOperands.insert(minOperands.begin() + 0, cst); }
  }
  // for (auto exp : minOperands) { llvm::outs() << exp << "\n"; }
  // llvm::outs() << "\n";
  return minOperands;
}

// void BinaryOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
//   for (auto binary : binarys) {
//     auto loops = binaryLoops[binary];
//     auto buffers = binaryBuffers[binary];
//     auto max_load = buffers.A, min_load = buffers.B, result_store = buffers.C;

//     auto extras = getCreateAffineMapArgs(loops);
//     auto new_loops = Rewriter::combineToTowDim(loops);   // 
//     extras.push_back(new_loops[1].getUpperBoundMap().getSingleConstantResult());
//     auto dimY = new_loops[0].getUpperBoundMap().getSingleConstantResult();
//     auto dimX = new_loops[1].getUpperBoundMap().getSingleConstantResult();

//     DUMP(module);
//     // 循环切块大小
//     auto split_out_loops = Rewriter::split(new_loops[0], 3, {binaryConfig["THREAD_SIZE_M"], binaryConfig["BLOCK_SIZE_M"]});  // 第一个是一个thread计算的维度，第二个是一个block计算的多大的维度
//     auto split_in_loops = Rewriter::split(new_loops[1], 3, {binaryConfig["THREAD_SIZE_N"], binaryConfig["BLOCK_SIZE_N"]});   // 
//     DUMP(module);

//     auto out_outer = split_out_loops[0], out_mider = split_out_loops[1], out_inner = split_out_loops[2];
//     auto in_outer = split_in_loops[0], in_mider = split_in_loops[1], in_inner = split_in_loops[2];
//     Rewriter::reorder({out_outer, in_outer, out_mider, in_mider, out_inner, in_inner});
//     DUMP(module);

//     auto gridLevel = Rewriter::parallel({out_outer, in_outer});
//     auto blockLevel = Rewriter::parallel({out_mider, in_mider});
//     DUMP(module);

//     auto *op = gridLevel->getParentOp();
//     auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);
//     funcOp->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));

//     auto blockElemIdx = Rewriter::getElementIdx(gridLevel);
//     auto ThreadElemIdx = Rewriter::getElementIdx(blockLevel);
    
//     if (dimX % binaryConfig["THREAD_SIZE_N"] || dimY % binaryConfig["THREAD_SIZE_M"]) {
//       std::vector<int> range{dimY, dimX, binaryConfig["THREAD_SIZE_M"], binaryConfig["THREAD_SIZE_N"]};
//       llvm::SmallVector<mlir::Value> operands{blockElemIdx[0], blockElemIdx[1], ThreadElemIdx[0], ThreadElemIdx[1]};  // by bx ty tx
//       auto ifop = Rewriter::irregularMat(out_inner, range, operands);
//       DUMP(module);
//     } else {
//       auto max_type = max_load.getType().dyn_cast<mlir::MemRefType>();
//       auto element = max_type.getElementType();

//       auto loadOrStoreMap = getAffineMap("MaxLoadOrStore", builder, extras);
//       auto cacheLoadOrStore = getAffineMap("CacheLoadOrStore", builder);

//       auto frag_a = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {binaryConfig["THREAD_SIZE_N"]}, element);  // 计算A -> reg
//       auto frag_c = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {binaryConfig["THREAD_SIZE_N"]}, element);  // 计算c -> reg

//       llvm::SmallVector<mlir::Value> operands{blockElemIdx[0], ThreadElemIdx[0], out_inner.getInductionVar(), blockElemIdx[1], ThreadElemIdx[1]};
//       auto load_a = Rewriter::read(max_load, frag_a, loadOrStoreMap, operands, binaryConfig["VECTORIZE_WIDTH"], out_inner, Position::begin);
//       auto store_c = Rewriter::write(frag_c, result_store, loadOrStoreMap, operands, binaryConfig["VECTORIZE_WIDTH"], in_inner, Position::after);
//       DUMP(module);

//       Rewriter::cache_read(in_inner, max_load, frag_a, cacheLoadOrStore, {in_inner.getInductionVar()});
//       Rewriter::cache_write(in_inner, result_store, frag_c, cacheLoadOrStore, {in_inner.getInductionVar()});
//       DUMP(module);

//       BinaryOpData data;
//       getBinaryOpData(max_load, min_load, data);   // 判断binary操作是什么类型的

//       switch (data.type) {
//         case BinaryType::constant: { //加常数（2，20，256） 1
//           in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantOp cst) {
//             Rewriter::schedule(cst, out_inner, Position::before);
//           });
//           break;
//         }
//         case BinaryType::allOne: { //全为1（2，20，256）（1, 1, 1）
//           in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantOp cst) {
//             Rewriter::schedule(cst, out_inner, Position::before);
//           });
//           in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
//             if (loadOp.getMemref() == min_load){
//               Rewriter::schedule(loadOp, out_inner, Position::before);
//             }
//           });
//           break;
//         }
//         case BinaryType::allEqual: { // 维度相等，shape相等（2，20，256）（2，20，256） vcetorize all
//           auto min_type = min_load.getType();
//           auto element_ = min_type.dyn_cast<mlir::MemRefType>().getElementType();
//           auto frag_b = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {binaryConfig["THREAD_SIZE_N"]}, element_);  // 计算b -> reg
//           auto load_b = Rewriter::read(min_load, frag_b, loadOrStoreMap, operands, binaryConfig["VECTORIZE_WIDTH"], load_a, Position::after);
//           Rewriter::cache_read(in_inner, min_load, frag_b, cacheLoadOrStore, {in_inner.getInductionVar()});
//           break;
//         }
//         case BinaryType::hasOneOrder: { // 维度相等，shape不等，按顺序（2， 20， 256） （1， 20， 256） vcetorize all
//           auto min_type = min_load.getType();
//           auto element_ = min_type.dyn_cast<mlir::MemRefType>().getElementType();
//           auto frag_b = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {binaryConfig["THREAD_SIZE_N"]}, element_);  // 计算b -> reg
//           mlir::Value cst;
//           in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantIndexOp cstOp) {
//             Rewriter::schedule(cstOp, blockLevel, Position::begin);
//             cst = cstOp.getResult();
//           });
//           auto minVectorLoad = getAffineMap("MinVectorLoad", builder, extras, data.needDimNums, data.oneDimNums);
//           auto minOperands = getMinLoadOperands(extras[0], operands, cst);
//           auto load_b = Rewriter::read(min_load, frag_b, minVectorLoad, minOperands, binaryConfig["VECTORIZE_WIDTH"], load_a, Position::after);
//           Rewriter::cache_read(in_inner, min_load, frag_b, cacheLoadOrStore, {in_inner.getInductionVar()});
//           break;
//         }
//         case BinaryType::noOneOrder: { // 维度不等/后续相等(2, 20, 256)  (20, 256) vcetorize all
//           auto min_type = min_load.getType();
//           auto element_ = min_type.dyn_cast<mlir::MemRefType>().getElementType();
//           auto frag_b = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {binaryConfig["THREAD_SIZE_N"]}, element_);  // 计算b -> reg
//           auto minVectorLoad = getAffineMap("MinVectorLoad", builder, extras, data.needDimNums, data.oneDimNums);
//           auto minOperands = getMinLoadOperands(extras[0], operands);
//           auto load_b = Rewriter::read(min_load, frag_b, minVectorLoad, minOperands, binaryConfig["VECTORIZE_WIDTH"], load_a, Position::after);
//           Rewriter::cache_read(in_inner, min_load, frag_b, cacheLoadOrStore, {in_inner.getInductionVar()});
//           break;
//         }
//         case BinaryType::hasOneUnorder: { // 维度不等/后续不等/后后续不等（2，20，256）（20，1） vcetorize max
//           // auto min_type = min_load.getType();
//           // auto element_ = min_type.dyn_cast<mlir::MemRefType>().getElementType();
//           // auto noVectorLoadOrStore = getAffineMap("NoVectorLoadOrStore", builder, extras);
//           // auto frag_b = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {binaryConfig["THREAD_SIZE_N"]}, element_);  // 计算b -> reg
//           // auto load_b = Rewriter::noVcetorRead(min_load, frag_b, noVectorLoadOrStore, operands, load_a, Position::after);
//           // Rewriter::cache_read(in_inner, min_load, frag_b, cacheLoadOrStore, {in_inner.getInductionVar()});
//           in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantOp cst) {
//             Rewriter::schedule(cst, out_inner, Position::before);
//           });
//           break;
//         }
//         default:
//           assert(false);
//       }
//       DUMP(module); 
//     }
//     Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
//     if (!forOp.hasConstantBounds()) return false;  // 判断forop的上界和下界是否是已知量，这个可以直接手动去除循环结构
//     auto step = forOp.getStep();
//     auto ub = forOp.getConstantUpperBound();
//     auto lb = forOp.getConstantLowerBound();
//     auto times = (ub - lb) / step;
//     // if (times >= std::min<int64_t>(64, 4)) return false;
//     if (times >= 2) return false;
//     return true;
//     });
//     DUMP(module);

//     Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {   // 这种循环是可以添加pragma unroll的
//     if (!forOp.hasConstantBounds()) return false;
//     auto step = forOp.getStep();
//     auto ub = forOp.getConstantUpperBound();
//     auto lb = forOp.getConstantLowerBound();
//     auto times = (ub - lb) / step;
//     if (times > 64) return false;
//     return true;
//     });
//     DUMP(module);
//   }
// }

void BinaryOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto binary : binarys) {
    auto loops = binaryLoops[binary];
    auto buffers = binaryBuffers[binary];
    auto A = buffers.A, B = buffers.B, C = buffers.C;

    auto extras = getCreateAffineMapArgs(loops);
    auto new_loops = Rewriter::combineToTowDim(loops);
    extras.push_back(new_loops[1].getUpperBoundMap().getSingleConstantResult());
    auto dimY = new_loops[0].getUpperBoundMap().getSingleConstantResult();
    auto dimX = new_loops[1].getUpperBoundMap().getSingleConstantResult();

    DUMP(module);
    // 循环切块大小
    auto split_out_loops = Rewriter::split(new_loops[0], 3, {binaryConfig["THREAD_SIZE_M"], binaryConfig["BLOCK_SIZE_M"]});  // 第一个是一个thread计算的维度，第二个是一个block计算的多大的维度
    auto split_in_loops = Rewriter::split(new_loops[1], 3, {binaryConfig["THREAD_SIZE_N"], binaryConfig["BLOCK_SIZE_N"]});   // 
    DUMP(module);

    auto out_outer = split_out_loops[0], out_mider = split_out_loops[1], out_inner = split_out_loops[2];
    auto in_outer = split_in_loops[0], in_mider = split_in_loops[1], in_inner = split_in_loops[2];
    Rewriter::reorder({out_outer, in_outer, out_mider, in_mider, out_inner, in_inner});
    DUMP(module);

    auto gridLevel = Rewriter::parallel({out_outer, in_outer});
    auto blockLevel = Rewriter::parallel({out_mider, in_mider});
    DUMP(module);

    // auto *op = gridLevel->getParentOp();
    // auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);
    // funcOp->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));

    auto blockElemIdx = Rewriter::getElementIdx(gridLevel);
    auto ThreadElemIdx = Rewriter::getElementIdx(blockLevel);
    
    if (dimX % binaryConfig["THREAD_SIZE_N"] || dimY % binaryConfig["THREAD_SIZE_M"]) {
      std::vector<int> range{dimY, dimX, binaryConfig["THREAD_SIZE_M"], binaryConfig["THREAD_SIZE_N"]};
      llvm::SmallVector<mlir::Value> operands{blockElemIdx[0], blockElemIdx[1], ThreadElemIdx[0], ThreadElemIdx[1]};  // by bx ty tx
      auto ifop = Rewriter::irregularMat(out_inner, range, operands);
      DUMP(module);
    }
    Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
    if (!forOp.hasConstantBounds()) return false;  // 判断forop的上界和下界是否是已知量，这个可以直接手动去除循环结构
    auto step = forOp.getStep().getLimitedValue();
    auto ub = forOp.getConstantUpperBound();
    auto lb = forOp.getConstantLowerBound();
    auto times = (ub - lb) / step;
    // if (times >= std::min<int64_t>(64, 4)) return false;
    if (times >= 2) return false;
    return true;
    });
    DUMP(module);

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {   // 这种循环是可以添加pragma unroll的
    if (!forOp.hasConstantBounds()) return false;
    auto step = forOp.getStep().getLimitedValue();
    auto ub = forOp.getConstantUpperBound();
    auto lb = forOp.getConstantLowerBound();
    auto times = (ub - lb) / step;
    if (times > 64) return false;
    return true;
    });
    DUMP(module);
  }
}
/*--------------------------------------------------------------------*/

/*-----------------------------elementwise----------------------------*/
bool ElementWiseOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& elementWiseFuncs = Analyzer::collectFunctions(module, "Elementwise");
  bool res = elementWiseFuncs.size() != 0 ? true : false;

  for (auto& elementWiseFunc : elementWiseFuncs) {
    if (elementWises.count(elementWiseFunc) != 0 || elementWiseLoops.count(elementWiseFunc) != 0
      || elementWiseBuffers.count(elementWiseFunc) != 0) {
      llvm::errs() << "Duplicated Elementwise in module\n";
    }
    elementWises.insert(elementWiseFunc);
    auto&& loops = Analyzer::collectFuncLoops(elementWiseFunc);
    elementWiseLoops[elementWiseFunc] = std::move(loops);
    auto funcArgs = elementWiseFunc.front().getArguments();

    MemoryBuffer buf;
    buf.input = funcArgs[0];
    auto &block = elementWiseFunc.front();
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    buf.output = returnOp.getOperand(0);
    elementWiseBuffers[elementWiseFunc] = buf;
  }
  return res;
}

mlir::AffineMap ElementWiseOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto width = 4;  // float4 type width

  if (mapIdentifier == "VectorLoadOrStore") {
    auto oneDimExpr_y = dim0 + dim1 + dim2;
    auto oneDimExpr_x = dim3 + dim4 + dim5 * width;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    if (extras[0] == 2) {
      exprs.push_back(oneDimExpr_y);
      exprs.push_back(oneDimExpr_x);
    } else {
      auto oneDimExpr = oneDimExpr_y * extras.back() + oneDimExpr_x;  // 这个是多维映射到一维的index
      for (int i=0; i<extras[0]; i++) {  // [old_loops_len, 5120, 256, 256, 80]
        if (i == 0) {
          exprs.push_back(oneDimExpr.floorDiv(extras[i+1]));
        } else if (i != extras[0] - 1) {
          auto tmpExpr = oneDimExpr % extras[i];
          exprs.push_back(tmpExpr.floorDiv(extras[i+1]));
        } else {
          exprs.push_back(oneDimExpr % extras[i+1]);
        }
      }
    }
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "NoVectorLoadOrStore") {
    auto oneDimExpr_y = dim0 + dim1 + dim2;
    auto oneDimExpr_x = dim3 + dim4 + dim5;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    if (extras[0] == 2) {
      exprs.push_back(oneDimExpr_y);
      exprs.push_back(oneDimExpr_x);
    } else {
      auto oneDimExpr = oneDimExpr_y * extras.back() + oneDimExpr_x;  // 这个是多维映射到一维的index
      for (int i=0; i<extras[0]; i++) {  // [old_loops_len, 5120, 256, 256, 80]
        if (i == 0) {
          exprs.push_back(oneDimExpr.floorDiv(extras[i+1]));
        } else if (i != extras[0] - 1) {
          auto tmpExpr = oneDimExpr % extras[i];
          exprs.push_back(tmpExpr.floorDiv(extras[i+1]));
        } else {
          exprs.push_back(oneDimExpr % extras[i+1]);
        }
      }
    }
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "PointLoadOrStore") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void ElementWiseOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto elementwise : elementWises) {
    auto loops = elementWiseLoops[elementwise];
    auto buffer = elementWiseBuffers[elementwise];
    auto input = buffer.input; auto output = buffer.output;
    auto extras = getCreateAffineMapArgs(loops);
    auto new_loops = Rewriter::combineToTowDim(loops);
    extras.push_back(new_loops[1].getUpperBoundMap().getSingleConstantResult());
    auto dimY = new_loops[0].getUpperBoundMap().getSingleConstantResult();
    auto dimX = new_loops[1].getUpperBoundMap().getSingleConstantResult();

    DUMP(module);
    // 循环切块大小
    auto split_out_loops = Rewriter::split(new_loops[0], 3, {elementWiseConfig["THREAD_SIZE_M"], elementWiseConfig["BLOCK_SIZE_M"]});
    auto split_in_loops = Rewriter::split(new_loops[1], 3, {elementWiseConfig["THREAD_SIZE_M"], elementWiseConfig["BLOCK_SIZE_M"]});
    DUMP(module);

    auto out_outer = split_out_loops[0], out_mider = split_out_loops[1], out_inner = split_out_loops[2];
    auto in_outer = split_in_loops[0], in_mider = split_in_loops[1], in_inner = split_in_loops[2];
    Rewriter::reorder({out_outer, in_outer, out_mider, in_mider, out_inner, in_inner});
    DUMP(module);

    auto gridLevel = Rewriter::parallel({out_outer, in_outer});
    auto blockLevel = Rewriter::parallel({out_mider, in_mider});
    DUMP(module);

    auto blockElemIdx = Rewriter::getElementIdx(gridLevel);
    auto ThreadElemIdx = Rewriter::getElementIdx(blockLevel);

    in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantOp cst) {
      Rewriter::schedule(cst, blockLevel, Position::begin);
    });

    if (dimX % elementWiseConfig["THREAD_SIZE_N"] || dimY % elementWiseConfig["THREAD_SIZE_M"]) {
      std::vector<int> range{dimY, dimX, elementWiseConfig["THREAD_SIZE_M"], elementWiseConfig["THREAD_SIZE_N"]};
      llvm::SmallVector<mlir::Value> operands{blockElemIdx[0], blockElemIdx[1], ThreadElemIdx[0], ThreadElemIdx[1]};  // by bx ty tx
      auto ifop = Rewriter::irregularMat(out_inner, range, operands);
      DUMP(module);
    } else {
      auto input_type = input.getType();
      auto element = input_type.dyn_cast<mlir::MemRefType>().getElementType();

      auto loadOrStoreMap = getAffineMap("VectorLoadOrStore", builder, extras);
      auto pointLoadOrStore = getAffineMap("PointLoadOrStore", builder);
      llvm::SmallVector<mlir::Value> operands({blockElemIdx[0], ThreadElemIdx[0], out_inner.getInductionVar(), blockElemIdx[1], ThreadElemIdx[1]});

      if (input != output) {
        auto output_type = output.getType();
        auto element_ = output_type.dyn_cast<mlir::MemRefType>().getElementType();
        auto noVectorLoadOrStore = getAffineMap("NoVectorLoadOrStore", builder, extras);
        auto frag = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {elementWiseConfig["THREAD_SIZE_N"]}, element);  // 计算input -> reg
        auto frag_ = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {elementWiseConfig["THREAD_SIZE_N"]}, element_);  // 计算input -> reg
        if (toStr(element) == "float32") {
          Rewriter::read(input, frag, loadOrStoreMap, operands, elementWiseConfig["VECTORIZE_WIDTH"], out_inner, Position::begin);
          Rewriter::write(frag_, output, noVectorLoadOrStore, operands, in_inner, Position::after);
        } else if (toStr(element_) == "float32"){
          Rewriter::read(input, frag, noVectorLoadOrStore, operands, out_inner, Position::begin);
          Rewriter::write(frag_, output, loadOrStoreMap, operands, elementWiseConfig["VECTORIZE_WIDTH"], in_inner, Position::after);
        }
        Rewriter::cache_read(in_inner, input, frag, pointLoadOrStore, {in_inner.getInductionVar()});
        Rewriter::cache_write(in_inner, output, frag_, pointLoadOrStore, {in_inner.getInductionVar()});
      } else {
        auto frag = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {elementWiseConfig["THREAD_SIZE_N"]}, element);  // 计算input -> reg
        Rewriter::read(input, frag, loadOrStoreMap, operands, elementWiseConfig["VECTORIZE_WIDTH"], out_inner, Position::begin);
        Rewriter::write(frag, input, loadOrStoreMap, operands, elementWiseConfig["VECTORIZE_WIDTH"], in_inner, Position::after);
        Rewriter::cache_read(in_inner, input, frag, pointLoadOrStore, {in_inner.getInductionVar()});
        Rewriter::cache_write(in_inner, input, frag, pointLoadOrStore, {in_inner.getInductionVar()});
      }

      DUMP(module);
    }

    Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
    if (!forOp.hasConstantBounds()) return false;  // 判断forop的上界和下界是否是已知量，这个可以直接手动去除循环结构
    auto step = forOp.getStep().getLimitedValue();
    auto ub = forOp.getConstantUpperBound();
    auto lb = forOp.getConstantLowerBound();
    auto times = (ub - lb) / step;
    // if (times >= std::min<int64_t>(64, 4)) return false;
    if (times >= 2) return false;
    return true;
    });
    DUMP(module);

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {   // 这种循环是可以添加pragma unroll的
    if (!forOp.hasConstantBounds()) return false;
    auto step = forOp.getStep().getLimitedValue();
    auto ub = forOp.getConstantUpperBound();
    auto lb = forOp.getConstantLowerBound();
    auto times = (ub - lb) / step;
    if (times > 64) return false;
    return true;
    });
    DUMP(module);
  }
}
/*--------------------------------------------------------------------*/

/*----------------------------layernorm-------------------------------*/
bool LayerNormOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& layerNormFuncs = Analyzer::collectFunctions(module, "LayerNorm");
  bool res = layerNormFuncs.size() != 0 ? true : false;

  for (auto& layerNormFunc : layerNormFuncs) {
    if (layerNorms.count(layerNormFunc) != 0 || layerNormLoops.count(layerNormFunc) != 0
      || layerNormBuffers.count(layerNormFunc) != 0) {
      llvm::errs() << "Duplicated layerNorm in module\n";
    }
    layerNorms.insert(layerNormFunc);
    auto&& loops = Analyzer::collectFuncLoops(layerNormFunc);

    int hasIterLoopNum = 0;
    for (auto loop : loops) if (loop.getNumIterOperands()) hasIterLoopNum++;
    int innerLoopItemNum = hasIterLoopNum / 2;
    int outerLoopNum = loops.size() - innerLoopItemNum * 3;
    std::vector<mlir::affine::AffineForOp> outerLoops(loops.begin(), loops.begin()+outerLoopNum);
    layerNormLoops[layerNormFunc].push_back(outerLoops);
    // for (auto loop : loops) {llvm::outs() << loop << "\n";}
    for (int i=0; i<3; i++) {
      std::vector<mlir::affine::AffineForOp> innerLoops(loops.begin()+outerLoopNum+i*innerLoopItemNum, loops.begin()+outerLoopNum+(i+1)*innerLoopItemNum);
      layerNormLoops[layerNormFunc].push_back(innerLoops);
    }
    // layerNormLoops[layerNormFunc] = std::move(loops);
    auto funcArgs = layerNormFunc.front().getArguments();
    MemoryBuffer buf;
    buf.input = funcArgs[0];
    buf.scale = funcArgs[1];
    buf.bias = funcArgs[2];
    auto &block = layerNormFunc.front();
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    buf.output = returnOp.getOperand(0);
    layerNormBuffers[layerNormFunc] = buf;
  }
  return res;
}

mlir::AffineMap LayerNormOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto width = 4;  // float4 type width

  if (mapIdentifier == "VectorLoad") {
    auto oneDimExpr = dim1 + dim2 + dim3 * width;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    if (extras[0] == 1) {
      exprs.push_back(oneDimExpr);
    } else {
      for (int i=0; i<extras[0]; i++) {
        if (i == 0) {
          exprs.push_back(oneDimExpr.floorDiv(extras[i+1]));
        } else if (i != extras[0] - 1) {
          auto tmpExpr = oneDimExpr % extras[i];
          exprs.push_back(tmpExpr.floorDiv(extras[i+1]));
        } else {
          exprs.push_back(oneDimExpr % extras[i+1]);
        }
      }
    }
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "VectorStore") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0 + dim1 * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "PointLoadOrStore") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0 + dim1);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

mlir::affine::AffineParallelOp LayerNormOptimizer::combineParallel(std::vector<mlir::affine::AffineParallelOp> pals) {
  std::map<mlir::affine::AffineParallelOp, std::vector<mlir::Operation*>> backOps; 
  std::vector<mlir::affine::AffineForOp> innerLoops;
  
  for (auto pal : pals) {  // 收集pal里面的op
    auto temp = pal->getNextNode();
    while (temp) {
      if (mlir::dyn_cast<mlir::affine::AffineParallelOp>(temp)) break;
      if (mlir::dyn_cast<mlir::func::ReturnOp>(temp)) break;
      backOps[pal].push_back(temp);
      temp = temp->getNextNode();
    }
    auto& loopOps = pal.getBody()->getOperations();
    for (auto &op : loopOps) {
      if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(&op)){
        innerLoops.push_back(loop);
        break;
      }
    }
  }

  std::vector<mlir::affine::AffineIfOp> ifOps;
  auto dim = mlir::getAffineDimExpr(0, pals[0].getContext());
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({dim}), llvm::ArrayRef<bool>({true}));
  auto palIdx = Rewriter::getParallelIdx(pals[0]);
  std::reverse(innerLoops.begin(), innerLoops.end());
  for (int i=0; i<innerLoops.size(); i++) {
    innerLoops[i]->moveAfter(&(pals[0].getBody()->getOperations().front()));
    if (i == 0 || i == 1) {
      mlir::OpBuilder builder(innerLoops[i]);
      auto ifOp = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), set, mlir::ValueRange(palIdx), false);
      ifOps.push_back(ifOp);
    }
  }

  std::reverse(backOps[pals[0]].begin(), backOps[pals[0]].end());
  for (auto op : backOps[pals[0]]) {
    op->moveBefore(&(ifOps[1].getBody()->getOperations().front()));
  }
  std::reverse(backOps[pals[1]].begin(), backOps[pals[1]].end());
  for (auto op : backOps[pals[1]]) {
    op->moveBefore(&(ifOps[0].getBody()->getOperations().front()));
  }

  auto mainIdx = Rewriter::getElementIdx(pals[0]);   // applyop value
  std::vector<mlir::affine::AffineParallelOp> tempPals(pals.begin()+1, pals.end());
  for (auto pal : tempPals) {
    auto idx = Rewriter::getElementIdx(pal);
    auto users = idx[0].getUsers();
    for (auto user : users) {
      idx[0].replaceAllUsesWith(mainIdx[0]);
    }
  }
  pals[1].erase();
  pals[2].erase();
  return pals[0];
}

void replaceOperand(mlir::Operation* op, mlir::Value src, mlir::Value dst) {
  auto oldOperands = op->getOperands();
  llvm::SmallVector<mlir::Value> operands;
  for (auto operand : oldOperands) {
    if (operand == src) {
      operands.push_back(dst);
    } else {
      operands.push_back(operand);
    }
  }
  op->setOperands(operands);
}

std::vector<mlir::affine::AffineForOp> LayerNormOptimizer::read(mlir::affine::AffineForOp forOp, std::vector<mlir::Value> buffers) {
  /*找到forop下,loadOp所操作的memref等于shared[1]的loadOp，并在forop上单独创建for，
  取shared[1]存到shared[0]中，最后将forop中的操作shared[1]的loadOp的memref全部替换成shared[2]然后向量化这个新的for循环*/
  auto lowerBound = forOp.getLowerBoundMap();
  auto upperBound = forOp.getUpperBoundMap();
  int step = forOp.getStep().getLimitedValue();
  int64_t lb = lowerBound.getSingleConstantResult();
  int64_t ub = upperBound.getSingleConstantResult();

  std::vector<mlir::affine::AffineLoadOp> targetLoadOps;
  auto &ops = forOp.getBody()->getOperations();
  for (auto& op : ops) {
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(&op)){
      auto mem = loadOp.getMemRef();
      if (mem == buffers[1]) {  // 收集从 buffers[1] 进行load的loadop，比如为input,scale bias
        targetLoadOps.push_back(loadOp);
      } 
    }
  }

  std::vector<mlir::affine::AffineForOp> loops;
  auto loopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
  };
  mlir::OpBuilder builder(forOp);  // 创建在它的前面
  auto oneForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);

  int64_t totalDim = 1;
  auto shape = buffers[1].getType().dyn_cast<mlir::MemRefType>().getShape();
  for (auto shape_ : shape) {totalDim *= shape_;}

  auto map = targetLoadOps[0].getAffineMap();
  auto indices = targetLoadOps[0].getIndices();
  builder.setInsertionPointToStart(oneForOp.getBody());
  llvm::SmallVector<mlir::Value> loadOperand;
  for (auto index : indices) {
    if (forOp.getInductionVar() == index) loadOperand.push_back(oneForOp.getInductionVar());
    else {
      auto indexOp = index.getDefiningOp();
      if (mlir::dyn_cast<mlir::arith::ConstantIndexOp>(indexOp)) Rewriter::schedule(indexOp, forOp->getParentOp(), Position::begin);
      loadOperand.push_back(index);
    }
  }
  // for (auto op__ : loadOperand) {op__.print(llvm::outs()); llvm::outs() << "\n";}llvm::outs() << "\n";
  auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffers[1], map, loadOperand);
  auto storeMap = getAffineMap("PointLoadOrStore", builder);

  auto op = forOp->getParentOp()->getParentOp();
  auto pal = mlir::dyn_cast<mlir::affine::AffineParallelOp>(op);
  auto threadIdx = Rewriter::getElementIdx(pal);
  llvm::SmallVector<mlir::Value> storeOperand({threadIdx[0], oneForOp.getInductionVar()});

  auto storeOp = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), loadOp.getResult(), buffers[0], storeMap, storeOperand);
  mlir::affine::AffineForOp firstLoop;
  if (totalDim >= 4) firstLoop = Rewriter::vectorize(oneForOp, layerNormConfig["VECTORIZE_WIDTH"]);
  else firstLoop = oneForOp;
  loops.push_back(firstLoop);

  for (auto targetLoadOp : targetLoadOps) {  // ceche read
    builder.setInsertionPointAfter(targetLoadOp);
    llvm::SmallVector<mlir::Value> loadOperand_({storeOperand[0], forOp.getInductionVar()});  // (threadidx, foriter)
    auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffers[0], storeMap, loadOperand_);
    targetLoadOp.getResult().replaceAllUsesWith(loadOp.getResult());
    targetLoadOp.erase();
  }
  loops.push_back(forOp);
  return loops;
}

mlir::affine::AffineForOp LayerNormOptimizer::extractOpsFromLoop(mlir::affine::AffineForOp forOp, std::vector<mlir::Value> buffers) {
  auto lowerBound = forOp.getLowerBoundMap();
  auto upperBound = forOp.getUpperBoundMap();
  int step = forOp.getStep().getLimitedValue();
  int64_t lb = lowerBound.getSingleConstantResult();
  int64_t ub = upperBound.getSingleConstantResult();

  // 收集
  std::vector<mlir::affine::AffineLoadOp> targetLoadOps;
  std::set<mlir::Operation*> targetArithOps;
  mlir::affine::AffineLoadOp tempArrayLoadOp;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
    auto mem = loadOp.getMemRef();
    auto it = std::find(buffers.begin(), buffers.end(), mem);
    if (it != buffers.end()) {
      targetLoadOps.push_back(loadOp);
      if (mem == buffers[0]) tempArrayLoadOp = loadOp;
      auto users = loadOp.getResult().getUsers();
      for (auto *user : users) {
        targetArithOps.insert(user);
      }
    }
  });

  // 找到最后的一个arith操作
  mlir::Operation* resultOp;
  for (auto *arithOp : targetArithOps) {
    auto users = arithOp->getResult(0).getUsers();
    for (auto *user : users) {
      auto it = std::find(targetArithOps.begin(), targetArithOps.end(), user);
      if (it != targetArithOps.end()) break;
      else {resultOp = arithOp; break;}
    }
  }

  auto map = tempArrayLoadOp.getAffineMap();
  auto operand = tempArrayLoadOp.getIndices();

  // 在resultOp下面添加一个loadop
  mlir::OpBuilder builder(forOp);
  builder.setInsertionPointAfter(resultOp);
  auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffers[0], map, operand);
  resultOp->getResult(0).replaceAllUsesWith(loadOp.getResult());

  // 创建新的for
  auto loopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
  };
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);

  // 转移到新的for
  for (auto loadOp : targetLoadOps) {
    loadOp->moveBefore(&(newForOp.getBody()->getOperations().back()));
    replaceOperand(loadOp, forOp.getInductionVar(), newForOp.getInductionVar());
  }
  for (auto arithOp : targetArithOps) {
    arithOp->moveBefore(&(newForOp.getBody()->getOperations().back()));
  }

  // 将转移过来的arith操作的最终结果存起来
  builder.setInsertionPoint(&(newForOp.getBody()->getOperations().back()));
  auto storeOp = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), resultOp->getResult(0), buffers[0], map, operand);

  return newForOp;
}

mlir::affine::AffineForOp LayerNormOptimizer::write(mlir::affine::AffineForOp forOp, std::vector<mlir::Value> buffers) {
  auto lowerBound = forOp.getLowerBoundMap();
  auto upperBound = forOp.getUpperBoundMap();
  int step = forOp.getStep().getLimitedValue();
  int64_t lb = lowerBound.getSingleConstantResult();
  int64_t ub = upperBound.getSingleConstantResult();

  mlir::affine::AffineLoadOp tempArrayLoadOp;
  mlir::affine::AffineStoreOp gloabStoreOp;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
    auto mem = loadOp.getMemRef();
    if (mem == buffers[0]) {
      tempArrayLoadOp = loadOp;
      auto users = loadOp.getResult().getUsers();
      for (auto *user : users) {
        if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
          auto mem_ = storeOp.getMemRef();
          if (mem_ = buffers[1]) gloabStoreOp = storeOp;
        }
      }
    }
  });

  // 在load后面创建一个相同的load
  auto map = tempArrayLoadOp.getAffineMap();
  auto operand = tempArrayLoadOp.getIndices();
  mlir::OpBuilder builder(forOp);
  builder.setInsertionPointAfter(tempArrayLoadOp);
  auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffers[0], map, operand);
  replaceOperand(gloabStoreOp, tempArrayLoadOp.getResult(), loadOp.getResult());

  // 创建新的for
  auto loopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
  };
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);

  // 转移到新的for
  loadOp->moveBefore(&(newForOp.getBody()->getOperations().back()));
  replaceOperand(loadOp, forOp.getInductionVar(), newForOp.getInductionVar());
  gloabStoreOp->moveBefore(&(newForOp.getBody()->getOperations().back()));
  replaceOperand(gloabStoreOp, forOp.getInductionVar(), newForOp.getInductionVar());

  return newForOp;
  // std::vector<mlir::affine::AffineLoadOp> targetLoadOps;
  // mlir::affine::AffineStoreOp targetStoreOp;
  // mlir::arith::MulFOp mulOp;
  // auto &ops = forOp.getBody()->getOperations();
  // for (auto& op : ops) {
  //   if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(&op)){
  //     auto mem = loadOp.getMemRef();
  //     if (mem == shareds[0] || mem == shareds[1]) {
  //       targetLoadOps.push_back(loadOp);
  //     } 
  //   } else if (auto storeOp_ = mlir::dyn_cast<mlir::affine::AffineStoreOp>(&op)) {
  //     auto mem = storeOp_.getMemRef();
  //     if (mem == shareds[2]) {
  //       targetStoreOp = storeOp_;
  //     } 
  //   } else if (auto ml = mlir::dyn_cast<mlir::arith::MulFOp>(&op)) {
  //     mulOp = ml;
  //   }
  // }

  // auto replace = [&](mlir::Operation* op, mlir::Value src, mlir::Value dst) {
  //   auto oldOperands = op->getOperands();
  //   llvm::SmallVector<mlir::Value> operands;
  //   for (auto operand : oldOperands) {
  //     if (operand == src) {
  //       operands.push_back(dst);
  //     } else {
  //       operands.push_back(operand);
  //     }
  //   }
  //   op->setOperands(operands);
  // };
  // auto countOpNum = [&](mlir::affine::AffineForOp loop) {
  //   int opNum = 0;
  //   auto& ops = loop.getBody()->getOperations();
  //   for (auto& op : ops) {opNum++;}
  //   return opNum;
  // };

  // std::vector<mlir::affine::AffineForOp> loops;
  // auto loopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
  //   mlir::OpBuilder::InsertionGuard nestedGuard(builder);
  //   builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
  // };
  // mlir::OpBuilder builder(forOp);
  // auto oneForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);

  // mlir::Value opResult;
  // for (auto loadOp : targetLoadOps) {
  //   loadOp->moveBefore(&(oneForOp.getBody()->getOperations().back()));
  //   replace(loadOp, forOp.getInductionVar(), oneForOp.getInductionVar());
  // }
  // auto users = targetLoadOps[0].getResult().getUsers();
  // for (auto user : users) {
  //   if(auto subOp = mlir::dyn_cast<mlir::arith::SubFOp>(user))
  //     opResult = subOp.getResult();
  //   else if (auto divOp = mlir::dyn_cast<mlir::arith::DivFOp>(user))
  //     opResult = divOp.getResult();
  //   user->moveBefore(&(oneForOp.getBody()->getOperations().back()));
  // }

  // auto map = targetLoadOps[0].getAffineMap();
  // auto operand = targetLoadOps[0].getIndices();

  // auto &op = oneForOp.getBody()->getOperations().back();
  // auto yeildOp = mlir::dyn_cast<mlir::affine::AffineYieldOp>(&op);
  // builder.setInsertionPointAfter(yeildOp);
  // auto storeOp = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), opResult, shareds[0], map, operand);
  // storeOp->moveBefore(yeildOp);
  // loops.push_back(oneForOp);

  // if (countOpNum(forOp) == 2) {  // 只能分两个循环
  //   builder.setInsertionPointToStart(forOp.getBody());
  //   llvm::SmallVector<mlir::Value> operand1({operand[0], forOp.getInductionVar()});
  //   auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), shareds[0], map, operand1);
  //   replace(targetStoreOp, opResult, loadOp.getResult());
  //   loops.push_back(forOp);
  // } else {
  //   builder.setInsertionPointAfter(oneForOp);
  //   auto twoForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);
  //   builder.setInsertionPointToStart(twoForOp.getBody());
  //   llvm::SmallVector<mlir::Value> operand1({operand[0], twoForOp.getInductionVar()});
  //   auto loadOp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), shareds[0], map, operand1);
  //   targetStoreOp->moveBefore(&(twoForOp.getBody()->getOperations().back()));
  //   replace(targetStoreOp, forOp.getInductionVar(), twoForOp.getInductionVar());
  //   replace(targetStoreOp, opResult, loadOp1.getResult());
  //   loops.push_back(twoForOp);

  //   builder.setInsertionPointAfter(twoForOp);
  //   auto threeForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);
  //   builder.setInsertionPointToStart(threeForOp.getBody());
  //   llvm::SmallVector<mlir::Value> operand2({operand[0], threeForOp.getInductionVar()});
  //   auto loadOp2 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), shareds[0], map, operand2);
  //   mulOp->moveAfter(loadOp2);
  //   replace(mulOp, opResult, loadOp2.getResult());
  //   opResult = mulOp.getResult();
  //   auto storeOp2 = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), opResult, shareds[0], map, operand2);
  //   loops.push_back(threeForOp);

  //   builder.setInsertionPointToStart(forOp.getBody());
  //   llvm::SmallVector<mlir::Value> operand3({operand[0], forOp.getInductionVar()});
  //   auto loadOp_ = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), shareds[0], map, operand3);
  //   auto &last_ops = forOp.getBody()->getOperations();
  //   for (auto& op : last_ops) {
  //     replace(&op, opResult, loadOp_.getResult());
  //   }
  //   loops.push_back(forOp);
  // }

  // return loops;
}

std::vector<mlir::Operation*> LayerNormOptimizer::reduceUnrollOptimize(mlir::affine::AffineForOp forOp, mlir::affine::AffineParallelOp pal) {
  /*循环展开，没有bank confilct，最后一个warp规约使用__shlf_down_sync*/
  mlir::Value buffer, resultBuf;
  std::vector<mlir::Operation*> operaitons;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
    auto temp = loadOp.getIndices();
    if (temp.size() > 1) {
      buffer = loadOp.getMemRef();
    } else {
      resultBuf = loadOp.getMemRef();
    }
  });
  int warpWidth = 32;
  auto threadidxs = Rewriter::getParallelIdx(pal);  // threadIdx.x
  auto threadNum = pal.getUpperBoundMap(0).getSingleConstantResult();  // thread num
  auto perThread = forOp.getUpperBoundMap().getSingleConstantResult();   // each thread evaluates four element
  auto mainForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(forOp->getParentOp());  // father forOp
  auto backIfOp = mlir::dyn_cast<mlir::affine::AffineIfOp>(mainForOp->getNextNode());  // fellow ifOp

  mlir::OpBuilder builder(forOp);
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(&(pal.getBody()->getOperations().front()));  // block level applyOp back

  // 在blocklevel创建loacl mem 存2048的reduceMean结果，再创建一个存储32 shfl_down的结果
  auto type = resultBuf.getType();
  auto resultType = type.dyn_cast<mlir::MemRefType>();
  auto bufferType = mlir::MemRefType::get({1}, resultType.getElementType(), {}, static_cast<int>(MemorySpace::local));
  auto resultAllocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), bufferType);

  mlir::Value cstIndex, cstZero;
  mlir::affine::AffineStoreOp removeStoreOp;
  auto prevOp = pal->getPrevNode();
  while(prevOp) {
    if (auto indexOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(prevOp)) {
      if (indexOp.value() == 0) cstIndex = indexOp.getResult();
    } else if (auto flaotOp = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(prevOp)) {
      if (flaotOp.value().convertToFloat() == 0.0f) cstZero = flaotOp.getResult();
    } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(prevOp)) {
      if (storeOp.getMemRef() == resultBuf) removeStoreOp = storeOp;
    }
    prevOp = prevOp->getPrevNode();
  }
  removeStoreOp.erase();
  if (!cstIndex) {
    auto cstOp1 = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
    cstIndex = cstOp1.getResult();
  }
  if (!cstZero) { 
    auto cstOp2 = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(resultType.getElementType(), 0));
    cstZero = cstOp2.getResult();
  }
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), cstZero, resultAllocOp, mlir::ValueRange({cstIndex}));
  builder.restoreInsertionPoint(ip);

  //-------------------------------x>512
  std::vector<mlir::Value> loadOpResults;
  for (int i=0; i<perThread; i++) {
    auto expr = builder.getAffineDimExpr(0) + i * threadNum;
    auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext());
    auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffer, map, mlir::ValueRange({threadidxs[0]}));
    loadOpResults.push_back(loadOp.getResult());
  }
  while (loadOpResults.size() > 1) {
    auto addOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadOpResults[0], loadOpResults[1]);
    loadOpResults.erase(loadOpResults.begin(), loadOpResults.begin()+2);
    loadOpResults.push_back(addOp.getResult());
  }
  auto oneStoreOp = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), loadOpResults[0], buffer, mlir::ValueRange({threadidxs[0]}));
  operaitons.push_back(oneStoreOp);

  //-------------------------------32<x<512
  for (int i=threadNum/2; i>32; i>>=1) {
    auto dim = i-1 - builder.getAffineDimExpr(0);
    auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({dim}), llvm::ArrayRef<bool>({false}));
    auto ifOp = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), set, mlir::ValueRange(threadidxs[0]), false);
    auto ip1 = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(ifOp.getBody());
    auto expr = builder.getAffineDimExpr(0) + i;
    auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext());
    auto loadOp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffer, mlir::ValueRange({threadidxs[0]}));
    auto loadOp2 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffer, map, mlir::ValueRange({threadidxs[0]}));
    auto addOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadOp1, loadOp2);
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), addOp.getResult(), buffer, mlir::ValueRange({threadidxs[0]}));
    operaitons.push_back(ifOp);
    builder.restoreInsertionPoint(ip1);
  }

  //----------------------------x<32
  auto dim = warpWidth - 1 - builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({dim}), llvm::ArrayRef<bool>({false}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), set, mlir::ValueRange(threadidxs[0]), false);

  builder.setInsertionPointToStart(ifOp.getBody());
  auto expr = builder.getAffineDimExpr(0) + warpWidth;
  auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext());
  auto loadOp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffer, mlir::ValueRange({threadidxs[0]}));
  auto loadOp2 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buffer, map, mlir::ValueRange({threadidxs[0]}));
  auto addOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), loadOp1, loadOp2);
  for (int i=warpWidth/2; i>0; i>>=1) {
    auto shflOp = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), addOp.getResult(), i, warpWidth, mlir::gpu::ShuffleMode::DOWN);
    addOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), shflOp.getResult(0), addOp.getResult());
  }


  //----------------------------x=0
  auto dim_ = builder.getAffineDimExpr(0);
  auto set_ = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({dim_}), llvm::ArrayRef<bool>({true}));
  auto ifOp_ = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), set_, mlir::ValueRange(threadidxs[0]), false);
  builder.setInsertionPointToStart(ifOp_.getBody());
  auto resultLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), resultAllocOp, mlir::ValueRange({cstIndex}));
  // auto sumloadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), sumAllocOp, mlir::ValueRange({cstIndex}));
  auto tempAddOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), resultLoadOp.getResult(), addOp.getResult());
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tempAddOp.getResult(), resultAllocOp, mlir::ValueRange({cstIndex}));
  forOp.erase();

  //--------------------------------
  backIfOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
    if (loadOp.getMemRef() == resultBuf) {
      builder.setInsertionPointAfter(loadOp);
      auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), resultAllocOp, mlir::ValueRange({cstIndex}));
      loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
      loadOp.erase();
    }
  });

  return operaitons;
}

void LayerNormOptimizer::elementWiseUnrollOptimize(mlir::affine::AffineForOp forOp, mlir::affine::AffineParallelOp pal) {
  /*将做elementwies的循环做没有bank confilct的循环展开*/
  std::vector<mlir::affine::AffineLoadOp> tempLoadOps;
  mlir::affine::AffineLoadOp cstLoadOp;
  std::vector<mlir::Operation*> arithOps;
  mlir::affine::AffineStoreOp tempStoreOp;
  auto threadidxs = Rewriter::getParallelIdx(pal);  // threadidx
  auto threadNum = pal.getUpperBoundMap(0).getSingleConstantResult();  // 512
  auto perThread = forOp.getUpperBoundMap().getSingleConstantResult();  // 4
  auto& ops = forOp.getBody()->getOperations();
  for (auto& op : ops) {
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(&op)) {
      if (loadOp.getIndices().size() > 1) tempLoadOps.push_back(loadOp);
      else cstLoadOp = loadOp;
    } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(&op)) {
      tempStoreOp = storeOp;
    } else if (!mlir::dyn_cast<mlir::affine::AffineYieldOp>(&op)){
      arithOps.push_back(&op);
    }
  }

  mlir::OpBuilder builder(forOp);
  if (cstLoadOp) cstLoadOp->moveBefore(forOp);

  for (int i=0; i<perThread; i++) {
    auto expr = builder.getAffineDimExpr(0) + i * threadNum;
    auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext());
    std::vector<mlir::affine::AffineLoadOp> newTempLoadOps;
    for (auto tempLoadOp : tempLoadOps) {
      auto mem = tempLoadOp.getMemRef();
      auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), mem, map, mlir::ValueRange({threadidxs[0]}));
      newTempLoadOps.push_back(loadOp);
    }
    mlir::Value result;
    if (arithOps.size() == 1 && cstLoadOp) {  // x - ave(x)
      auto subOp = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), newTempLoadOps[0].getResult(), cstLoadOp.getResult());
      result = subOp.getResult();
    } else if (arithOps.size() == 1 && !cstLoadOp) {  // [x - ave(x)]^2
      auto mulOp = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), newTempLoadOps[0].getResult(), newTempLoadOps[0].getResult());
      result = mulOp.getResult();
    } else if (arithOps.size() == 3 && cstLoadOp) {  // scale*result+bias
      auto divOp = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), newTempLoadOps[0].getResult(), cstLoadOp.getResult());
      auto mulOp = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), newTempLoadOps[1].getResult(), divOp.getResult());
      auto addOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), newTempLoadOps[2].getResult(), mulOp.getResult());
      result = addOp.getResult();
    }
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), result, newTempLoadOps[0].getMemRef(), map, mlir::ValueRange({threadidxs[0]}));
  }
  forOp.erase();
}

void LayerNormOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto layerNorm: layerNorms) {  // func
    auto loops = layerNormLoops[layerNorm];
    auto buffer = layerNormBuffers[layerNorm];
    auto input = buffer.input; auto output = buffer.output; 
    auto scale = buffer.scale; auto bias = buffer.bias;
    std::vector<mlir::affine::AffineForOp> outLoops = loops[0], sonLoops1 = loops[1], sonLoops2 = loops[2], sonLoops3 = loops[3];
    // std::vector<mlir::affine::AffineForOp> sonLoops1, sonLoops2, sonLoops3;
    // int sonLoopsNum = (loops.size() - 1) / 3;
    // int iter1 = 1; int iter2 = iter1 + sonLoopsNum; int iter3 = iter2 + sonLoopsNum;
    // for (int i=0; i<sonLoopsNum; i++) {
    //   sonLoops1.push_back(loops[iter1++]);
    //   sonLoops2.push_back(loops[iter2++]);
    //   sonLoops3.push_back(loops[iter3++]);
    // }
    // auto extras = getCreateAffineMapArgs(sonLoops1);
    auto outLoop = Rewriter::combineToOneDim(outLoops);
    auto sonLoop1 = Rewriter::combineToOneDim(sonLoops1);
    auto sonLoop2 = Rewriter::combineToOneDim(sonLoops2);
    auto sonLoop3 = Rewriter::combineToOneDim(sonLoops3);
    DUMP(module);

    auto iterBuffer1 = Rewriter::bufferizeLoopCarryVar(sonLoop1, outLoop.getBody());
    auto iterBuffer2 = Rewriter::bufferizeLoopCarryVar(sonLoop2, outLoop.getBody());
    DUMP(module);

    auto split_loops1 = Rewriter::split(sonLoop1, 3, {layerNormConfig["THREAD_SIZE"], layerNormConfig["BLOCK_SIZE"]});
    auto split_loops2 = Rewriter::split(sonLoop2, 3, {layerNormConfig["THREAD_SIZE"], layerNormConfig["BLOCK_SIZE"]});
    auto split_loops3 = Rewriter::split(sonLoop3, 3, {layerNormConfig["THREAD_SIZE"], layerNormConfig["BLOCK_SIZE"]});
    DUMP(module);

    // split_loops1[1]，split_loops2[1]，split_loops3[1] no exist
    Rewriter::swapLoops({{split_loops1[0], split_loops1[1]}, {split_loops2[0], split_loops2[1]}, {split_loops3[0], split_loops3[1]}});
    DUMP(module);
    
    auto gridLevel = Rewriter::parallel({outLoop});
    auto blockLevel1 = Rewriter::parallel({split_loops1[1]});
    auto blockLevel2 = Rewriter::parallel({split_loops2[1]});
    auto blockLevel3 = Rewriter::parallel({split_loops3[1]});
    DUMP(module);

    Rewriter::bufferizeOpResult(blockLevel2->getPrevNode(), iterBuffer1);  // 将计算mean的过程的结果存入到iterBuffer1
    Rewriter::bufferizeOpResult(blockLevel3->getPrevNode(), iterBuffer2);  // 将计算std的过程的结果存入到iterBuffer2
    DUMP(module);

    // blockLevel2, blockLevel3 no exist
    auto blockLevel = combineParallel({blockLevel1, blockLevel2, blockLevel3});
    Rewriter::barrier(split_loops2[0], Position::before);
    Rewriter::barrier(split_loops3[0], Position::before);
    DUMP(module);

    auto input_type = input.getType();
    auto element = input_type.dyn_cast<mlir::MemRefType>().getElementType();
    auto tempArray = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {layerNormConfig["BLOCK_SIZE"]}, element);
    auto tempScaleArray = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {layerNormConfig["BLOCK_SIZE"]}, element);
    auto tempBiasArray = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {layerNormConfig["BLOCK_SIZE"]}, element);
    auto frontLoops1 = read(split_loops1[2], {tempArray, input}); // 从input取数存到shared，且进行向量化
    auto frontLoops2 = read(split_loops2[2], {tempArray, input});
    auto frontLoops3 = read(split_loops3[2], {tempArray, output});

    auto vecScaleLoop = read(split_loops3[2], {tempScaleArray, scale});
    auto vecBiasLoop = read(split_loops3[2], {tempBiasArray, bias});
    DUMP(module);

    for (auto frontLoop: frontLoops1) {Rewriter::barrier(frontLoop, Position::after);}
    for (auto frontLoop: frontLoops2) {Rewriter::barrier(frontLoop, Position::after);}
    for (auto frontLoop: vecBiasLoop) {Rewriter::barrier(frontLoop, Position::after);}
    DUMP(module);

    auto fristLoop1 = extractOpsFromLoop(frontLoops2[1], {tempArray, iterBuffer1});
    auto midLoop = write(frontLoops2[1], {tempArray, output});
    auto lastLoop = extractOpsFromLoop(frontLoops2[1], {tempArray});
    auto fristLoop2 = extractOpsFromLoop(frontLoops3[1], {tempArray, tempScaleArray, tempBiasArray, iterBuffer2});
    DUMP(module);

    Rewriter::vectorize(midLoop, 4);
    Rewriter::vectorize(frontLoops3[1], 4);
    Rewriter::barrier(fristLoop1, Position::after);
    Rewriter::barrier(midLoop, Position::after);
    Rewriter::barrier(lastLoop, Position::after);
    Rewriter::barrier(fristLoop2, Position::after);
    DUMP(module);

    auto ops1 = reduceUnrollOptimize(frontLoops1[1], blockLevel);
    auto ops2 = reduceUnrollOptimize(frontLoops2[1], blockLevel);
    for (auto op: ops1) {
      auto bar = Rewriter::barrier(split_loops1[0], Position::after);
      Rewriter::schedule(bar, op, Position::after);
    }
    for (auto op: ops2) {
      auto bar = Rewriter::barrier(split_loops2[0], Position::after);
      Rewriter::schedule(bar, op, Position::after);
    }
    DUMP(module);

    elementWiseUnrollOptimize(fristLoop1, blockLevel);
    elementWiseUnrollOptimize(lastLoop, blockLevel);
    elementWiseUnrollOptimize(fristLoop2, blockLevel);
    DUMP(module);

    Rewriter::scheduleOpGridToBlock(gridLevel, blockLevel);

    Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
    if (!forOp.hasConstantBounds()) return false;
    auto step = forOp.getStep().getLimitedValue();
    auto ub = forOp.getConstantUpperBound();
    auto lb = forOp.getConstantLowerBound();
    auto times = (ub - lb) / step;
    // if (times >= std::min<int64_t>(64, 4)) return false;
    if (times >= 2) return false;
    return true;
    });
    DUMP(module);

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
    if (!forOp.hasConstantBounds()) return false;
    auto step = forOp.getStep().getLimitedValue();
    auto ub = forOp.getConstantUpperBound();
    auto lb = forOp.getConstantLowerBound();
    auto times = (ub - lb) / step;
    // if (times >= 64) return false;
    if (times >= 16) return false;
    return true;
    });
    DUMP(module);
    Rewriter::deleteExtraCstOp(blockLevel);
    DUMP(module);
  }
}
/*--------------------------------------------------------------------*/

/*-----------------------------gather----------------------------*/
bool GatherOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& gatherFuncs = Analyzer::collectFunctions(module, "Gather");
  bool res = gatherFuncs.size() != 0 ? true : false;

  for (auto& gatherFunc : gatherFuncs) {
    if (gathers.count(gatherFunc) != 0 || gatherLoops.count(gatherFunc) != 0
      || gatherBuffers.count(gatherFunc) != 0) {
      llvm::errs() << "Duplicated Gather in module\n";
    }
    gathers.insert(gatherFunc);
    auto&& loops = Analyzer::collectFuncLoops(gatherFunc);
    gatherLoops[gatherFunc] = std::move(loops);
    auto funcArgs = gatherFunc.front().getArguments();

    MemoryBuffer buf;
    buf.input = funcArgs[0];
    buf.indices = funcArgs[1];
    auto &block = gatherFunc.front();
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    buf.output = returnOp.getOperand(0);
    gatherBuffers[gatherFunc] = buf;
  }
  return res;
}

mlir::AffineMap GatherOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, const std::vector<int64_t> &extras) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto width = 4;  // float4 type width

  if (mapIdentifier == "VectorLoadOrStore") {
    auto oneDimExpr_y = dim0 + dim1 + dim2;
    auto oneDimExpr_x = dim3 + dim4 + dim5 * width;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    auto oneDimExpr = oneDimExpr_y * extras.back() + oneDimExpr_x;
    for (int i=0; i<extras[0]; i++) { 
      if (i == 0) {
        exprs.push_back(oneDimExpr.floorDiv(extras[i+1]));
      } else if (i != extras[0] - 1) {
        auto tmpExpr = oneDimExpr % extras[i];
        exprs.push_back(tmpExpr.floorDiv(extras[i+1]));
      } else {
        exprs.push_back(oneDimExpr % extras[i+1]);
      }
    }
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "PointLoadOrStore") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void GatherOptimizer::oneIndexLoad(mlir::affine::AffineForOp forOp, mlir::affine::AffineParallelOp pal) {
  mlir::affine::AffineLoadOp loadOp;
  mlir::arith::ConstantIndexOp cstOp;
  auto &ops = forOp.getBody()->getOperations();
  for (auto &op : ops) {
    if (auto cstOp_ = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(&op)) {
      cstOp = cstOp_;
      auto users = cstOp_.getResult().getUsers();
      if (auto loadOp_ = mlir::dyn_cast<mlir::affine::AffineLoadOp>(*users.begin())) {
        loadOp = loadOp_;
        break;
      }
    }
  }
  Rewriter::schedule(cstOp, pal, Position::begin);
  Rewriter::schedule(loadOp, cstOp, Position::after);
}

void GatherOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto gather : gathers) {
    auto loops = gatherLoops[gather];
    auto buffer = gatherBuffers[gather];
    auto input = buffer.input; auto indices = buffer.indices; auto output = buffer.output;
    auto extras = getCreateAffineMapArgs(loops);
    auto twoLoops = Rewriter::combineToTowDim(loops);
    extras.push_back(twoLoops[1].getUpperBoundMap().getSingleConstantResult());
    DUMP(module);

    auto split_out_loops = Rewriter::split(twoLoops[0], 3, {gatherConfig["THREAD_SIZE_M"], gatherConfig["BLOCK_SIZE_M"]});
    auto split_in_loops = Rewriter::split(twoLoops[1], 3, {gatherConfig["THREAD_SIZE_M"], gatherConfig["BLOCK_SIZE_M"]});
    DUMP(module);

    auto out_outer = split_out_loops[0], out_mider = split_out_loops[1], out_inner = split_out_loops[2];
    auto in_outer = split_in_loops[0], in_mider = split_in_loops[1], in_inner = split_in_loops[2];
    Rewriter::reorder({out_outer, in_outer, out_mider, in_mider, out_inner, in_inner});
    DUMP(module);

    auto gridLevel = Rewriter::parallel({out_outer, in_outer});
    auto blockLevel = Rewriter::parallel({out_mider, in_mider});
    DUMP(module);

    auto blockElemIdx = Rewriter::getElementIdx(gridLevel);
    auto ThreadElemIdx = Rewriter::getElementIdx(blockLevel);
    
    auto indicesType = indices.getType();
    auto type_ = indicesType.dyn_cast<mlir::MemRefType>();
    auto shape = type_.getShape();

    if (shape.size() == 1 && shape[0] == 1) {
      oneIndexLoad(in_inner, blockLevel);
      DUMP(module);
    } else {
      auto input_type = input.getType();
      auto element = input_type.dyn_cast<mlir::MemRefType>().getElementType();
      auto storeReg = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {gatherConfig["THREAD_SIZE_N"]}, element);  // 计算input -> reg
      auto writeMap = getAffineMap("VectorLoadOrStore", builder, extras);
      auto cacheWriteMap = getAffineMap("PointLoadOrStore", builder);

      llvm::SmallVector<mlir::Value> operands({blockElemIdx[0], ThreadElemIdx[0], out_inner.getInductionVar(), blockElemIdx[1], ThreadElemIdx[1]});
      Rewriter::write(storeReg, output, writeMap, operands, gatherConfig["VECTORIZE_WIDTH"], in_inner, Position::after);
      Rewriter::cache_write(in_inner, output, storeReg, cacheWriteMap, {in_inner.getInductionVar()});
    }

    // if (!indices) {  // 按常数取
    //   in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::arith::ConstantOp cstOp) {
    //     Rewriter::schedule(cstOp, blockLevel, Position::begin);
    //   });
    //   bool vecTag = false;
    //   in_inner.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp loadOp) {
    //     if (loadOp.getMemRef() == input) {
    //       auto maps = loadOp.getAffineMap();
    //       auto expr = maps.getResults().back();
    //       if (expr.isa<mlir::AffineBinaryOpExpr>()) vecTag = true;
    //     }
    //   });
    //   if (vecTag) Rewriter::vectorize(in_inner, 4);
    // } else {  // 按照索引indices取

    // }
  }
}
/*--------------------------------------------------------------------*/

void splitString(const std::string& input, char target, std::vector<std::string>& output) {
  std::string cur {""};
  int len = input.size();
  for (int i = 0; i < len; i ++) {
    if (input[i] == target) {
      output.push_back(cur);
      cur = std::string("");
      continue;
    }
    cur += input[i];
  }
  output.push_back(cur);
}

void identifyBatchMatmul(const std::string& name, BatchMatmulDescriptor& matmul) {
  auto len = name.size();
  matmul.transA = name[len - 2] == 'N' ? false : true;
  matmul.transB = name[len - 1] == 'N' ? false : true;

  std::vector<std::string> stringFrags;
  splitString(name, '_', stringFrags);
  std::vector<int> batch;

  for (int i = 1; i < stringFrags.size() - 4; i ++) {
    batch.push_back(std::stoi(stringFrags[i]));
  }
  matmul.batch = batch;
  std::string M = stringFrags[stringFrags.size() - 4];
  std::string N = stringFrags[stringFrags.size() - 3];
  std::string K = stringFrags[stringFrags.size() - 2];
  M.erase(M.begin()); N.erase(N.begin()); K.erase(K.begin());
  matmul.m = std::stoi(M);
  matmul.n = std::stoi(N);
  matmul.k = std::stoi(K);
}

bool FMHAOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto funcCalls = Analyzer::collectFuncCalls(module);
  int funcNum = funcCalls.size();
  for (int i = 0; i < funcNum; i += 1) {
    auto call2Matmul = funcCalls[i];
    // auto func = call.getCalleeAttrName().str();
    auto funcName = call2Matmul.getCallee().str();
    if (funcName.find(std::string("BatchMatmul")) != std::string::npos) {
      auto matmul = Analyzer::getTargetFunction(module, funcName);
      auto attr = matmul->getAttr(std::string("func.state")).dyn_cast<mlir::StringAttr>();
      if (attr.str() != std::string("cpu")) continue;
      auto retValue = call2Matmul.getResult(0);
      auto users = retValue.getUsers();
      ///< If we want to fuse kernels, The only user has to be gurantee. 
      if (Analyzer::getUsersNumber(users) != 1) {
        continue;
      }
      auto call2Softmax = mlir::dyn_cast<mlir::func::CallOp>(*users.begin());
      if (!call2Softmax) {
        continue;
      }
      auto funcName = call2Softmax.getCallee().str();
      if (funcName.find(std::string("Softmax")) != std::string::npos) {
        auto softmax = Analyzer::getTargetFunction(module, funcName);
        auto attr = softmax->getAttr(std::string("func.state")).dyn_cast<mlir::StringAttr>();
        if (attr.str() != std::string("cpu")) continue;
        auto retValue = call2Softmax->getResult(0);
        auto users = retValue.getUsers();
        ///< If we want to fuse kernels, The only user has to be gurantee. 
        if (Analyzer::getUsersNumber(users) != 1) {
          continue;
        }
        auto call2Matmul2 = mlir::dyn_cast<mlir::func::CallOp>(*users.begin());
        if (!call2Matmul2) {
          continue;
        }
        auto funcName = call2Matmul2.getCallee().str();
        if (funcName.find(std::string("BatchMatmul")) != std::string::npos
          && uniqueFuncCalls.count(call2Matmul) == 0 && uniqueFuncCalls.count(call2Softmax) == 0
          && uniqueFuncCalls.count(call2Matmul2) == 0 && call2callsMap.count(call2Matmul) == 0
          && call2bufferMap.count(call2Matmul) == 0) {

          auto matmul2 = Analyzer::getTargetFunction(module, funcName);
          auto attr = matmul2->getAttr(std::string("func.state")).dyn_cast<mlir::StringAttr>();
          if (attr.str() != std::string("cpu")) continue;

          ///< Collect necessary information.
          MemoryBuffer buf;
          auto matmulArgs = call2Matmul.getArgOperands();
          buf.Q = matmulArgs[0];
          buf.K = matmulArgs[1];
          buf.S = matmulArgs[2];
          BatchMatmulDescriptor descMatmul;
          identifyBatchMatmul(call2Matmul.getCallee().str(), descMatmul);
          
          auto matmul2Args = call2Matmul2.getArgOperands();
          buf.V = matmul2Args[1];
          buf.O = matmul2Args[2];
          BatchMatmulDescriptor descMatmul2;
          identifyBatchMatmul(call2Matmul2.getCallee().str(), descMatmul2);
          

          ///< Not impatible.
          if (descMatmul.batch != descMatmul2.batch ||
            descMatmul.m != descMatmul2.m || descMatmul.n != descMatmul2.k) {
            continue;
          }
          buf.matmul1 = descMatmul;
          buf.matmul2 = descMatmul2;
          // descMatmul.log(); descMatmul2.log();

          ///< Record the buffer.
          call2bufferMap[call2Matmul] = buf;

          ///< Record the graph.
          call2callsMap[call2Matmul] = std::vector<mlir::func::CallOp>{call2Softmax, call2Matmul2};

          ///< Unique all three func calls.
          uniqueFuncCalls.insert(call2Matmul);
          uniqueFuncCalls.insert(call2Softmax);
          uniqueFuncCalls.insert(call2Matmul2);
        }
      }
    }
  }
}

mlir::AffineMap FMHAOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto dim7 = builder.getAffineDimExpr(7);
  int64_t BLOCK_SIZE = fmhaConfig["BLOCK_SIZE"];
  int width = fmhaConfig["Width"];

  // std::vector<int64_t> warpOrg {2, 4};
  // std::vector<int64_t> threadOrg {8, 4};

  const int LaneX_S = fmhaConfig["Bc"] / fmhaConfig["BcTileS"];
  const int LaneY_S = fmhaConfig["WARP_SIZE"] / LaneX_S;

  const int LaneX_O = fmhaConfig["Hd"] / fmhaConfig["HdTileO"] / fmhaConfig["WarpX_O"];
  const int LaneY_O = fmhaConfig["WARP_SIZE"] / LaneX_O;

  if (mapIdentifier == "loadTileQ") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.x, hd_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim3;
    auto vThreadIdExpr = threadIdExpr + dim5 * BLOCK_SIZE;
    auto Br_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim2 * fmhaConfig["Br"] + Br_Offset);
    exprs.push_back(dim4 + Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadTileK") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, threadIdx.x, k_outer, hd_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim2;
    auto vThreadIdExpr = threadIdExpr + dim5 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim3 + Bc_Offset);
    exprs.push_back(dim4 + Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileQ") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, iv, ivInVector]
    auto threadIdExpr = dim0;
    auto vThreadIdExpr = threadIdExpr + dim1 * BLOCK_SIZE;
    auto Br_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(Hd_Offset * width + dim2);
    exprs.push_back(Br_Offset);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileK") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, iv, ivInVector]
    auto threadIdExpr = dim0;
    auto vThreadIdExpr = threadIdExpr + dim1 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(Hd_Offset * width + dim2);
    exprs.push_back(Bc_Offset);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragQ") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, hd_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ylane_s = laneId.floorDiv(LaneX_S);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1);
    exprs.push_back(warpId * LaneY_S * fmhaConfig["BrTileS"] + dim2 * LaneY_S * width + ylane_s * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());    
  } else if (mapIdentifier == "loadFragK") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, hd_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xlane_s = laneId % (LaneX_S);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1);
    exprs.push_back(dim2 * LaneX_S * width + xlane_s * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());    
  } else if (mapIdentifier == "brIdxS") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ylane_s = laneId.floorDiv(LaneX_S);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(warpId * LaneY_S * fmhaConfig["BrTileS"] + dim1.floorDiv(width) * LaneY_S * width + ylane_s * width + dim1 % width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
  } else if (mapIdentifier == "storeTileP") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, bc, br]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ylane_s = laneId.floorDiv(LaneX_S);
    auto xlane_s = laneId % (LaneX_S);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1.floorDiv(width) * LaneX_S * width + xlane_s * width + dim1 % width);
    exprs.push_back(warpId * LaneY_S * fmhaConfig["BrTileS"] + dim2 * LaneY_S + ylane_s * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
  } else if (mapIdentifier == "readTileS") {
    // dims are:[dim0, dim1]
    // operands are: [bc, br]
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFactor") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, br]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xwarp_o = warpId % fmhaConfig["WarpX_O"];
    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim1 * LaneY_O * width + ylane_o * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadTileV") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, threadIdx.x, k_outer, bc_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim2;
    auto vThreadIdExpr = threadIdExpr + dim5 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Hd"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Hd"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim3 + Bc_Offset + dim4);
    exprs.push_back(Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileV") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, iv]
    auto threadIdExpr = dim0;
    auto vThreadIdExpr = threadIdExpr + dim1 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Hd"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Hd"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(Bc_Offset);
    exprs.push_back(Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());    
  } else if (mapIdentifier == "loadFragP") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.x, bc_outer, bc_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1 + dim2);
    exprs.push_back(ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim3 * width * LaneY_O + ylane_o * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());   
  } else if (mapIdentifier == "loadFragV") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, bc_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xwarp_o = warpId % fmhaConfig["WarpX_O"];
    auto xlane_o = laneId % LaneX_O;

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1);
    exprs.push_back(xwarp_o * LaneX_O * fmhaConfig["HdTileO"] + dim2 * width * LaneX_O + xlane_o * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());   
  } else if (mapIdentifier == "brIdxO") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim1 * LaneY_O * width + ylane_o * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
  } else if (mapIdentifier == "storeTileO") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.x, br, hd]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xwarp_o = warpId % fmhaConfig["WarpX_O"];
    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);
    auto xlane_o = laneId % (LaneX_O);

    auto Br_Offset = ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim4.floorDiv(width) * width * LaneY_O + ylane_o * width + dim4 % width;
    auto Hd_Offset = xwarp_o * LaneX_O * fmhaConfig["HdTileO"] + dim5 * LaneX_O + xlane_o * width;

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim2 * fmhaConfig["Br"] + Br_Offset);
    exprs.push_back(Hd_Offset);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void initMaxSum(mlir::OpBuilder& builder, int64_t lowerBound, int64_t upperBound, int64_t step, mlir::Value tid,
      mlir::Value smemMax, mlir::Value flt_min, mlir::Value smemSum, mlir::Value zero) {
  auto initMaxSumBody = [&](mlir::OpBuilder &kBuilder, mlir::Location kLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard kGuard(kBuilder);

    auto yieldOp = builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
    builder.setInsertionPoint(yieldOp);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    llvm::SmallVector<bool> eqFlags;
    // iv + 2 * step <= ub
    //-> ub - 2 * step - iv >= 0
    auto dim0 = builder.getAffineDimExpr(0);
    auto dim1 = builder.getAffineDimExpr(1);
    exprs.push_back(upperBound - 1 - dim0 - dim1);
    eqFlags.push_back(false);///< true: ==, false: >=
    auto cst = mlir::IntegerSet::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), llvm::ArrayRef<bool>(eqFlags));
    auto ifOp = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{tid, iv}, 
                                                /*withElseRegion=*/false);

    auto ifBody = ifOp.getBody();
    builder.setInsertionPointToStart(ifBody);

    mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>({dim0 + dim1}), builder.getContext());

    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), 
      zero, smemSum, map, mlir::ValueRange{tid, iv});
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), 
      flt_min, smemMax, map, mlir::ValueRange{tid, iv});

  };
  auto initMaxSum = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 
    lowerBound, upperBound, step, /*iterArgs=llvm::None*/ std::nullopt, initMaxSumBody);
}

void FMHAOptimizer::softmaxIR(mlir::OpBuilder& builder, mlir::Value tileS, mlir::Value rowMax, mlir::Value smMax, mlir::Value rowSum, 
  mlir::Value smSum, mlir::Value smFac, mlir::Value zero, mlir::Value flt_min, mlir::Value tid) {
  auto br_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br = br_loop.getInductionVar();
  builder.setInsertionPointToStart(br_loop.getBody());
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), flt_min, rowMax, mlir::ValueRange({br}));
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), zero, rowSum, mlir::ValueRange({br}));

  auto bc_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BcTileS"], 1);
  auto bc = bc_loop.getInductionVar();
  builder.setInsertionPointToStart(bc_loop.getBody());

  auto tmp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br}));
  auto tmp2 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), tileS, mlir::ValueRange({bc, br}));
  auto tmp3 = builder.create<mlir::arith::MaxNumFOp>(builder.getUnknownLoc(), tmp1, tmp2);

  auto tmp4 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br}));
  auto tmp5 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), tmp3.getResult());
  auto tmp6 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp5.getResult());
  auto tmp7 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowSum, mlir::ValueRange({br}));
  auto tmp8 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp6.getResult(), tmp7.getResult());

  auto tmp9 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp2.getResult(), tmp3.getResult());
  auto tmp10 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp9.getResult());

  auto tmp11 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), tmp8.getResult(), tmp10.getResult());
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp11.getResult(), rowMax, mlir::ValueRange({br}));
  
  builder.setInsertionPointAfter(br_loop);
  auto br_shfl_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br_sf = br_shfl_loop.getInductionVar();
  builder.setInsertionPointToStart(br_shfl_loop.getBody());

  for (int i = 1; i < fmhaConfig["Bc"] / fmhaConfig["BcTileS"]; i *= 2) {
    auto tmp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br_sf}));
    auto tmp2 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowSum, mlir::ValueRange({br_sf}));

    auto tmp3 = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), tmp1.getResult(), i, 
        fmhaConfig["Bc"] / fmhaConfig["BcTileS"], mlir::gpu::ShuffleMode::DOWN);
    auto tmp4 = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), tmp2.getResult(), i, 
        fmhaConfig["Bc"] / fmhaConfig["BcTileS"], mlir::gpu::ShuffleMode::DOWN);
    
    auto tmp5 = builder.create<mlir::arith::MaxNumFOp>(builder.getUnknownLoc(), tmp1.getResult(), tmp3.getResult(0));

    auto tmp6 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), tmp5.getResult());
    auto tmp7 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp6.getResult());
    auto tmp8 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp2.getResult(), tmp7.getResult());

    auto tmp9 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp3.getResult(0), tmp5.getResult());
    auto tmp10 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp9.getResult());
    auto tmp11 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp4.getResult(0), tmp10.getResult());

    auto tmp12 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), tmp8.getResult(), tmp11.getResult());
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp12.getResult(), rowSum, mlir::ValueRange({br_sf}));
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp5.getResult(), rowMax, mlir::ValueRange({br_sf}));
  }

  builder.setInsertionPointAfter(br_shfl_loop);

  llvm::SmallVector<mlir::AffineExpr> exprs;
  llvm::SmallVector<bool> eqFlags;
  // iv + 2 * step <= ub
  //-> ub - 2 * step - iv >= 0
  auto dim0 = builder.getAffineDimExpr(0);
  exprs.push_back(dim0 % (fmhaConfig["Bc"] / fmhaConfig["BcTileS"]));
  eqFlags.push_back(true);///< true: == 0, false: >= 0
  auto cst = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), llvm::ArrayRef<bool>(eqFlags));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{tid}, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(ifOp.getBody());

  auto br_wb_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br_wb = br_wb_loop.getInductionVar();
  builder.setInsertionPointToStart(br_wb_loop.getBody());

  auto map = getAffineMap("brIdxS", builder);
  {
  auto oldMax = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), smMax, map, mlir::ValueRange{tid, br_wb});
  auto oldSum = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), smSum, map, mlir::ValueRange{tid, br_wb});
  auto tmp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br_wb}));
  auto newMax = builder.create<mlir::arith::MaxNumFOp>(builder.getUnknownLoc(), oldMax.getResult(), tmp1.getResult());
  auto tmp3 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), newMax.getResult());
  auto rowFactor = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp3.getResult());
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), newMax.getResult(), smMax, map, mlir::ValueRange({tid, br_wb}));
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), rowFactor.getResult(), smFac, map, mlir::ValueRange({tid, br_wb}));

  auto tmp4 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), oldSum.getResult(), rowFactor.getResult());

  auto tmp5 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowSum, mlir::ValueRange({br_wb}));
  auto tmp6 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), newMax.getResult());
  auto tmp7 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp6.getResult());
  auto tmp8 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp5.getResult(), tmp7.getResult());

  auto tmp9 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), tmp4.getResult(), tmp8.getResult());

  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp9.getResult(), smSum, map, mlir::ValueRange({tid, br_wb}));
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), newMax.getResult(), rowMax, mlir::ValueRange({br_wb}));
  }
  builder.setInsertionPointAfter(ifOp);
  ///< Write P to shared memory.
  ///< Update rowMax within the same row.
  auto br_broadcast_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br_broadcast = br_broadcast_loop.getInductionVar();
  builder.setInsertionPointToStart(br_broadcast_loop.getBody());
  {
  auto tmp1 = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br_broadcast}));
  auto tmp2 = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), tmp1.getResult(), 0, 
      fmhaConfig["Bc"] / fmhaConfig["BcTileS"], mlir::gpu::ShuffleMode::IDX);
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp2.getResult(0), rowMax, mlir::ValueRange({br_broadcast}));
  }
  builder.setInsertionPointAfter(br_broadcast_loop);
  
  ///< factor each element by exp.
  ///< tileS[bc][br] = exp(tileS[bc][br] - rowMax[br]);
  auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BcTileS"], 1);
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(outerLoop.getBody());
  auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  builder.setInsertionPointToStart(innerLoop.getBody());
  {
    auto bc = outerLoop.getInductionVar();
    auto br = innerLoop.getInductionVar();
    auto ld_s = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), tileS, mlir::ValueRange({bc, br}));
    auto ld_row_max = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br}));
    auto tmp1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), ld_s.getResult(), ld_row_max.getResult());
    auto tmp2 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp1.getResult());
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp2.getResult(), tileS, mlir::ValueRange({bc, br}));
     
  }
  builder.restoreInsertionPoint(ip);
}

void FMHAOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto& item : call2callsMap) {
    auto call2Matmul = item.first;
    auto call2Softmax = item.second[0];
    auto call2Matmul2 = item.second[1];
    auto matmul = Analyzer::collectFunctions(module, call2Matmul.getCallee().str())[0];
    auto softmax = Analyzer::collectFunctions(module, call2Softmax.getCallee().str())[0];
    auto matmul2 = Analyzer::collectFunctions(module, call2Matmul2.getCallee().str())[0];
    auto buf = call2bufferMap[call2Matmul];
    auto matmul1Desc = buf.matmul1;
    auto Q = buf.Q;
    auto K = buf.K;
    auto V = buf.V;
    auto O = buf.O;

    auto QType = Q.getType().dyn_cast<mlir::MemRefType>();
    auto elementType = QType.getElementType();

    const int seq_len = matmul1Desc.m;
    const int head_dim = matmul1Desc.k;

    ///< New Func Name.
    auto fusedFuncName = std::string({"Fused_Multi_Head_Attention"});
    for (auto b : matmul1Desc.batch) {
      fusedFuncName += "_";
      fusedFuncName += std::to_string(b);
    }
    fusedFuncName += "_SL" + std::to_string(seq_len) + "_HD" + std::to_string(head_dim);
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(matmul);
    auto funcOp = buildFuction(module, builder, fusedFuncName, {Q.getType(), K.getType(), V.getType(), O.getType()}, {O.getType()});
    builder.setInsertionPointAfter(call2Matmul2);
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({Q, K, V, O}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));

    ///< Erase old three function calls.
    call2Matmul2.getResult(0).replaceAllUsesWith(callOp.getResult(0));
    call2Matmul2.erase();
    call2Softmax.erase();
    call2Matmul.erase();

    auto& bodyBlock = funcOp.front();
    auto newArgs = bodyBlock.getArguments();
    builder.setInsertionPointToStart(&bodyBlock);
    auto returnOp = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), newArgs[3]);

    builder.setInsertionPoint(returnOp);

    ///<===================================  Generate Fused Kernel ==========================
    Q = newArgs[0];
    K = newArgs[1];
    V = newArgs[2];
    O = newArgs[3];
    std::vector<int64_t> bounds;

    for (auto b : matmul1Desc.batch) bounds.push_back(b);

    auto Hd = head_dim;
    auto Br = fmhaConfig["HdxBr"] / Hd;
    auto Bc = fmhaConfig["BrxBc"] / Br;
    auto Slice = fmhaConfig["Slice"];
    auto BLOCK_SIZE = fmhaConfig["BLOCK_SIZE"];

    fmhaConfig["Hd"] = Hd;
    fmhaConfig["Br"] = Br;
    fmhaConfig["Bc"] = Bc;

    bounds.push_back(seq_len / Br);
    bounds.push_back(fmhaConfig["BLOCK_SIZE"]);

    mlir::SmallVector<int64_t, 8> lowerBounds(bounds.size(), /*Value=*/0);
    mlir::SmallVector<int64_t, 8> steps(bounds.size(), /*Value=*/1);
    mlir::SmallVector<int64_t, 8> upperBounds(bounds.begin(), bounds.end());
    mlir::affine::buildAffineLoopNest(
      builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {});
    
    auto loops = Analyzer::collectFuncLoops(funcOp);
    auto batch = loops[0];
    auto head_num = loops[1];
    auto seq_frag = loops[2];
    auto blockLoop = loops[3];

    auto gridLevel = Rewriter::parallel({batch, head_num, seq_frag});
    auto blockLevel = Rewriter::parallel({blockLoop});
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto smQ = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Slice, Br}, elementType);
    auto smK = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Slice, Bc}, elementType);
    auto smV = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Slice, Bc}, elementType);
    auto smP = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br, Bc}, elementType);
    auto smMax = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br}, elementType);
    auto smSum = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br}, elementType);
    auto smFac = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br}, elementType);

    auto BrTileS = fmhaConfig["BrTileS"];
    auto BcTileS = fmhaConfig["BcTileS"];
    auto BrTileO = fmhaConfig["BrTileO"];
    auto HdTileO = fmhaConfig["HdTileO"];

    auto tileO = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {BrTileO, HdTileO}, elementType);

    builder.setInsertionPointAfter(Analyzer::getLastOp<mlir::memref::AllocOp>(blockLevel));

    auto zero = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), 
        builder.getFloatAttr(elementType, 0));
    auto flt_min = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), 
        builder.getFloatAttr(elementType, -FLT_MAX));

    Rewriter::set_buffer(builder, tileO, zero.getResult());
    
    initMaxSum(builder, 0, Br, BLOCK_SIZE, blockLevel.getIVs()[0], smMax, flt_min.getResult(), smSum, zero.getResult());

    auto outer_reduce = Rewriter::create_constant_loop(builder, 0, seq_len, Bc);
    
    auto ldgQ = Rewriter::alloc_buffer(outer_reduce, MemorySpace::local, {Slice * Br / BLOCK_SIZE}, elementType);
    auto ldgK = Rewriter::alloc_buffer(outer_reduce, MemorySpace::local, {Slice * Bc / BLOCK_SIZE}, elementType);

    auto tileS = Rewriter::alloc_buffer(outer_reduce, MemorySpace::local, {BcTileS, BrTileS}, elementType);

    builder.setInsertionPointAfter(Analyzer::getLastOp<mlir::memref::AllocOp>(outer_reduce));
    Rewriter::set_buffer(builder, tileS, zero.getResult());

    auto hd_outer = Rewriter::create_constant_loop(builder, 0, Hd, Slice);
    auto bar1 = Rewriter::barrier(hd_outer, Position::begin);

    builder.setInsertionPointAfter(bar1);

    Rewriter::read(builder, Q, ldgQ, getAffineMap("loadTileQ", builder), {gridLevel.getIVs()[0], gridLevel.getIVs()[1], 
      gridLevel.getIVs()[2], blockLevel.getIVs()[0], hd_outer.getInductionVar()}, fmhaConfig["Width"]);
    Rewriter::read(builder, K, ldgK, getAffineMap("loadTileK", builder), {gridLevel.getIVs()[0], gridLevel.getIVs()[1], 
      blockLevel.getIVs()[0], outer_reduce.getInductionVar(), hd_outer.getInductionVar()}, fmhaConfig["Width"]);
    Rewriter::write(builder, ldgQ, smQ, getAffineMap("storeTileQ", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);
    auto writeSmK = Rewriter::write(builder, ldgK, smK, getAffineMap("storeTileK", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);

    auto bar2 = Rewriter::barrier(writeSmK, Position::after);

    auto fragQ = Rewriter::alloc_buffer(bar2, Position::after, MemorySpace::local, {BrTileS}, elementType);
    auto fragK = Rewriter::alloc_buffer(fragQ.getDefiningOp(), Position::after, MemorySpace::local, {BcTileS}, elementType);

    auto hd_inner = Rewriter::create_constant_loop(builder, 0, Slice, 1);
    builder.setInsertionPointToStart(hd_inner.getBody());
    Rewriter::read(builder, smQ, fragQ, getAffineMap("loadFragQ", builder), {blockLevel.getIVs()[0], hd_inner.getInductionVar()}, 
      fmhaConfig["Width"]);
    Rewriter::read(builder, smK, fragK, getAffineMap("loadFragK", builder), {blockLevel.getIVs()[0], hd_inner.getInductionVar()}, 
      fmhaConfig["Width"]);

    Rewriter::outer_product(builder, tileS, fragK, fragQ, BcTileS, BrTileS);

    builder.setInsertionPointAfter(hd_outer);

    auto rowMax = Rewriter::alloc_buffer(hd_outer, Position::after, MemorySpace::local, {BrTileS}, elementType);
    auto rowSum = Rewriter::alloc_buffer(rowMax.getDefiningOp(), Position::after, MemorySpace::local, {BrTileS}, elementType);

    softmaxIR(builder, tileS, rowMax, smMax, rowSum, smSum, smFac, zero.getResult(), flt_min.getResult(), blockLevel.getIVs()[0]);

    ///< Write P(S) to shared memory.
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BcTileS"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], fmhaConfig["Width"]);
    builder.setInsertionPointToStart(innerLoop.getBody());
    {
      auto bc = outerLoop.getInductionVar();
      auto br = innerLoop.getInductionVar();
      auto vectorType = mlir::VectorType::get(fmhaConfig["Width"], tileS.getType().dyn_cast<mlir::MemRefType>().getElementType());
      auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, tileS, getAffineMap("readTileS", builder), mlir::ValueRange({bc, br}));
      auto st = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), smP, getAffineMap("storeTileP", builder), mlir::ValueRange{blockLevel.getIVs()[0], bc, br}); 
    }
    auto bar3 = Rewriter::barrier(outerLoop, Position::after);
    auto factor = Rewriter::alloc_buffer(bar3, Position::after, MemorySpace::local, {BrTileO}, elementType);
    builder.setInsertionPointAfter(factor.getDefiningOp());
    Rewriter::read(builder, smFac, factor, getAffineMap("loadFactor", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);

    ///< Refactor tileO.
    {
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileO"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["HdTileO"], 1);
    builder.setInsertionPointToStart(innerLoop.getBody());
  
    auto br = outerLoop.getInductionVar();
    auto hd = innerLoop.getInductionVar();
    
    auto ld_tile_o = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), tileO, mlir::ValueRange({br, hd}));
    auto fac = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), factor, mlir::ValueRange({br}));
    auto tmp = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_tile_o.getResult(), fac.getResult());
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp.getResult(), tileO, mlir::ValueRange{br, hd});
    builder.setInsertionPointAfter(outerLoop);
    }

    ///< Compute O = PV
    auto bc_outer = Rewriter::create_constant_loop(builder, 0, Bc, Slice);
    const int LdgV = Slice * Hd / BLOCK_SIZE;
    auto ldgV = Rewriter::alloc_buffer(bc_outer, Position::before, MemorySpace::local, {LdgV}, elementType);

    auto bar4 = Rewriter::barrier(bc_outer, Position::begin);

    builder.setInsertionPointAfter(bar4);

    Rewriter::read(builder, V, ldgV, getAffineMap("loadTileV", builder), {gridLevel.getIVs()[0], gridLevel.getIVs()[1], 
      blockLevel.getIVs()[0], outer_reduce.getInductionVar(), bc_outer.getInductionVar()}, fmhaConfig["Width"]);

    auto writeSmV = Rewriter::write(builder, ldgV, smV, getAffineMap("storeTileV", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);

    auto bar5 = Rewriter::barrier(writeSmV, Position::after);

    auto fragP = Rewriter::alloc_buffer(bar5, Position::after, MemorySpace::local, {BrTileO}, elementType);
    auto fragV = Rewriter::alloc_buffer(fragP.getDefiningOp(), Position::after, MemorySpace::local, {HdTileO}, elementType);

    auto bc_inner = Rewriter::create_constant_loop(builder, 0, Slice, 1);
    builder.setInsertionPointToStart(bc_inner.getBody());
    Rewriter::read(builder, smP, fragP, getAffineMap("loadFragP", builder), {blockLevel.getIVs()[0], bc_outer.getInductionVar(), bc_inner.getInductionVar()}, 
      fmhaConfig["Width"]);
    Rewriter::read(builder, smV, fragV, getAffineMap("loadFragV", builder), {blockLevel.getIVs()[0], bc_inner.getInductionVar()}, 
      fmhaConfig["Width"]);

    Rewriter::outer_product(builder, tileO, fragP, fragV, BrTileO, HdTileO);

    ///< Load sum
    auto rowSumO = Rewriter::alloc_buffer(outer_reduce, Position::after, MemorySpace::local, {BrTileO}, elementType);
    builder.setInsertionPointAfter(rowSumO.getDefiningOp());
    Rewriter::read(builder, smSum, rowSumO, getAffineMap("brIdxO", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);
    ///< Refactor tileO
    {
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileO"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["HdTileO"], 1);
    builder.setInsertionPointToStart(innerLoop.getBody());
  
    auto br = outerLoop.getInductionVar();
    auto hd = innerLoop.getInductionVar();
    
    auto ld_tile_o = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), tileO, mlir::ValueRange({br, hd}));
    auto tmp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), rowSumO, mlir::ValueRange({br}));
    auto tmp1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), ld_tile_o.getResult(), tmp.getResult());
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tmp1.getResult(), tileO, mlir::ValueRange{br, hd});
    builder.setInsertionPointAfter(outerLoop);
    }
    
    ///< Write back O
    {
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileO"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["HdTileO"], fmhaConfig["Width"]);
    builder.setInsertionPointToStart(innerLoop.getBody());
  
    auto br = outerLoop.getInductionVar();
    auto hd = innerLoop.getInductionVar();
    
    auto vectorType = mlir::VectorType::get(fmhaConfig["Width"], tileO.getType().dyn_cast<mlir::MemRefType>().getElementType());
    auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, tileO, mlir::ValueRange({br, hd}));
    auto st = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), O, getAffineMap("storeTileO", builder), 
        mlir::ValueRange({gridLevel.getIVs()[0], gridLevel.getIVs()[1], gridLevel.getIVs()[2], blockLevel.getIVs()[0], br, hd})); 
    }
  }
}

/*----------------------------batch matmul-------------------------------*/
bool BatchMatmulOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& batchMatmulFuncs = Analyzer::collectFunctions(module, "BatchMatmul");
  bool res = batchMatmulFuncs.size() != 0 ? true : false;

  for (auto& batchMatmulFunc : batchMatmulFuncs) {
    if (batchMatmuls.count(batchMatmulFunc) != 0 || batchMatmulLoops.count(batchMatmulFunc) != 0
      || batchMatmulBuffers.count(batchMatmulFunc) != 0) {
      llvm::errs() << "Duplicated batchMatmul in module\n";
    }
    batchMatmuls.insert(batchMatmulFunc);
    auto&& loops = Analyzer::collectFuncLoops(batchMatmulFunc);
    batchMatmulLoops[batchMatmulFunc] = std::move(loops);
    auto funcArgs = batchMatmulFunc.front().getArguments();

    MemoryBuffer buf;
    buf.A = funcArgs[0];
    buf.B = funcArgs[1];
    auto &block = batchMatmulFunc.front();
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    buf.C = returnOp.getOperand(0);

    BatchMatmulDescriptor descirpe;
    auto funcName = batchMatmulFunc.getSymName().str();
    identifyBatchMatmul(funcName, descirpe);
    buf.matmul = descirpe;

    batchMatmulBuffers[batchMatmulFunc] = buf;
  }
  return res;
}

mlir::AffineExpr shiftExprDim(mlir::OpBuilder& builder, mlir::AffineExpr expr, int shift) {
  auto context = builder.getContext();
  if (auto dimExpr_ = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return mlir::getAffineDimExpr(dimExpr_.getPosition() + shift, context);
  } else if (auto binaryExpr_ = expr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = shiftExprDim(builder, binaryExpr_.getLHS(), shift);
    auto RHS = shiftExprDim(builder, binaryExpr_.getRHS(), shift);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = expr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

// mlir::AffineMap BatchMatmulOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, int64_t batchNum) {
//   auto dim0 = builder.getAffineDimExpr(0);
//   auto dim1 = builder.getAffineDimExpr(1);
//   auto dim2 = builder.getAffineDimExpr(2);
//   auto dim3 = builder.getAffineDimExpr(3);
//   auto dim4 = builder.getAffineDimExpr(4);
//   auto dim5 = builder.getAffineDimExpr(5);
//   auto dim6 = builder.getAffineDimExpr(6);
//   auto dim7 = builder.getAffineDimExpr(7);
//   auto dim8 = builder.getAffineDimExpr(8);
//   auto dim9 = builder.getAffineDimExpr(9);
//   int width = batchMatmulConfig["VECTORIZE_WIDTH"];
//   int block_size = batchMatmulConfig["BLOCK_SIZE_M"];
//   int for_size = batchMatmulConfig["FOR_SIZE_N"];
//   int thread_size = batchMatmulConfig["THREAD_SIZE"];
//   int slice = batchMatmulConfig["Slice"];

//   if (mapIdentifier == "one") {
//     std::vector<mlir::AffineExpr> exprs;
//     int interval = block_size/((block_size*slice/width)/block_size); // 128/(((128*8)/4)/128)
//     for (int i=0; i<batchNum; i++) {
//       exprs.push_back(builder.getAffineDimExpr(i));
//     }
//     auto yDimExpr = dim0+dim1.floorDiv(slice/width)+dim3*interval;
//     auto xDimExpr = dim2+(dim1%(slice/width))*width;
//     exprs.push_back(shiftExprDim(builder, yDimExpr, batchNum));
//     exprs.push_back(shiftExprDim(builder, xDimExpr, batchNum));
//     return mlir::AffineMap::get(4+batchNum, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "two") {
//     std::vector<mlir::AffineExpr> exprs;
//     int interval = for_size/((for_size*slice/width)/block_size);  // block size is count of thread in per block.
//     for (int i=0; i<batchNum; i++) {
//       exprs.push_back(builder.getAffineDimExpr(i));
//     }
//     auto yDimExpr = dim0+dim1.floorDiv(for_size/width)+dim3*interval;
//     auto xDimExpr = dim2+(dim1%(for_size/width))*width;
//     exprs.push_back(shiftExprDim(builder, yDimExpr, batchNum));
//     exprs.push_back(shiftExprDim(builder, xDimExpr, batchNum));
//     return mlir::AffineMap::get(4+batchNum, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "three"){
//     std::vector<mlir::AffineExpr> exprs;
//     int interval = block_size/((block_size*slice/width)/block_size);
//     exprs.push_back(dim0%(slice/width)*width+dim2);
//     exprs.push_back(dim0.floorDiv(slice/width)+dim1*interval);
//     return mlir::AffineMap::get(3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "four"){
//     std::vector<mlir::AffineExpr> exprs;
//     int interval = for_size/((for_size*slice/width)/block_size);
//     exprs.push_back(dim0.floorDiv(for_size/width)+dim1*interval);
//     exprs.push_back(dim0 % (for_size/width)*width);
//     return mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "five"){
//     std::vector<mlir::AffineExpr> exprs;
//     exprs.push_back(dim0);
//     exprs.push_back(dim1*4+dim2*(block_size/(thread_size/width)));
//     return mlir::AffineMap::get(3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "six"){
//     std::vector<mlir::AffineExpr> exprs;
//     exprs.push_back(dim0);
//     exprs.push_back(dim1*4+dim2*(for_size/(thread_size/width)));
//     return mlir::AffineMap::get(3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "eight") {
//     std::vector<mlir::AffineExpr> exprs;
//     exprs.push_back(dim0);
//     return mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   } 
//   else if (mapIdentifier == "nine") {
//     std::vector<mlir::AffineExpr> exprs;
//     int intervalY = block_size / (thread_size / width);
//     int intervalX = for_size / (thread_size / width);
//     for (int i=0; i<batchNum; i++) {
//       exprs.push_back(builder.getAffineDimExpr(i));
//     }
//     auto yDimExpr = dim0+dim1*intervalY+dim2*width+dim3;
//     auto xDimExpr = dim4+dim5*width+dim6*intervalX+dim7;
//     exprs.push_back(shiftExprDim(builder, yDimExpr, batchNum));
//     exprs.push_back(shiftExprDim(builder, xDimExpr, batchNum));
//     // for (auto e : exprs) {llvm::outs() << e << "\n";} llvm::outs() << "\n";
//     return mlir::AffineMap::get(8+batchNum, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
//   }
//   else {
//     assert(false);
//   }
// }

// void BatchMatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
//   for (auto batchMatmul : batchMatmuls) {
//     auto loops = batchMatmulLoops[batchMatmul];
//     auto buffer = batchMatmulBuffers[batchMatmul];
//     auto A = buffer.A; auto B = buffer.B; auto C = buffer.C;
//     // auto descirpe = buffer.matmul;
//     auto batchNum = buffer.matmul.batch.size();

//     auto m_split_loops = Rewriter::split(loops[loops.size()-3], 3, {batchMatmulConfig["THREAD_SIZE"], batchMatmulConfig["BLOCK_SIZE_M"]});
//     auto n_split_loops = Rewriter::split(loops[loops.size()-2], 3, {batchMatmulConfig["THREAD_SIZE"], batchMatmulConfig["FOR_SIZE_N"]});

//     auto loopK = loops[loops.size() - 1];
//     auto m_outer = m_split_loops[0]; auto m_mider = m_split_loops[1]; auto m_inner = m_split_loops[2];
//     auto n_outer = n_split_loops[0]; auto n_mider = n_split_loops[1]; auto n_inner = n_split_loops[2];

//     Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
//     m_mider = Rewriter::modifyLoopStepToOne(m_mider);
//     n_mider = Rewriter::modifyLoopStepToOne(n_mider);
//     DUMP(module);

//     auto combineLoop = Rewriter::combineToOneDim({m_mider, n_mider});
//     DUMP(module);

//     std::vector<mlir::affine::AffineForOp> palLoops;
//     for (int i=0; i< batchNum; i++) {palLoops.push_back(loops[i]);}
//     palLoops.push_back(m_outer);
//     auto gridLevel = Rewriter::parallel(palLoops);
//     auto blockLevel = Rewriter::parallel({combineLoop});
//     DUMP(module);

//     std::vector<mlir::affine::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
//     auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
//     loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];
//     Rewriter::reorder({loopK, m_inner, n_inner});
//     DUMP(module);

//     auto k_axes = Rewriter::split(loopK, 2, {batchMatmulConfig["BLOCK_SIZE_K"]});
//     auto k_outer = k_axes[0], k_inner = k_axes[1];
//     DUMP(module);

//     int64_t blockThreads;
//     auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);

//     auto ldgASize = batchMatmulConfig["BLOCK_SIZE_K"] * batchMatmulConfig["BLOCK_SIZE_M"] / blockThreads;
//     auto ldgBSize = batchMatmulConfig["BLOCK_SIZE_K"] * batchMatmulConfig["FOR_SIZE_N"] / blockThreads;
//     auto fragSize = batchMatmulConfig["THREAD_SIZE"];

//     auto blockElemIdx = Rewriter::getElementIdx(gridLevel);
//     auto blockIdx = Rewriter::getParallelIdx(gridLevel);
//     auto threadIdx = Rewriter::getParallelIdx(blockLevel);

//     auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
//     auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();
//     auto tileA = Rewriter::alloc_buffer(gridLevel, MemorySpace::local, {ldgASize}, elementA);  // reg4
//     auto tileB = Rewriter::alloc_buffer(gridLevel, MemorySpace::local, {ldgBSize}, elementB);  // reg8

//     auto fragA = Rewriter::alloc_buffer(gridLevel, MemorySpace::local, {fragSize}, elementA);  // reg8
//     auto fragB = Rewriter::alloc_buffer(gridLevel, MemorySpace::local, {fragSize}, elementB);

//     auto smA = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {batchMatmulConfig["BLOCK_SIZE_K"], batchMatmulConfig["BLOCK_SIZE_M"]}, elementA);
//     auto smB = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {batchMatmulConfig["BLOCK_SIZE_K"], batchMatmulConfig["FOR_SIZE_N"]}, elementB);

//     llvm::SmallVector<mlir::Value> operandsA({blockElemIdx[batchNum], threadIdx[0], k_outer.getInductionVar()});
//     for (int i=0; i<batchNum; i++) { operandsA.insert(operandsA.begin(), blockIdx[i]); }
//     auto loadTileA = Rewriter::read(A, tileA, getAffineMap("one", builder, batchNum), operandsA, batchMatmulConfig["VECTORIZE_WIDTH"], k_outer, Position::begin);
//     llvm::SmallVector<mlir::Value> operandsB({k_outer.getInductionVar(), threadIdx[0], n_outer.getInductionVar()});
//     for (int i=0; i<batchNum; i++) { operandsB.insert(operandsB.begin(), blockIdx[i]); }
//     auto loadTileB = Rewriter::read(B, tileB, getAffineMap("two", builder, batchNum), operandsB, batchMatmulConfig["VECTORIZE_WIDTH"], loadTileA, Position::after);
//     DUMP(module);

//     auto storeTileA = Rewriter::write(tileA, smA, getAffineMap("three", builder), 
//                                       {threadIdx[0]}, batchMatmulConfig["VECTORIZE_WIDTH"], loadTileB, Position::after);
//     auto storeTileB = Rewriter::write(tileB, smB, getAffineMap("four", builder), 
//                                       {threadIdx[0]}, batchMatmulConfig["VECTORIZE_WIDTH"], storeTileA, Position::after);
//     auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
//     auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);
//     DUMP(module);

//     int64_t oneDimLen = sqrt(batchMatmulConfig["FOR_SIZE_N"]);
//     auto threadIdx_ =  Rewriter::blockLevelOneToTwo(blockLevel, oneDimLen);

//     auto loadFragA = Rewriter::read(smA, fragA, getAffineMap("five", builder),
//                                     {k_inner.getInductionVar(), threadIdx_[0]}, batchMatmulConfig["VECTORIZE_WIDTH"], k_inner, Position::begin);
//     auto loadFragB = Rewriter::read(smB, fragB, getAffineMap("six", builder), 
//                                     {k_inner.getInductionVar(), threadIdx_[1]}, batchMatmulConfig["VECTORIZE_WIDTH"], loadFragA, Position::after);
//     DUMP(module);

//     Rewriter::cache_read(k_inner, A, fragA, getAffineMap("eight", builder), {m_inner.getInductionVar()});
//     Rewriter::cache_read(k_inner, B, fragB, getAffineMap("eight", builder), {n_inner.getInductionVar()});
//     DUMP(module);

//     auto writeCbody = Rewriter::get_write(blockLevel, C);
//     assert(writeCbody.size() == 1);
//     auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {batchMatmulConfig["VECTORIZE_WIDTH"]});
//     auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {batchMatmulConfig["VECTORIZE_WIDTH"]});
//     auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
//     auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
//     Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
//     m_inner_0 = Rewriter::modifyLoopStepToOne(m_inner_0);
//     n_inner_0 = Rewriter::modifyLoopStepToOne(n_inner_0);
//     DUMP(module);
    
//     llvm::SmallVector<mlir::Value> operandsC({blockElemIdx[batchNum], m_inner_0.getInductionVar(), threadIdx_[0], m_inner_1.getInductionVar(),
//                                               n_outer.getInductionVar(), threadIdx_[1], n_inner_0.getInductionVar(), n_inner_1.getInductionVar()});
//     for (int i=0; i<batchNum; i++) { operandsC.insert(operandsC.begin(), blockIdx[i]); }
//     Rewriter::cache_write(m_inner_0, C, C, getAffineMap("nine", builder, batchNum), operandsC);
//     Rewriter::vectorize(n_inner_1, batchMatmulConfig["VECTORIZE_WIDTH"]);
//     DUMP(module);

//     auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
//     auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
//     auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
//     auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
//     DUMP(module);

//     Rewriter::detach_last_loop(k_inner);
//     DUMP(module);

//     Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
//     Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
//     Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
//     Rewriter::schedule(doubleLoadTileB[1][0], doubleLoadTileA[1][0], Position::after);
//     Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
//     Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
//     auto ifOp = doubleLoadTileA[1][1]->getParentOp();
//     Rewriter::schedule(ifOp, k_inner, Position::after); 
//     Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
//     Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
//     Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
//     Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
//     DUMP(module);

//     Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
//     Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);;
//     DUMP(module);

//     Rewriter::take_off_true_if(module);
//     Rewriter::delete_false_if(module);
//     DUMP(module);
  
//     int64_t threshold = std::max(batchMatmulConfig["BLOCK_SIZE_K"], std::max(batchMatmulConfig["THREAD_SIZE_M"], batchMatmulConfig["THREAD_SIZE"]));
//     Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
//       if (!forOp.hasConstantBounds()) return false;
//       auto step = forOp.getStep();
//       auto ub = forOp.getConstantUpperBound();
//       auto lb = forOp.getConstantLowerBound();
//       auto times = (ub - lb) / step;
//       if (times >= std::min<int64_t>(threshold, batchMatmulConfig["VECTORIZE_WIDTH"])) return false;
//       return true;
//     });
//     DUMP(module);

//     Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
//       if (!forOp.hasConstantBounds()) return false;
//       auto step = forOp.getStep();
//       auto ub = forOp.getConstantUpperBound();
//       auto lb = forOp.getConstantLowerBound();
//       auto times = (ub - lb) / step;
//       if (times > threshold) return false;
//       return true;
//     });
//     Rewriter::deleteExtraCstOp(gridLevel);
//     DUMP(module);
//   }
// }

mlir::AffineMap BatchMatmulOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto dim7 = builder.getAffineDimExpr(7);
  auto dim8 = builder.getAffineDimExpr(8);

  int64_t block_size_m = batchMatmulConfig["BLOCK_SIZE_M"];
  int64_t block_size_n = batchMatmulConfig["BLOCK_SIZE_N"];
  int64_t block_size_k = batchMatmulConfig["BLOCK_SIZE_K"];
  int64_t thread_size_m = batchMatmulConfig["THREAD_SIZE_M"];
  int64_t thread_size_n = batchMatmulConfig["THREAD_SIZE_N"];
  int64_t warpSize = batchMatmulConfig["WARP_SIZE"];

  int64_t blockDimY = block_size_m / thread_size_m;   // thready
  int64_t blockDimX = block_size_n / thread_size_n;   // threadx
  bool vectorize = batchMatmulConfig.count("VECTORIZE_WIDTH") != 0;
  int width = vectorize ? batchMatmulConfig["VECTORIZE_WIDTH"] : 1;

  int64_t warp_size_y = 8;  // 将一个warp排列为y方向8个thread
  int64_t warp_size_x = 4;  // 将一个warp排列为x方向4个thread
  int64_t perBlockWarp_x = 4;  // 将一个block划分为x方向4个warp
  int64_t perBlockWarp_y = (blockDimY * blockDimX) / warpSize / perBlockWarp_x;  // 将一个block划分为y方向blockThreadNum/32/4个warp

  // std::vector<int64_t> warpOrg {2, 4};  
  // std::vector<int64_t> threadOrg {8, 4};

  if (mapIdentifier == "loadTileA") {
    auto threadIdExpr = dim1 * blockDimX + dim2;  // tid
    auto virtaulThreadIxExpr = threadIdExpr + dim5 * blockDimY * blockDimX;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(block_size_k) / width);
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(block_size_k) / width); 
    auto M_Base = dim3 * block_size_m;
    auto K_Base = dim4;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(M_Offset + M_Base);
    exprs.push_back(K_Offset * width + K_Base);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "loadTileB") {
    auto threadIdExpr = dim1 * blockDimX + dim2;
    auto virtaulThreadIxExpr = threadIdExpr + dim5 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(block_size_n) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(block_size_n) / width); 
    auto K_Base = dim3;
    auto N_Base = dim4 * block_size_n;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(K_Offset + K_Base);
    exprs.push_back(N_Offset * width + N_Base);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "storeTileA") {
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(block_size_k) / width);
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(block_size_k) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset * width + dim3);
    exprs.push_back(M_Offset);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "storeTileB") {
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(block_size_n) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(block_size_n) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset);
    exprs.push_back(N_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "loadFragA") {
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(warpSize));
    auto laneId = threadIdExpr % static_cast<uint64_t>(warpSize);
    auto M_offset = laneId.floorDiv(warp_size_x) + warp_size_y * (warpId.floorDiv(warp_size_x) + dim3 * perBlockWarp_y);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(M_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "loadFragB") {
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(warpSize));
    auto laneId = threadIdExpr % static_cast<uint64_t>(warpSize);
    auto N_offset = laneId % warp_size_x + warp_size_x * (warpId % warp_size_x + dim3 * perBlockWarp_x);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(N_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "cacheReadA" || mapIdentifier == "cacheReadB") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else if (mapIdentifier == "cacheWriteC") {
    auto threadIdExpr = dim1 * blockDimX + dim2;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(warpSize));
    auto laneId = threadIdExpr % static_cast<uint64_t>(warpSize);

    auto M_offset = laneId.floorDiv(warp_size_x) + warp_size_y * (warpId.floorDiv(warp_size_x) + dim5.floorDiv(width) * perBlockWarp_y);
    auto N_offset = laneId % warp_size_x + warp_size_x * (warpId % warp_size_x + dim6.floorDiv(width) * perBlockWarp_x);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim3 * block_size_m + M_offset * width + dim7);
    exprs.push_back(dim4 * block_size_m + N_offset * width + dim8);
    return mlir::AffineMap::get(/*dimCount*/9, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } 
  else {
    assert(false);
  }
}

void BatchMatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto batchMatmul : batchMatmuls) {
    auto loops = batchMatmulLoops[batchMatmul];
    auto buffer = batchMatmulBuffers[batchMatmul];
    auto A = buffer.A; auto B = buffer.B; auto C = buffer.C;
    // auto descirpe = buffer.matmul;
    auto batchNum = buffer.matmul.batch.size();

    auto m_split_loops = Rewriter::split(loops[loops.size()-3], 3, {batchMatmulConfig["THREAD_SIZE_M"], batchMatmulConfig["BLOCK_SIZE_M"]});
    auto n_split_loops = Rewriter::split(loops[loops.size()-2], 3, {batchMatmulConfig["THREAD_SIZE_N"], batchMatmulConfig["BLOCK_SIZE_N"]});

    auto loopK = loops[loops.size() - 1];
    auto m_outer = m_split_loops[0]; auto m_mider = m_split_loops[1]; auto m_inner = m_split_loops[2];
    auto n_outer = n_split_loops[0]; auto n_mider = n_split_loops[1]; auto n_inner = n_split_loops[2];

    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    DUMP(module);
    
    auto gridLevel = Rewriter::parallel({loops[0], m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    DUMP(module);

    std::vector<mlir::affine::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
    auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
    loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];
    Rewriter::reorder({loopK, m_inner, n_inner});
    DUMP(module);

    auto k_axes = Rewriter::split(loopK, 2, {batchMatmulConfig["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_inner = k_axes[1];
    DUMP(module);

    int64_t blockThreads;
    auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);

    auto ldgASize = batchMatmulConfig["BLOCK_SIZE_K"] * batchMatmulConfig["BLOCK_SIZE_M"] / blockThreads;
    auto ldgBSize = batchMatmulConfig["BLOCK_SIZE_K"] * batchMatmulConfig["BLOCK_SIZE_N"] / blockThreads;
    auto fragASize = batchMatmulConfig["THREAD_SIZE_M"];
    auto fragBSize = batchMatmulConfig["THREAD_SIZE_N"];
    auto vectorize_width = batchMatmulConfig["VECTORIZE_WIDTH"];
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();

    auto tileB = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {ldgBSize}, elementB);
    auto tileA = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {ldgASize}, elementA);

    auto fragB = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {fragBSize}, elementB);
    auto fragA = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {fragASize}, elementA);

    auto smB = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {batchMatmulConfig["BLOCK_SIZE_K"], batchMatmulConfig["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {batchMatmulConfig["BLOCK_SIZE_K"], batchMatmulConfig["BLOCK_SIZE_M"]}, elementA);
    DUMP(module);
    
    auto blockIdx = Rewriter::getParallelIdx(gridLevel);
    auto threadIdx = Rewriter::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder);
    llvm::SmallVector<mlir::Value> operand1{blockIdx[0], threadIdx[0], threadIdx[1], blockIdx[1], k_outer.getInductionVar()};
    auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, operand1, vectorize_width, k_outer, Position::begin);
    auto loadTileBMap = getAffineMap("loadTileB", builder);
    llvm::SmallVector<mlir::Value> operand2{blockIdx[0], threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[2]};
    auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, operand2, vectorize_width, loadTileA, Position::after);
    DUMP(module);

    auto storeTileAMap = getAffineMap("storeTileA", builder);
    auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, vectorize_width, loadTileB, Position::after);
    auto storeTileBMap = getAffineMap("storeTileB", builder);
    auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, vectorize_width, storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);

    DUMP(module);

    auto loadFragAMap = getAffineMap("loadFragA", builder);
    auto loadFragA = Rewriter::read(smA, fragA, loadFragAMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, vectorize_width, k_inner, Position::begin);
    auto loadFragBMap = getAffineMap("loadFragB", builder);
    auto loadFragB = Rewriter::read(smB, fragB, loadFragBMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, vectorize_width, loadFragA, Position::after);
    DUMP(module);

    Rewriter::cache_read(k_inner, A, fragA, getAffineMap("cacheReadA", builder), {m_inner.getInductionVar()});
    Rewriter::cache_read(k_inner, B, fragB, getAffineMap("cacheReadB", builder), {n_inner.getInductionVar()});
    DUMP(module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {vectorize_width});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {vectorize_width});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
    DUMP(module);

    Rewriter::cache_write(m_inner_0, C, C, getAffineMap("cacheWriteC", builder), 
                          {blockIdx[0], threadIdx[0], threadIdx[1], blockIdx[1], blockIdx[2], m_inner_0.getInductionVar(),
                          n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), n_inner_1.getInductionVar()});
    DUMP(module);

    Rewriter::vectorize(n_inner_1, vectorize_width);
    DUMP(module);

    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    DUMP(module);

    Rewriter::detach_last_loop(k_inner);
    DUMP(module);

    Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
    Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
    Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
    Rewriter::schedule(doubleLoadTileB[1][0], doubleLoadTileA[1][0], Position::after);
    Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
    Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
    auto ifOp = doubleLoadTileA[1][1]->getParentOp();
    Rewriter::schedule(ifOp, k_inner, Position::after); 
    Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
    Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
    Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
    Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
    DUMP(module);

    Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
    Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);;
    DUMP(module);

    Rewriter::take_off_true_if(module);
    Rewriter::delete_false_if(module);
    DUMP(module);
  
    int64_t threshold = std::max(batchMatmulConfig["BLOCK_SIZE_K"], std::max(batchMatmulConfig["THREAD_SIZE_M"], batchMatmulConfig["THREAD_SIZE"]));
    Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times >= std::min<int64_t>(threshold, vectorize_width)) return false;
      return true;
    });
    DUMP(module);

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      int step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times > threshold) return false;
      return true;
    });
    Rewriter::deleteExtraCstOp(gridLevel);
    DUMP(module);
  }
}
}