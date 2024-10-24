#pragma once

#include "IR/IR.h"
#include "Optimizer/Analyzer.h"
#include "enum.h"

#include "mlir/Support/LLVM.h"

#include <vector>

namespace KernelCodeGen {

struct Rewriter {
  Rewriter() = default;

  static mlir::OpBuilder getBuilder(mlir::affine::AffineForOp op, Position pos) {
    switch (pos) {
      case Position::after: {
        mlir::OpBuilder builder(op->getContext());
        builder.setInsertionPointAfter(op);
        return builder;
      }
      case Position::before: {
        mlir::OpBuilder builder(op);
        return builder;
      }
      case Position::begin: {
        return mlir::OpBuilder::atBlockBegin(op.getBody());
      }
      case Position::end: {
        return mlir::OpBuilder::atBlockEnd(op.getBody());
      }
      default:
        assert(false);
    } 
  }

  static std::vector<mlir::Value> getParallelIdx(mlir::affine::AffineParallelOp parallelLevel) {
    auto dim = parallelLevel.getNumDims();
    std::vector<mlir::Value> idxes;
    auto ivs = parallelLevel.getIVs();
    for (auto iv : ivs) {
      idxes.push_back(iv);
    }
    return idxes;
  }

  static std::vector<mlir::Value> getElementIdx(mlir::affine::AffineParallelOp parallelLevel) {
    std::vector<mlir::Value> idxes;
    auto ivs = parallelLevel.getIVs();
    for (auto iv : ivs) {
      auto users = iv.getUsers();
      for (auto user : users) {
        if (auto mapOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(user)) {
          idxes.push_back(mapOp.getResult());
          break;
        }
      }
    }
    return idxes;
  }

  /// @brief 
  /// @param forOp 
  /// @param num_output 
  /// @param factors 
  /// @return 
  static std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, 
                                              uint64_t num_output, std::vector<int64_t>&& factors);
  
  /// @brief 
  /// @param loops 
  /// @return 
  static mlir::Value bufferizeLoopCarryVar(std::vector<mlir::affine::AffineForOp>& loops);

  /// @brief 
  /// @param forOp 
  static void reorder(const std::vector<mlir::affine::AffineForOp>& forOp);

  /// @brief 
  /// @param forOp 
  /// @return 
  static mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp>& forOp);

  /// @brief 
  /// @param parallelLevel 
  /// @param ms 
  /// @param shape 
  /// @param dtype 
  /// @return 
  template<typename ParentOpType>
  static mlir::Value alloc_buffer(ParentOpType father, MemorySpace ms, 
                          const std::vector<int64_t> shape_, mlir::Type dtype) {
    llvm::ArrayRef<int64_t> shape (shape_);
    mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));
    
    mlir::OpBuilder builder(father->getContext());
    builder.setInsertionPointToStart(father.getBody());
    return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
  }

  template<typename ContextOp>
  static mlir::Value alloc_buffer(ContextOp contextOp, Position pos, MemorySpace ms, 
                          const std::vector<int64_t> shape_, mlir::Type dtype) {
    llvm::ArrayRef<int64_t> shape (shape_);
    mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));
    
    switch (pos) {
      case Position::before: {
        mlir::OpBuilder builder(contextOp);
        return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
      }
      case Position::after: {
        mlir::OpBuilder builder(contextOp->getContext());
        builder.setInsertionPointAfter(contextOp);
        return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
      }
      default: {
        assert(false);
      }
    }
  }

  /// @brief 
  /// @param src 
  /// @param dst 
  /// @param map 
  /// @param operands 
  /// @param width 
  /// @param compute_at 
  /// @param pos 
  /// @return 
  static mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::affine::AffineForOp compute_at, Position pos);

  static mlir::affine::AffineForOp read(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

  /// @brief 
  /// @param src 
  /// @param dst 
  /// @param map 
  /// @param operands 
  /// @param width 
  /// @param compute_at 
  /// @param pos 
  /// @return 
  static mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::affine::AffineForOp compute_at, Position pos);

  static mlir::affine::AffineForOp write(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

  /// @brief 
  /// @param compute_at 
  /// @param pos 
  /// @return 
  static mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos);

  /// @brief 
  /// @param readOrWrite 
  /// @param width 
  /// @return 
  static mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width);

  /// @brief 
  /// @param scope 
  /// @param src 
  /// @param cached 
  /// @param map 
  /// @param operands 
  static void cache_read(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

  /// @brief 
  /// @param scope 
  /// @param src 
  /// @param cached 
  /// @param map 
  /// @param operands 
  static void cache_write(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

  /// @brief 
  /// @param parallelLevel 
  /// @param dst 
  /// @return 
  static std::vector<std::vector<mlir::affine::AffineForOp>> get_write(mlir::affine::AffineParallelOp parallelLevel, mlir::Value dst);


  /// @brief double buffer for `buffer`, and pipeline for `readBody`. all of them are computead at `compute_at`
  /// @param readBody 
  /// @param buffer 
  /// @param compute_at 
  /// @return
  static std::vector<std::vector<mlir::affine::AffineForOp>> pipeline(std::vector<mlir::affine::AffineForOp> readBodys, mlir::Value& buffer, mlir::affine::AffineForOp compute_at);

  static void change_double_buffer(mlir::affine::AffineForOp, mlir::Value buffer);

  /// @brief 
  static void detach_last_loop(mlir::affine::AffineForOp forOp);

  /// @brief 
  /// @param srcOp 
  /// @param dstOp 
  /// @param pos 
  static void schedule(mlir::Operation* srcOp, mlir::Operation* dstOp, Position pos);


  /// @brief 
  /// @param srcOp 
  /// @param dstOp 
  /// @param pos 
  static void extract_loop(mlir::Operation* srcOp, mlir::affine::AffineForOp forOp, int64_t iteration);

  /// @brief 
  /// @param op 
  static void take_off_true_if(mlir::ModuleOp module);

  /// @brief 
  /// @param op 
  static void delete_false_if(mlir::ModuleOp module);

  /// @brief 
  /// @param forOp 
  static void unroll(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn);

  /// @brief 
  /// @param forOp 
  static void unrollAttribute(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn);

  /// @brief 
  /// @param module 
  static void loweringAffineDialect(mlir::ModuleOp module);

  static void set_buffer(mlir::OpBuilder& builder, mlir::Value mem, mlir::Value targetValue);

  static mlir::affine::AffineForOp create_constant_loop(mlir::OpBuilder& builder, int64_t lowerBound, int64_t upperBound, int64_t step);

  static mlir::affine::AffineForOp outer_product(mlir::OpBuilder& builder, mlir::Value tileC, 
    mlir::Value fragA, mlir::Value fragB, int64_t m, int64_t n);

  /*----------------------------------------------------------------*/
  
  static std::vector<mlir::affine::AffineForOp> combineToTowDim(std::vector<mlir::affine::AffineForOp> loops);


  static mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                  llvm::SmallVector<mlir::Value> operands, mlir::affine::AffineForOp compute_at, Position pos);


  static mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                  llvm::SmallVector<mlir::Value> operands, mlir::affine::AffineForOp compute_at, Position pos);

  static mlir::affine::AffineIfOp irregularMat(mlir::affine::AffineForOp forOp, std::vector<int> range, llvm::SmallVector<mlir::Value> operands);

  static mlir::affine::AffineForOp combineToOneDim(std::vector<mlir::affine::AffineForOp> loops);

  static mlir::Value bufferizeLoopCarryVar(mlir::affine::AffineForOp &loop, mlir::Block* buildBlock);

  static void swapLoops(std::vector<std::vector<mlir::affine::AffineForOp>> loops);

  static void bufferizeOpResult(mlir::Operation* resultOp, mlir::Value buffer);
  
  static void scheduleOpGridToBlock(mlir::affine::AffineParallelOp gridLevel, mlir::affine::AffineParallelOp blockLevel);
  
  static void deleteExtraCstOp(mlir::affine::AffineParallelOp blockLevel);

  static mlir::affine::AffineForOp modifyLoopStepToOne(mlir::affine::AffineForOp forOp);

  static std::vector<mlir::Value> blockLevelOneToTwo(mlir::affine::AffineParallelOp pal, int64_t oneDimLen);

};

}