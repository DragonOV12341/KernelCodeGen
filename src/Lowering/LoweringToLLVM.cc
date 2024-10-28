#include "Lowering/LowerToLLVM.h"

using namespace mlir;
namespace KernelCodeGen {

void LoweringToLLVMPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&getContext());

  target.addLegalDialect<
    scf::SCFDialect, 
    arith::ArithDialect,
    // vector::VectorDialect
    gpu::GPUDialect 
    >();
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateAffineToVectorConversionPatterns(patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  // populateSCFToControlFlowConversionPatterns(patterns);
  // mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerToLLVMPass() {
  return std::make_unique<LoweringToLLVMPass>();
}


void ArithCFLoweringToLLVMPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&getContext());

  target.addLegalDialect<gpu::GPUDialect>();
  RewritePatternSet patterns(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);  // cf ->llvm  （这个转不过去，应该是类型不匹配）

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createArithCFLowerToLLVMPass() {
  return std::make_unique<ArithCFLoweringToLLVMPass>();
}

}