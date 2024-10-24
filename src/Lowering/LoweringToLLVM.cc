#include "Lowering/LowerToLLVM.h"

using namespace mlir;
namespace KernelCodeGen {

void LoweringToLLVMPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateAffineToVectorConversionPatterns(patterns);
  // populateSCFToControlFlowConversionPatterns(patterns);
  target.addLegalDialect<
                          scf::SCFDialect, vector::VectorDialect, gpu::GPUDialect>();
  // target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
  //                     scf::ExecuteRegionOp>();
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
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

}