#include "Frontend/Operators.h"

namespace KernelCodeGen {

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype) {
  if(dtype == "float32") return builder.getF32Type();
  if(dtype == "float64") return builder.getF64Type();
  if(dtype == "float16") return builder.getF16Type();
  if(dtype == "int64") return builder.getIntegerType(64);
  if(dtype == "int32") return builder.getIntegerType(32);
  if(dtype == "int16") return builder.getIntegerType(16);
  if(dtype == "index") return builder.getIndexType();
  if(dtype == "bool") return builder.getIntegerType(1);
  return nullptr;
}

std::string toStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"float16"};
  if(type.isa<mlir::Float32Type>()) return {"float32"};
  if(type.isa<mlir::Float64Type>()) return {"float64"};
  if(auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    if (int_type.getWidth() == 1) return {"bool"};
    else if (int_type.getWidth() == 16) return {"int16"};
    else if (int_type.getWidth() == 32) return {"int32"};
    else if (int_type.getWidth() == 64) return {"int64"};
  }
  if(type.isa<mlir::IndexType>()) return {"index"};
  return nullptr;
}

mlir::affine::AffineForOp buildAffineLoopNest_(mlir::OpBuilder &builder, mlir::Location loc, llvm::ArrayRef<int64_t> lbs, llvm::ArrayRef<int64_t> ubs, 
                                        llvm::ArrayRef<int64_t> steps, mlir::ValueRange iterArgs, loopfunc bodyBuilderFn) {

  assert(lbs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(lbs.size() == steps.size() && "Mismatch in number of arguments");
  mlir::OpBuilder::InsertionGuard guard(builder);

  mlir::affine::AffineForOp main_loop;
  mlir::affine::AffineForOp out_loop = nullptr;
  llvm::SmallVector<mlir::Value, 4> ivs;
  ivs.reserve(lbs.size());
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    auto loopBody = [&](mlir::OpBuilder &builder_, mlir::Location loc_, mlir::Value iv_, mlir::ValueRange iterArgs_) {
      ivs.push_back(iv_);
      if (i == e - 1 && bodyBuilderFn) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder_);
        auto result = bodyBuilderFn(builder_, loc_, ivs, iterArgs_);
        builder_.create<mlir::affine::AffineYieldOp>(loc_, result);
      } else {
        builder_.create<mlir::affine::AffineYieldOp>(loc_, iterArgs);
      }
    };
    mlir::affine::AffineForOp loop;
    if (!out_loop) {
      loop = builder.create<mlir::affine::AffineForOp>(loc, lbs[i], ubs[i], steps[i], iterArgs, loopBody);
      main_loop = loop;
    } else {
      loop = builder.create<mlir::affine::AffineForOp>(loc, lbs[i], ubs[i], steps[i], out_loop.getRegionIterArgs()[0], loopBody);
      out_loop.getBody()->back().erase();
      builder.setInsertionPointToEnd(out_loop.getBody());
      builder.create<mlir::affine::AffineYieldOp>(loc, loop.getResult(0));
    }
    out_loop = loop;
    builder.setInsertionPointToStart(loop.getBody());
  }
  return main_loop;
}

mlir::func::FuncOp buildFuction(mlir::ModuleOp module, mlir::OpBuilder& builder, const std::string& funcName, 
                                const std::vector<mlir::Type>& inputsTypes, const std::vector<mlir::Type>& outputsTypes) {
  mlir::func::FuncOp result;
  bool break_ = false;
  
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp func) {
    if (break_) return;
    // auto otherName = func.getFunctionTypeAttrName();
    auto otherName = func.getSymName();
    if (otherName == funcName) {
      // Function already exists;
      result = func;
      break_ = true;
    }
  });
  if (break_) return result;


  builder.setInsertionPointToStart(module.getBody());
  
  llvm::ArrayRef<mlir::Type> inputsTypesArray(inputsTypes);
  llvm::ArrayRef<mlir::Type> outputsTypesArray(outputsTypes);
  auto functionType = builder.getFunctionType(mlir::TypeRange(inputsTypesArray), 
    mlir::TypeRange(outputsTypesArray));

  auto funcOp = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), llvm::StringRef(funcName), functionType);

  auto& region = funcOp->getRegion(0);
  if (!region.hasOneBlock()) {
    region.emplaceBlock();
  }
  auto& body =  funcOp.front(); //? region.front()  : ;

  int nums = static_cast<int>(inputsTypes.size());
  for (int i = 0; i < nums; i++ ) {
    body.addArguments(inputsTypes[i], builder.getUnknownLoc());
  }

  return funcOp;

}

mlir::Value PlaceHolder::build(ComputeDAG* graph, const std::vector<int64_t>& shapes, const std::string& dtype) {
  auto builder = graph->builder;
  auto dtype_ = getDType(builder, dtype);
  auto tType = mlir::MemRefType::get(shapes, dtype_, {}, static_cast<int>(MemorySpace::global));
  auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tType);
  return allocOp.getResult();
}

mlir::Value Matmul::build(ComputeDAG* graph, mlir::Value A, mlir::Value B/*, MemorySpace ms*/, const std::string& dtype_) {
  
  auto builder = graph->builder;
  auto typeA = A.getType();
  auto typeB = B.getType();
  int64_t m {-1}, n {-1}, k1{-1}, k2{-1};
  mlir::Attribute memorySpace;
  mlir::Type elementTypeA;

  if(typeA.isa<mlir::MemRefType>()) {
    auto shapeA = typeA.dyn_cast<mlir::MemRefType>();
    m = shapeA.getShape()[0];
    k1 = shapeA.getShape()[1];
    elementTypeA = shapeA.getElementType();
    memorySpace = shapeA.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of left operand of Matmul is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementTypeA);

  if(typeB.isa<mlir::MemRefType>()) {
    auto shapeB = typeB.dyn_cast<mlir::MemRefType>();
    k2 = shapeB.getShape()[0];
    n = shapeB.getShape()[1];
  }
  else {
    llvm::errs() << "Type of right operand of Matmul is not Memref.\n";
    return nullptr;
  }

  if (k1 != k2) {
    llvm::errs() << 
      "Can't apply Matmul Operation due to imcompatible K-dim.\n";
    return nullptr;
  }

  auto funcName = std::string({"Matmul_m"}) + std::to_string(m) + "n" + std::to_string(n) +  "k" + std::to_string(k1);

  auto emType = getDType(builder, dtype);
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(std::vector<int64_t>{m, n}), emType, {}, static_cast<int>(MemorySpace::global));

  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {typeA, typeB}, {typeC});
  // auto& bodyBlock = funcOp.getBody().front(); // the same
  auto& bodyBlock = funcOp.front();

  if (bodyBlock.getOperations().size() > 0) {
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
    return callOp.getResult(0);
  } 

  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();

  mlir::Value output;
  auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
  output = allocOp.getResult();

  // void buildAffineLoopNest(OpBuilder &builder, Location loc,
  //                         ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs,
  //                         ArrayRef<int64_t> steps,
  //                         function_ref<void(OpBuilder &, Location, ValueRange)>
  //                             bodyBuilderFn = nullptr);
  mlir::SmallVector<int64_t, 3> lowerBounds(2, /*Value=*/0);
  mlir::SmallVector<int64_t, 3> steps(2, /*Value=*/1);
  mlir::SmallVector<int64_t, 3> upperBounds({m, n});
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto i = ivs[0];
      auto j = ivs[1];
      // FloatAttr Builder::getFloatAttr(Type type, double value) {
      //   return FloatAttr::get(type, value);
      // }
      // initilize to 0
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), nestedBuilder.getFloatAttr(emType, 0));

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        auto k = iv;
        auto ld_a = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), /*A*/operands[0], mlir::ValueRange({i, k}));
        auto ld_b = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), /*B*/operands[1], mlir::ValueRange({k, j}));
        auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::affine::AffineForOp>(nestedBuilder.getUnknownLoc(), 0, k1, 1, mlir::ValueRange({zero.getResult()}), kLoopBody);

      nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), Cij.getResult(0), /*C*/output, mlir::ValueRange({i, j}));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);


  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

mlir::Value Relu::build(ComputeDAG* graph, mlir::Value input, MemorySpace ms, const std::string& dtype_) {
  
  auto builder = graph->builder;
  auto type = input.getType();

  mlir::Attribute memorySpace;
  mlir::Type elementType;

  llvm::ArrayRef<int64_t> shape;

  if(type.isa<mlir::MemRefType>()) {
    auto type_ = type.dyn_cast<mlir::MemRefType>();
    shape = type_.getShape();
    elementType = type_.getElementType();
    memorySpace = type_.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of input of Relu is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

  auto funcName = std::string({"Relu_Elementwise"});

  for (auto dim : shape) {
    funcName += "_" + std::to_string(dim);
  }

  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {input.getType()});
  
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();

  auto emType = getDType(builder, dtype);

  mlir::Value output;
  if (ms != MemorySpace::inplace) {
    auto typeC = mlir::MemRefType::get(shape, emType, {}, static_cast<int>(ms));
    auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
    output = allocOp.getResult();
  } else {
    output = operands[0];
  }

  mlir::SmallVector<int64_t, 8> lowerBounds(shape.size(), /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(shape.size(), /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shape.begin(), shape.end());
  mlir::affine::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {

      // initilize to 0
      auto dtypeOutput = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeOutput, 0));
      auto ld_element = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], ivs);
      auto max = nestedBuilder.create<mlir::arith::MaxNumFOp>(nestedBuilder.getUnknownLoc(), zero, ld_element);
      nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), max, output, ivs);
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

mlir::Value BatchedMatmul::build(ComputeDAG* graph, mlir::Value A, Layout layoutA, mlir::Value B, Layout layoutB, const std::string& dtype_) {
  auto builder = graph->builder;
  auto typeA = A.getType();
  auto typeB = B.getType();
  int64_t m {-1}, n {-1}, k1{-1}, k2{-1};
  mlir::Attribute memorySpace;
  mlir::Type elementTypeA;

  int totalDims = -1;
  std::vector<int64_t> shapeC;

  if(typeA.isa<mlir::MemRefType>()) {
    auto mrTypeA = typeA.dyn_cast<mlir::MemRefType>();
    auto shapeA = mrTypeA.getShape();
    totalDims = shapeA.size();
    if (totalDims < 2) {
      llvm::errs() << "BatchedMatmul needs at least 2 dim but got " << totalDims << "\n";
      exit(EXIT_FAILURE);
    }
    shapeC.reserve(totalDims);
    for(auto dim : shapeA) {
      shapeC.push_back(dim);
    }
    if (layoutA == Layout::rowMajor) {
      m = shapeA[totalDims - 2];
      k1 = shapeA[totalDims - 1];
    } else {
      m = shapeA[totalDims - 1];
      k1 = shapeA[totalDims - 2];      
    }

    elementTypeA = mrTypeA.getElementType();
    memorySpace = mrTypeA.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of left operand of BatchedMatmul is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementTypeA);

  if(typeB.isa<mlir::MemRefType>()) {
    auto mrTypeB = typeB.dyn_cast<mlir::MemRefType>();
    auto shapeB = mrTypeB.getShape();
    if (totalDims != shapeB.size()) {
      llvm::errs() << "BatchedMatmul: A, B dim not matched.\n";
    }

    if (layoutB == Layout::colMajor) {
      k2 = shapeB[totalDims - 1];
      n = shapeB[totalDims - 2];
    } else {
      k2 = shapeB[totalDims - 2];
      n = shapeB[totalDims - 1];
    }
    shapeC[totalDims - 1] = n;
  }
  else {
    llvm::errs() << "Type of right operand of BatchedMatmul is not Memref.\n";
    return nullptr;
  }

  if (k1 != k2) {
    llvm::errs() << 
      "Can't apply BatchedMatmul Operation due to imcompatible K-dim.\n";
    return nullptr;
  }


  // Create C buffer as the result.
  auto C = graph->create<PlaceHolder>(shapeC, dtype);

  C.getDefiningOp()->moveAfter(B.getDefiningOp());

  int batch_dim_num = shapeC.size() - 2;

  auto funcName = std::string({"BatchMatmul"});

  for (int i = 0; i < batch_dim_num; i ++) {
    funcName += "_";
    funcName += std::to_string(shapeC[i]);
  }
  char transposeA = layoutA == Layout::rowMajor ? 'N' : 'T';
  char transposeB = layoutB== Layout::rowMajor ? 'N' : 'T';
  funcName += "_m" + std::to_string(shapeC[batch_dim_num]) + 
              "_n" + std::to_string(shapeC[batch_dim_num + 1]) +  
              "_k" + std::to_string(k1) + "_" + transposeA + transposeB;

  auto emType = getDType(builder, dtype);
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shapeC), 
    emType, {}, static_cast<int>(MemorySpace::global));

  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {typeA, typeB, typeC}, {typeC});
  // auto& bodyBlock = funcOp.getBody().front(); // the same
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);

  mlir::ValueRange operands = bodyBlock.getArguments();

  mlir::SmallVector<int64_t, 8> lowerBounds(totalDims, /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(totalDims, /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shapeC.begin(), shapeC.end());
  mlir::affine::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto i = ivs[totalDims - 2];
      auto j = ivs[totalDims - 1];
      // FloatAttr Builder::getFloatAttr(Type type, double value) {
      //   return FloatAttr::get(type, value);
      // }
      // initilize to 0
      auto dtypeC = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeC, 0));
      
      std::vector<mlir::Value> indexA;
      std::vector<mlir::Value> indexB;
      std::vector<mlir::Value> indexC;

      // fill with batch dimension.
      int counter = 0;
      for (auto iv : ivs) {
        if (counter++ < totalDims - 2) {
          indexA.push_back(iv); indexB.push_back(iv); indexC.push_back(iv);
        } else {
          break;
        }
      }
      indexC.push_back(i); indexC.push_back(j);

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        auto k = iv;
        if (layoutA == Layout::rowMajor) {
          indexA.push_back(i); indexA.push_back(k);
        } else {
          indexA.push_back(k); indexA.push_back(j);
        }

        if (layoutB == Layout::rowMajor) {
          indexB.push_back(k); indexB.push_back(j);
        } else {
          indexB.push_back(j); indexB.push_back(k);
        }

        auto ld_a = builder.create<mlir::affine::AffineLoadOp>(
                      builder.getUnknownLoc(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(indexA)));
        auto ld_b = builder.create<mlir::affine::AffineLoadOp>(
                      builder.getUnknownLoc(), operands[1], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(indexB)));
        auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::affine::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, k1, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({zero.getResult()}), kLoopBody);

      nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), 
          Cij.getResult(0), operands[2], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(indexC)));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), operands[2]);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B, C}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

mlir::Value Transpose::build(ComputeDAG* graph, mlir::Value input) {
}

mlir::Value Softmax::build(ComputeDAG* graph, mlir::Value input, int axis, MemorySpace ms, const std::string& dtype_) {
  auto builder = graph->builder;
  auto type = input.getType();

  mlir::Attribute memorySpace;
  mlir::Type elementType;

  llvm::ArrayRef<int64_t> shape;

  if(type.isa<mlir::MemRefType>()) {
    auto type_ = type.dyn_cast<mlir::MemRefType>();
    shape = type_.getShape();
    elementType = type_.getElementType();
    memorySpace = type_.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of input of Softmax is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

  int totalDims = shape.size();

  if (axis < 0 || axis >= totalDims) {
    llvm::errs() << "Illegal reduction axis in Softmax.\n";
  }
  auto reduceStartAxis = axis == -1 ? totalDims - 1 : axis;

  auto funcName = std::string({"Softmax"});

  for (int i = 0; i < totalDims; i ++) {
    funcName += "_";
    funcName += std::to_string(shape[i]);
  }
  funcName += "_axis" + std::to_string(reduceStartAxis);


  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {input.getType()});
  // auto& bodyBlock = funcOp.getBody().front(); // the same
  auto& bodyBlock = funcOp.front();
  builder.setInsertionPointToStart(&bodyBlock);

  mlir::ValueRange operands = bodyBlock.getArguments();

  mlir::SmallVector<int64_t, 8> lowerBounds(reduceStartAxis, /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(reduceStartAxis, /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shape.begin(), shape.begin() + reduceStartAxis);
  mlir::affine::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {


      
      // At present, only support reduction on the last dim.
      if (totalDims - reduceStartAxis != 1) {
        llvm::errs() << "At present, only support reduction on the last dim.\n";
      }

      // initilize to 0
      auto dtypeOutput = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeOutput, 0));
      
      std::vector<mlir::Value> index;
      index.reserve(totalDims);

      // fill with batch dimension.
      int counter = 0;
      for (auto iv : ivs) {
        if (counter++ < totalDims - 1) {
          index.push_back(iv);
        } else {
          break;
        }
      }
      
      // Reduction.
      auto kLoopBody = [&](mlir::OpBuilder &kBuilder, mlir::Location kLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard kGuard(kBuilder);
        auto k = iv;
        index.push_back(k);
        auto ld = kBuilder.create<mlir::affine::AffineLoadOp>(
                      kBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(index)));
        auto exp = kBuilder.create<mlir::math::ExpOp>(kBuilder.getUnknownLoc(), ld.getResult());
        auto add = kBuilder.create<mlir::arith::AddFOp>(kBuilder.getUnknownLoc(), exp.getResult(), iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto sum = nestedBuilder.create<mlir::affine::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, shape.back(), 1, /*iterArgs=lvm::None*/ mlir::ValueRange({zero.getResult()}), kLoopBody);

      // Elementwise.
      auto ewLoopBody = [&](mlir::OpBuilder &ewBuilder, mlir::Location ewLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard kGuard(ewBuilder);
        auto ew = iv;
        index.back() = ew;
        auto ld = ewBuilder.create<mlir::affine::AffineLoadOp>(
                      ewBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(index)));
        auto exp = ewBuilder.create<mlir::math::ExpOp>(ewBuilder.getUnknownLoc(), ld.getResult());
        auto div = ewBuilder.create<mlir::arith::DivFOp>(ewBuilder.getUnknownLoc(), exp.getResult(), sum.getResult(0));
        
        ewBuilder.create<mlir::affine::AffineStoreOp>(ewBuilder.getUnknownLoc(), 
            div.getResult(), operands[0], mlir::ValueRange(llvm::ArrayRef<mlir::Value>(index)));
        ewBuilder.create<mlir::affine::AffineYieldOp>(ewBuilder.getUnknownLoc());
      };
      nestedBuilder.create<mlir::affine::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, shape.back(), 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), ewLoopBody);
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), operands[0]);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

/*------------------------------------Binary--------------------------------------*/
mlir::Value Binary::add(mlir::OpBuilder& builder, mlir::Value elem_1, mlir::Value elem_2) {
  return builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), elem_1, elem_2);
}

mlir::Value Binary::mul(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2){
  return builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), elem_1, elem_2);
}

mlir::Value Binary::div(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2){
  return builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), elem_1, elem_2);
}

mlir::Value Binary::sub(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2){
  return builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), elem_1, elem_2);
}

mlir::Value Binary::pow(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2){
  return builder.create<mlir::math::PowFOp>(builder.getUnknownLoc(), elem_1, elem_2);
}

mlir::Value Binary::equal(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2){
  return builder.create<mlir::arith::CmpFOp>(builder.getUnknownLoc(), mlir::arith::CmpFPredicate::OEQ, elem_1, elem_2);
}

mlir::Value Binary::greater(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2){
  return builder.create<mlir::arith::CmpFOp>(builder.getUnknownLoc(), mlir::arith::CmpFPredicate::OGT, elem_1, elem_2);
}

std::map<std::string, std::function<mlir::Value(mlir::OpBuilder&, mlir::Value, mlir::Value)>> Binary::operationMap = {
    {"Add", &Binary::add}, {"Mul", &Binary::mul}, {"Div", &Binary::div}, {"Sub", &Binary::sub},
    {"Pow", &Binary::pow}, {"Equal", &Binary::equal}, {"Greater", &Binary::greater}
  };
/*-------------------------------------------------------------------------------------*/

// mlir::Value Binary::build(ComputeDAG* graph, mlir::Value A, mlir::Value B, std::string operation, MemorySpace ms, const std::string& dtype_) {
//   auto builder = graph->builder;
//   auto typeA = A.getType();
//   auto typeB = B.getType();

//   llvm::ArrayRef<int64_t> input_shapeA, input_shapeB;
//   mlir::Type elementType;
//   if(typeA.isa<mlir::MemRefType>()) {
//     auto shapeA = typeA.dyn_cast<mlir::MemRefType>();
//     input_shapeA = shapeA.getShape();
//     elementType = shapeA.getElementType();
//   }
//   else {
//     llvm::errs() << "Type of left operand of" << operation << "is not Memref.\n";
//     return nullptr;
//   }
//   auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);
//   auto emType = getDType(builder, dtype);

//   if(typeB.isa<mlir::MemRefType>()) {
//     auto shapeB = typeB.dyn_cast<mlir::MemRefType>();
//     input_shapeB = shapeB.getShape();
//   }
//   else {
//     llvm::errs() << "Type of right operand of " << operation << " is not Memref.\n";
//     return nullptr;
//   }

//   size_t max_dim, min_dim;
//   mlir::Value max_tensor, min_tensor;
//   std::vector<int> oneDimIndexs;
//   mlir::Value temp_tensor = nullptr;
//   std::vector<int64_t> max_shape, min_shape;
//   if (input_shapeA.size() > input_shapeB.size()) {  // A dim > B dim
//     max_tensor = A; min_tensor = B;
//     max_shape = std::vector<int64_t>(input_shapeA);
//     min_shape = std::vector<int64_t>(input_shapeB);
//   } else if (input_shapeA.size() < input_shapeB.size()) {  // B dim > A dim
//     max_tensor = B; min_tensor = A;
//     max_shape = std::vector<int64_t>(input_shapeB);
//     min_shape = std::vector<int64_t>(input_shapeA);
//   } else {   // A dim == B dim
//     for (int i=0; i<input_shapeA.size(); i++) {
//       if (input_shapeA[i] != input_shapeB[i]) {
//         if (input_shapeA[i] != 1 && input_shapeB[i] != 1) {
//           llvm::errs() << "A-dim is not equal to B-dim and Can't apply " << operation << " Operation due to imcompatible "<< i <<"-dim.\n";
//           return nullptr;
//         } else {
//           oneDimIndexs.push_back(i);
//           if (input_shapeA[i] == 1) {
//             if (temp_tensor && temp_tensor != A) {
//               llvm::errs() << "A-dim is not equal to B-dim and Can't apply " << operation << " Operation due to imcompatible "<< i <<"-dim.\n";
//               return nullptr;
//             } else temp_tensor = A;
//           } else {
//             if (temp_tensor && temp_tensor != B) {
//               llvm::errs() << "A-dim is not equal to B-dim and Can't apply " << operation << " Operation due to imcompatible "<< i <<"-dim.\n";
//               return nullptr;
//             } else temp_tensor = B;
//           }
//         }
//       }
//     }
//     if (temp_tensor) {
//       if (temp_tensor == A){
//         max_tensor = B; min_tensor = A;
//         max_shape = std::vector<int64_t>(input_shapeB);
//         min_shape = std::vector<int64_t>(input_shapeA);
//       } else {
//         max_tensor = A; min_tensor = B;
//         max_shape = std::vector<int64_t>(input_shapeA);
//         min_shape = std::vector<int64_t>(input_shapeB);
//       }
//     } else {
//       max_tensor = A; min_tensor = B;
//       max_shape = std::vector<int64_t>(input_shapeA);
//       min_shape = std::vector<int64_t>(input_shapeB);
//     }
//   }
//   max_dim = max_shape.size();
//   min_dim = min_shape.size();
//   if (max_dim != min_dim){
//     int mo = max_dim - min_dim;
//     for (int i=0; i<min_dim; i++) {
//       if (max_shape[i + mo] != min_shape[i]) {
//         if (min_shape[i] != 1) {
//           llvm::errs() << "A-dim is not equal to B-dim and Can't apply " << operation << " Operation due to imcompatible "<< i <<"-dim.\n";
//           return nullptr;          
//         } else
//           oneDimIndexs.push_back(i + mo);
//       }
//     }
//   }

//   auto funcName = std::string({operation + "_Binary"});

//   for (auto dim : max_shape) {
//     funcName += "_" + std::to_string(dim);
//   }
//   funcName += "_" + operation;

//   for (auto dim : min_shape) {
//     funcName += "_" + std::to_string(dim);
//   }

//   auto ip = builder.saveInsertionPoint();
//   auto funcOp = buildFuction(graph->module, builder, funcName, {max_tensor.getType(), min_tensor.getType()}, {max_tensor.getType()});
  
//   auto& bodyBlock = funcOp.front();

//   if (bodyBlock.getOperations().size() > 0) {
//     auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({max_tensor, min_tensor}));
//     funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
//     return callOp.getResult(0);
//   } 

//   builder.setInsertionPointToStart(&bodyBlock);
//   mlir::ValueRange operands = bodyBlock.getArguments();  // 参数
//   mlir::Value output;
//   if (ms != MemorySpace::inplace) {
//     auto typeC = mlir::MemRefType::get(max_shape, emType, {}, static_cast<int>(ms));
//     auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
//     output = allocOp.getResult();
//   } else {
//     output = operands[0];
//   }

//   mlir::SmallVector<int64_t> lowerBounds(max_dim, /*Value=*/0);
//   mlir::SmallVector<int64_t> steps(max_dim, /*Value=*/1);
//   mlir::SmallVector<int64_t> upperBounds(max_shape.begin(), max_shape.end());
//   mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
//     [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
//       mlir::SmallVector<mlir::Value> min_ivs;
//       if (oneDimIndexs.size()) {
//         auto one = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
//         for (int i=max_dim-min_dim; i<ivs.size(); i++) {
//           if (std::find(oneDimIndexs.begin(), oneDimIndexs.end(), i) != oneDimIndexs.end())
//             min_ivs.push_back(one);
//           else 
//             min_ivs.push_back(ivs[i]);
//         }
//       } else {
//         min_ivs = mlir::SmallVector<mlir::Value>(ivs.take_back(min_dim));
//       }
//       auto ld_max = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(ivs));
//       auto ld_min = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[1], mlir::ValueRange(min_ivs));
//       auto result = operationMap[operation](nestedBuilder, ld_max, ld_min);
//       nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), result, output, mlir::ValueRange(ivs));
//     }
//   );
//   builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);
//   builder.restoreInsertionPoint(ip);
//   auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({max_tensor, min_tensor}));
//   funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
//   return callOp.getResult(0);
// }

mlir::Value Binary::build(ComputeDAG* graph, mlir::Value A, mlir::Value B, std::string operation, MemorySpace ms, const std::string& dtype_) {
  auto builder = graph->builder;
  auto typeA = A.getType();
  auto typeB = B.getType();

  llvm::ArrayRef<int64_t> input_shapeA, input_shapeB;
  mlir::Type elementType;
  if(typeA.isa<mlir::MemRefType>()) {
    auto shapeA = typeA.dyn_cast<mlir::MemRefType>();
    input_shapeA = shapeA.getShape();
    elementType = shapeA.getElementType();
  }
  else {
    llvm::errs() << "Type of left operand of" << operation << "is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);
  auto emType = getDType(builder, dtype);

  if(typeB.isa<mlir::MemRefType>()) {
    auto shapeB = typeB.dyn_cast<mlir::MemRefType>();
    input_shapeB = shapeB.getShape();
  }
  else {
    llvm::errs() << "Type of right operand of " << operation << " is not Memref.\n";
    return nullptr;
  }

  std::vector<int64_t> shapeA(input_shapeA.begin(), input_shapeA.end());
  std::vector<int64_t> shapeB(input_shapeB.begin(), input_shapeB.end());
  int minDim = shapeA.size() <= shapeB.size() ? shapeA.size() : shapeB.size();
  int maxDim = shapeA.size() >= shapeB.size() ? shapeA.size() : shapeB.size();
  bool AIsMax = shapeA.size() >= shapeB.size() ? true : false;
  std::reverse(shapeA.begin(), shapeA.end());
  std::reverse(shapeB.begin(), shapeB.end());
  bool hasOneDim = false;
  std::vector<int64_t> newShape;
  for (int i=0; i<minDim; i++) {
    if (shapeA[i] != shapeB[i]){
      if (shapeA[i] != 1 && shapeB[i] != 1) {llvm::errs() << "dim not equal" <<"\n"; return nullptr;}
      else {
        hasOneDim = true;
        if (shapeA[i] == 1) newShape.push_back(shapeB[i]);
        else newShape.push_back(shapeA[i]);
      }
    } else {
      newShape.push_back(shapeA[i]);
    }
  }
  if (AIsMax) newShape.insert(newShape.end(), shapeA.begin()+minDim, shapeA.end());
  else newShape.insert(newShape.end(), shapeB.begin()+minDim, shapeB.end());
  std::reverse(newShape.begin(), newShape.end());

  auto funcName = std::string({operation + "_Binary"});
  for (int i=shapeA.size()-1; i >=0; i--) {
    funcName += "_" + std::to_string(shapeA[i]);
  }
  funcName += "_" + operation;
  for (int i=shapeB.size()-1; i >=0; i--) {
    funcName += "_" + std::to_string(shapeB[i]);
  }

  auto ip = builder.saveInsertionPoint();
  auto typeC = mlir::MemRefType::get(newShape, emType, {}, static_cast<int>(ms));
  auto funcOp = buildFuction(graph->module, builder, funcName, {A.getType(), B.getType()}, {typeC});
  
  auto& bodyBlock = funcOp.front();

  if (bodyBlock.getOperations().size() > 0) {
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
    return callOp.getResult(0);
  } 

  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();  // 参数
  mlir::Value output;
  if (ms != MemorySpace::inplace) {
    auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
    output = allocOp.getResult();
  } else {
    output = operands[0];
  }

  mlir::SmallVector<int64_t> lowerBounds(maxDim, /*Value=*/0);
  mlir::SmallVector<int64_t> steps(maxDim, /*Value=*/1);
  mlir::SmallVector<int64_t> upperBounds(newShape.begin(), newShape.end());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      mlir::SmallVector<mlir::Value> tempIvs(ivs.begin(), ivs.end());
      mlir::SmallVector<mlir::Value> aIvs, bIvs;
      std::reverse(tempIvs.begin(), tempIvs.end());
      std::reverse(newShape.begin(), newShape.end());
      mlir::arith::ConstantOp zero;
      if (hasOneDim)
        zero = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
      for (int i=0; i<shapeA.size(); i++) {
        if (shapeA[i] == newShape[i])  aIvs.push_back(tempIvs[i]);
        else aIvs.push_back(zero);
      }
      for (int i=0; i<shapeB.size(); i++) {
        if (shapeB[i] == newShape[i])  bIvs.push_back(tempIvs[i]);
        else bIvs.push_back(zero);
      }
      std::reverse(aIvs.begin(), aIvs.end());
      std::reverse(bIvs.begin(), bIvs.end());
      auto ld_a = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(aIvs));
      auto ld_b = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[1], mlir::ValueRange(bIvs));
      auto result = operationMap[operation](nestedBuilder, ld_a, ld_b);
      nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), result, output, mlir::ValueRange(ivs));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A, B}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

// mlir::Value Binary::build(ComputeDAG* graph, mlir::Value A, float B, std::string operation, MemorySpace ms, const std::string& dtype_) {
//   auto builder = graph->builder;
//   auto typeA = A.getType();

//   llvm::ArrayRef<int64_t> input_shape;
//   mlir::Type elementType;
//   if(typeA.isa<mlir::MemRefType>()) {
//     auto shape = typeA.dyn_cast<mlir::MemRefType>();
//     input_shape = shape.getShape();
//     elementType = shape.getElementType();
//   }
//   else {
//     llvm::errs() << "Type of left operand of" << operation << "is not Memref.\n";
//     return nullptr;
//   }
//   auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

//   auto funcName = std::string({operation + "_Binary"});

//   for (auto dim : input_shape) {
//     funcName += "_" + std::to_string(dim);
//   }
//   auto cst = std::to_string(B);
//   funcName += "_" + operation + "_cst_" + cst.substr(0, 1) + "_" + cst.substr(2);

//   auto ip = builder.saveInsertionPoint();
//   auto funcOp = buildFuction(graph->module, builder, funcName, {A.getType()}, {A.getType()});
  
//   auto& bodyBlock = funcOp.front();

//   if (bodyBlock.getOperations().size() > 0) {
//     auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A}));
//     funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
//     return callOp.getResult(0);
//   } 

//   builder.setInsertionPointToStart(&bodyBlock);
//   mlir::ValueRange operands = bodyBlock.getArguments();  // 参数

//   auto emType = getDType(builder, dtype);

//   mlir::Value output;
//   if (ms != MemorySpace::inplace) {
//     auto typeC = mlir::MemRefType::get(input_shape, emType, {}, static_cast<int>(ms));
//     auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
//     output = allocOp.getResult();
//   } else {
//     output = operands[0];
//   }

//   mlir::SmallVector<int64_t> lowerBounds(input_shape.size(), /*Value=*/0);
//   mlir::SmallVector<int64_t> steps(input_shape.size(), /*Value=*/1);
//   mlir::SmallVector<int64_t> upperBounds(input_shape.begin(), input_shape.end());
//   mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
//     [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
//       auto ld_max = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(ivs));
//       auto num = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), nestedBuilder.getFloatAttr(getDType(nestedBuilder, dtype), B));
//       auto result = operationMap[operation](nestedBuilder, ld_max, num);
//       nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), result, output, mlir::ValueRange(ivs));
//     }
//   );
//   builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);
//   builder.restoreInsertionPoint(ip);
//   auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({A}));
//   funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
//   return callOp.getResult(0);
// }

/*------------------------------------ElementWise--------------------------------------*/
mlir::Value ElementWise::tanh(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type) {
  return builder.create<mlir::math::TanhOp>(builder.getUnknownLoc(), elem);
}

mlir::Value ElementWise::sqrt(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type) {
  return builder.create<mlir::math::SqrtOp>(builder.getUnknownLoc(), elem);
}

mlir::Value ElementWise::log(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type) {
  return builder.create<mlir::math::LogOp>(builder.getUnknownLoc(), elem);
}

mlir::Value ElementWise::relu(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type) {
  auto zero = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(type, 0));
  return builder.create<mlir::arith::MaxNumFOp>(builder.getUnknownLoc(), zero, elem);
}

mlir::Value ElementWise::cast(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type) {
  return builder.create<mlir::arith::BitcastOp>(builder.getUnknownLoc(), type, elem);
}

mlir::Value ElementWise::gelu(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type) {
  auto one = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(type, 1.0));
  auto dotFive = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(type, 0.5));
  auto sqrtHaftPi = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(type, 0.797884));
  auto cst = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getFloatAttr(type, 0.044715));
  auto temp1 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), elem, dotFive);
  auto temp2 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), elem, elem);
  auto temp3 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), elem, temp2);
  auto temp4 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), cst, temp3);
  auto temp5 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), temp4, elem);
  auto temp6 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), sqrtHaftPi, temp5);
  auto temp7 = builder.create<mlir::math::TanhOp>(builder.getUnknownLoc(), temp6);
  auto temp8 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), temp7, one);
  return builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), temp1, temp8);
}

std::map<std::string, std::function<mlir::Value(mlir::OpBuilder&, mlir::Value, mlir::Type)>> ElementWise::operationMap = {
    {"Tanh", &ElementWise::tanh}, {"Sqrt", &ElementWise::sqrt}, {"Log", &ElementWise::log},
    {"Relu", &ElementWise::relu}, {"Cast", &ElementWise::cast}, {"Gelu", &ElementWise::gelu}
};
/*-------------------------------------------------------------------------------------*/

mlir::Value ElementWise::build(ComputeDAG* graph, mlir::Value input, std::string operation, MemorySpace ms, const std::string& dtype_) {
  auto builder = graph->builder;
  auto type_input = input.getType();

  mlir::Type elementType;
  llvm::ArrayRef<int64_t> input_shape;

  if(type_input.isa<mlir::MemRefType>()) {
    auto shape = type_input.dyn_cast<mlir::MemRefType>();
    input_shape = shape.getShape();
    elementType = shape.getElementType();
  }
  else {
    llvm::errs() << "Type of operand of " << operation << " is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

  auto funcName = std::string({operation + "_Elementwise"});

  for (auto dim : input_shape) {
    funcName += "_" + std::to_string(dim);
  }

  auto ip = builder.saveInsertionPoint();
  mlir::func::FuncOp funcOp;
  if (ms != MemorySpace::inplace) {
    auto emType = getDType(builder, dtype);
    auto typeC = mlir::MemRefType::get(input_shape, emType, {}, static_cast<int>(ms));
    funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {typeC});
  } else {
    funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {input.getType()});
  }
  
  auto& bodyBlock = funcOp.front();

  if (bodyBlock.getOperations().size() > 0) {
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
    return callOp.getResult(0);
  } 

  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();  // 参数

  mlir::Value output;
  if (ms != MemorySpace::inplace) {
    auto emType = getDType(builder, dtype);
    auto typeC = mlir::MemRefType::get(input_shape, emType, {}, static_cast<int>(ms));
    auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
    output = allocOp.getResult();
  } else {
    output = operands[0];
  }

  mlir::SmallVector<int64_t> lowerBounds(input_shape.size(), /*Value=*/0);
  mlir::SmallVector<int64_t> steps(input_shape.size(), /*Value=*/1);
  mlir::SmallVector<int64_t> upperBounds(input_shape.begin(), input_shape.end());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto ld_elem = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(ivs));
      auto result = operationMap[operation](nestedBuilder, ld_elem, getDType(builder, dtype));
      nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), result, output, mlir::ValueRange(ivs));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);

  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

mlir::Value LayerNorm::build(ComputeDAG* graph, mlir::Value input, mlir::Value scale, mlir::Value bias,
                                          int64_t axis, const float &eps, MemorySpace ms, const std::string& dtype_) {
  auto builder = graph->builder;
  auto type_input = input.getType();
  auto scaleType = scale.getType();
  auto biasType = bias.getType();
  llvm::ArrayRef<int64_t> input_shape;
  mlir::Type elementType;
  if(type_input.isa<mlir::MemRefType>()) {
    auto shape = type_input.dyn_cast<mlir::MemRefType>();
    input_shape = shape.getShape();
    elementType = shape.getElementType();
  } else {
    llvm::errs() << "Type of operand of LayerNorm is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);
  auto emType = getDType(builder, dtype);

  auto scaleType_ = scaleType.dyn_cast<mlir::MemRefType>();
  auto biasType_ = biasType.dyn_cast<mlir::MemRefType>();
  auto scaleShape = scaleType_.getShape();
  auto biasShape = biasType_.getShape();
  /*-----------------scale bias合法性检查------------------*/
  bool hasOneDim = false;
  for (int i=0; i<scaleShape.size(); i++) {
    if (scaleShape[scaleShape.size()-1-i] != input_shape[input_shape.size()-1-i]) {
      if (scaleShape[scaleShape.size()-1-i] != 1) {
        llvm::errs() << "scale dim is not equal.\n";
        return nullptr;
      } else {
        hasOneDim = true;
      }
    }
  }
  for (int i=0; i<biasShape.size(); i++) {
    if (biasShape[biasShape.size()-1-i] != input_shape[input_shape.size()-1-i]) {
      if (biasShape[biasShape.size()-1-i] != 1) {
        llvm::errs() << "bias dim is not equal.\n";
        return nullptr;
      } else {
        hasOneDim = true;
      }
    }
  }

  /*----------------检查合法性，获取关键数据------------------*/
  int64_t dim = input_shape.size();
  if (axis >= dim || axis <= dim*-1) {  // 指定维度超出范围
    llvm::errs() << "axis is out of bounds for array of dimension.\n";
    return nullptr;
  }
  if (axis < 0) axis += dim;  // 将负数变成正数
  int in_num = 0, out_num = 0, elemNum = 1;
  std::vector<int64_t> in_shape, out_shape;
  for (int i=0; i<dim; i++) {
    if (i < axis) {
      out_num++;
      out_shape.push_back(input_shape[i]);
    } else {
      in_num++;
      elemNum *= input_shape[i];
      in_shape.push_back(input_shape[i]);
    }
  }
  /*----------------合成循环变量func------------------*/
  auto getIndexArgs = [&](mlir::ValueRange inIvs, mlir::ValueRange outIvs) {
    mlir::SmallVector<mlir::Value> ivs;
    ivs.append(outIvs.begin(), outIvs.end());
    ivs.append(inIvs.begin(), inIvs.end());
    return ivs;
  };
  /*----------------设置函数名称------------------*/
  auto funcName = std::string({"LayerNorm"});
  for (auto shape : input_shape) {
    funcName += "_" + std::to_string(shape);
  }
  funcName += "_axis_" + std::to_string(axis);
  /*----------------func返回值及参数列表-----------------*/
  std::vector<mlir::Type> inputTypes({type_input, scaleType, biasType});
  std::vector<mlir::Type> outputTypes({type_input});
  /*---------------建立funcop-------------------*/
  auto ip = builder.saveInsertionPoint();
  auto funcOp = buildFuction(graph->module, builder, funcName, inputTypes, outputTypes);
  auto& bodyBlock = funcOp.front();

  if (bodyBlock.getOperations().size() > 0) {
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input, scale, bias}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
    return callOp.getResult(0);
  } 
  /*----------------设置output是否直接是input------------------*/
  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();  // 参数
  mlir::Value output;
  if (ms != MemorySpace::inplace) {
    auto typeC = mlir::MemRefType::get(input_shape, emType, {}, static_cast<int>(ms));
    auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
    output = allocOp.getResult();
  } else {
    output = operands[0];
  }

  mlir::SmallVector<int64_t> outLowerBounds(out_num, /*Value=*/0);
  mlir::SmallVector<int64_t> outSteps(out_num, /*Value=*/1);
  mlir::SmallVector<int64_t> outUpperBounds(out_shape.begin(), out_shape.end());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), outLowerBounds, outUpperBounds, outSteps,
    [&](mlir::OpBuilder &outNestedBuilder, mlir::Location outLoc, mlir::ValueRange outIvs) {
      auto reduceNum = outNestedBuilder.create<mlir::arith::ConstantOp>(outNestedBuilder.getUnknownLoc(), outNestedBuilder.getFloatAttr(emType, elemNum));
      auto epsNum = outNestedBuilder.create<mlir::arith::ConstantOp>(outNestedBuilder.getUnknownLoc(), outNestedBuilder.getFloatAttr(emType, eps));
      // auto one = outNestedBuilder.create<mlir::arith::ConstantOp>(outNestedBuilder.getUnknownLoc(), outNestedBuilder.getFloatAttr(emType, 1));
      auto iter = outNestedBuilder.create<mlir::arith::ConstantOp>(outNestedBuilder.getUnknownLoc(), outNestedBuilder.getFloatAttr(emType, 0));

      mlir::SmallVector<int64_t> inLowerBounds(in_num, /*Value=*/0);
      mlir::SmallVector<int64_t> inSteps(in_num, /*Value=*/1);
      mlir::SmallVector<int64_t> inUpperBounds(in_shape.begin(), in_shape.end());

      auto loop1 = buildAffineLoopNest_(outNestedBuilder, outLoc, inLowerBounds, inUpperBounds, inSteps, mlir::ValueRange({iter}), 
      [&](mlir::OpBuilder& inNestedBuilder, mlir::Location inLoc, mlir::ValueRange inIvs, mlir::ValueRange iterArgs) {
        auto ivs = getIndexArgs(inIvs, outIvs);
        auto ld = inNestedBuilder.create<mlir::affine::AffineLoadOp>(inNestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(ivs));
        return inNestedBuilder.create<mlir::arith::AddFOp>(inNestedBuilder.getUnknownLoc(), ld, iterArgs[0]);
      });
      auto div1 = outNestedBuilder.create<mlir::arith::DivFOp>(outNestedBuilder.getUnknownLoc(), loop1.getResult(0), reduceNum);
      // outNestedBuilder.create<mlir::affine::AffineStoreOp>(outNestedBuilder.getUnknownLoc(), div1, mean, mlir::ValueRange(outIvs));

      auto loop2 = buildAffineLoopNest_(outNestedBuilder, outLoc, inLowerBounds, inUpperBounds, inSteps, mlir::ValueRange({iter}), 
      [&](mlir::OpBuilder& inNestedBuilder, mlir::Location inLoc, mlir::ValueRange inIvs, mlir::ValueRange iterArgs) {
        auto ivs = getIndexArgs(inIvs, outIvs);
        auto ld = inNestedBuilder.create<mlir::affine::AffineLoadOp>(inNestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(ivs));
        auto subOp = inNestedBuilder.create<mlir::arith::SubFOp>(inNestedBuilder.getUnknownLoc(), ld, div1);
        inNestedBuilder.create<mlir::affine::AffineStoreOp>(inNestedBuilder.getUnknownLoc(), subOp, output, mlir::ValueRange(ivs));
        auto mulOp = inNestedBuilder.create<mlir::arith::MulFOp>(inNestedBuilder.getUnknownLoc(), subOp, subOp);
        return inNestedBuilder.create<mlir::arith::AddFOp>(inNestedBuilder.getUnknownLoc(), mulOp, iterArgs[0]);
      });
      auto div2 = outNestedBuilder.create<mlir::arith::DivFOp>(outNestedBuilder.getUnknownLoc(), loop2.getResult(0), reduceNum);
      auto addOp = outNestedBuilder.create<mlir::arith::AddFOp>(outNestedBuilder.getUnknownLoc(), div2, epsNum);
      auto sqrtOp = outNestedBuilder.create<mlir::math::SqrtOp>(outNestedBuilder.getUnknownLoc(), addOp);
      // auto div3 = outNestedBuilder.create<mlir::arith::DivFOp>(outNestedBuilder.getUnknownLoc(), one, sqrtOp);
      // outNestedBuilder.create<mlir::affine::AffineStoreOp>(outNestedBuilder.getUnknownLoc(), div3, invStdDev, mlir::ValueRange(outIvs));

      mlir::affine::buildAffineLoopNest(outNestedBuilder, outLoc, inLowerBounds, inUpperBounds, inSteps,
      [&](mlir::OpBuilder &inNestedBuilder, mlir::Location inLoc, mlir::ValueRange inIvs) {
        auto ivs = getIndexArgs(inIvs, outIvs);
        mlir::SmallVector<mlir::Value> scaleIvs, biasIvs, tempIvs(ivs.begin(), ivs.end());
        mlir::arith::ConstantOp zero;
        if (hasOneDim)
          zero = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
        for (int i=0; i<scaleShape.size(); i++) {
          if (scaleShape[scaleShape.size()-1-i] == input_shape[input_shape.size()-1-i])
            scaleIvs.insert(scaleIvs.begin(), tempIvs[tempIvs.size()-1-i]);
          else scaleIvs.insert(scaleIvs.begin(), zero);
        }
        for (int i=0; i<biasShape.size(); i++) {
          if (biasShape[biasShape.size()-1-i] == input_shape[input_shape.size()-1-i])
            biasIvs.insert(biasIvs.begin(), tempIvs[tempIvs.size()-1-i]);
          else biasIvs.insert(biasIvs.begin(), zero);
        }
        auto ldElem = inNestedBuilder.create<mlir::affine::AffineLoadOp>(inNestedBuilder.getUnknownLoc(), output, mlir::ValueRange(ivs));
        auto ldScale = inNestedBuilder.create<mlir::affine::AffineLoadOp>(inNestedBuilder.getUnknownLoc(), operands[1], mlir::ValueRange(scaleIvs));
        auto ldBias = inNestedBuilder.create<mlir::affine::AffineLoadOp>(inNestedBuilder.getUnknownLoc(), operands[2], mlir::ValueRange(biasIvs));
        // auto mul = inNestedBuilder.create<mlir::arith::MulFOp>(inNestedBuilder.getUnknownLoc(), ldElem, div3);
        auto div = inNestedBuilder.create<mlir::arith::DivFOp>(inNestedBuilder.getUnknownLoc(), ldElem, sqrtOp);
        auto mul = inNestedBuilder.create<mlir::arith::MulFOp>(inNestedBuilder.getUnknownLoc(), ldScale, div);
        auto add = inNestedBuilder.create<mlir::arith::AddFOp>(inNestedBuilder.getUnknownLoc(), mul, ldBias);
        inNestedBuilder.create<mlir::affine::AffineStoreOp>(inNestedBuilder.getUnknownLoc(), add, output, mlir::ValueRange(ivs));
      });
  });
  // builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), llvm::makeArrayRef({output, mean, invStdDev}));
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);
  
  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input, scale, bias}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

mlir::Value Gather::build(ComputeDAG* graph, mlir::Value input, mlir::Value indices, const std::int64_t& axis, const std::string& dtype_) {
  auto builder = graph->builder;
  auto type_input = input.getType();
  auto type_indices = indices.getType();
  llvm::ArrayRef<int64_t> input_shape, indices_shape;
  mlir::Type elementType;

  if(type_input.isa<mlir::MemRefType>()) {
    auto shape = type_input.dyn_cast<mlir::MemRefType>();
    input_shape = shape.getShape();
    elementType = shape.getElementType();
  }
  else {
    llvm::errs() << "Type of operand of Gather is not Memref.\n";
    return nullptr;
  }

  if(type_indices.isa<mlir::MemRefType>()) {
    auto shape_ = type_indices.dyn_cast<mlir::MemRefType>();
    indices_shape = shape_.getShape();
  }
  else {
    llvm::errs() << "Type of operand of Gather is not Memref.\n";
    return nullptr;
  }

  if (axis >= input_shape.size()){
    llvm::errs() << "Can't apply Gather Operation due to axis is greater than shape of input.\n";
    return nullptr;
  }
  
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);
  auto emType = getDType(builder, dtype);

  std::vector<int64_t> new_shape = input_shape;
  new_shape.erase(new_shape.begin() + axis);
  if (!(indices_shape.size() == 1 && indices_shape[0] == 1)) {
    for (int i=0; i<indices_shape.size(); i++) {
        auto loc = new_shape.begin() + axis + i;
        new_shape.insert(loc, indices_shape[i]);
    }
  }

  auto funcName = std::string({"Gather"});
  for (auto shape : input_shape) {
    funcName += "_" + std::to_string(shape);
  }
  funcName += "_indices";
  for (auto shape : indices_shape) {
    funcName += "_" + std::to_string(shape);
  }
  funcName += "_axis_" + std::to_string(axis);

  auto ip = builder.saveInsertionPoint();
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(new_shape), emType, {}, static_cast<int>(MemorySpace::global));
  auto funcOp = buildFuction(graph->module, builder, funcName, {input.getType(), indices.getType()}, {typeC});

  auto& bodyBlock = funcOp.front();
  if (bodyBlock.getOperations().size() > 0) {
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input, indices}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
    return callOp.getResult(0);
  } 
  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();

  auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
  auto output = allocOp.getResult();

  mlir::SmallVector<int64_t> lowerBounds(new_shape.size(), /*Value=*/0);
  mlir::SmallVector<int64_t> steps(new_shape.size(), /*Value=*/1);
  mlir::SmallVector<int64_t> upperBounds(new_shape.begin(), new_shape.end());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      mlir::SmallVector<mlir::Value> input_index = ivs;
      mlir::SmallVector<mlir::Value> indeces_index;
      if (indices_shape.size() == 1 && indices_shape[0] == 1) {
        auto cstOp = nestedBuilder.create<mlir::arith::ConstantIndexOp>(nestedBuilder.getUnknownLoc(), 0);
        indeces_index.push_back(cstOp.getResult());
      } else {
        for (int i=0; i<indices_shape.size(); i++) {
          indeces_index.push_back(ivs[axis + i]);
        }
      }
      auto index = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[1], mlir::ValueRange(indeces_index));
      if (!(indices_shape.size() == 1 && indices_shape[0] == 1)) {
        input_index.erase(input_index.begin() + axis, input_index.begin() + axis + indices_shape.size());
      }
      input_index.insert(input_index.begin() + axis, index);
      auto ld_elem = nestedBuilder.create<mlir::memref::LoadOp>(nestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(input_index));
      nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), ld_elem, output, mlir::ValueRange(ivs));
    }
  );
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);

  builder.restoreInsertionPoint(ip);
  auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input, indices}));
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  return callOp.getResult(0);
}

// mlir::Value Gather::build(ComputeDAG* graph, mlir::Value input, int indices, const std::int64_t& axis, const std::string& dtype_) {
//   // gather 的axis=0表示最高维度
//   auto builder = graph->builder;
//   auto type_input = input.getType();
//   llvm::ArrayRef<int64_t> input_shape;
//   mlir::Type elementType;

//   if(type_input.isa<mlir::MemRefType>()) {
//     auto shape = type_input.dyn_cast<mlir::MemRefType>();
//     input_shape = shape.getShape();
//     elementType = shape.getElementType();
//   }
//   else {
//     llvm::errs() << "Type of operand of Gather is not Memref.\n";
//     return nullptr;
//   }

//   if (axis >= input_shape.size()){
//     llvm::errs() << "Can't apply Gather Operation due to axis is greater than shape of input.\n";
//     return nullptr;
//   }
  
//   auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

//   std::vector<int64_t> new_shape = input_shape;
//   new_shape.erase(new_shape.begin() + axis);

//   auto funcName = std::string({"Gather"});

//   for (auto shape : input_shape) {
//     funcName += "_" + std::to_string(shape);
//   }
//   funcName += "_indices_" + std::to_string(indices) + "_axis_" + std::to_string(axis);

//   auto ip = builder.saveInsertionPoint();
//   auto emType = getDType(builder, dtype);
//   auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(new_shape), emType, {}, static_cast<int>(MemorySpace::global));
//   auto funcOp = buildFuction(graph->module, builder, funcName, {input.getType()}, {typeC});

//   auto& bodyBlock = funcOp.front();
//   builder.setInsertionPointToStart(&bodyBlock);
//   mlir::ValueRange operands = bodyBlock.getArguments();

//   auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), typeC);
//   auto output = allocOp.getResult();

//   mlir::SmallVector<int64_t> lowerBounds(new_shape.size(), /*Value=*/0);
//   mlir::SmallVector<int64_t> steps(new_shape.size(), /*Value=*/1);
//   mlir::SmallVector<int64_t> upperBounds(new_shape.begin(), new_shape.end());
//   mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
//     [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
//       mlir::SmallVector<mlir::Value> input_index = ivs;
//       auto index = nestedBuilder.create<mlir::arith::ConstantIndexOp>(nestedBuilder.getUnknownLoc(), indices);
//       input_index.insert(input_index.begin() + axis, index);
//       auto ld_elem = nestedBuilder.create<mlir::affine::AffineLoadOp>(nestedBuilder.getUnknownLoc(), operands[0], mlir::ValueRange(input_index));
//       nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), ld_elem, output, mlir::ValueRange(ivs));
//     }
//   );
//   builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), output);

//   builder.restoreInsertionPoint(ip);
//   auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({input}));
//   funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
//   return callOp.getResult(0);
// }

}
