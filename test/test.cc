#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
using namespace KernelCodeGen;


void test_operators() {
  KernelCodeGenerator generator("CUDA");
  auto graph = generator.createGraph("demo");
  generator.setLogMode(Log::Debug);
  // generator.opts.push_back(std::move(std::make_unique<BatchMatmulOptimizer>()));
  // generator.opts.push_back(std::move(std::make_unique<MatmulOptimizer>()));
  // generator.opts.push_back(std::move(std::make_unique<BinaryOptimizer>()));
  generator.opts.push_back(std::move(std::make_unique<ElementWiseOptimizer>()));
  // generator.opts.push_back(std::move(std::make_unique<LayerNormOptimizer>()));
  // generator.opts.push_back(std::move(std::make_unique<GatherOptimizer>()));
  

  auto A = graph.create<PlaceHolder>(std::vector<int64_t>{2, 768, 768}, std::string{"float32"});
  // int64_t axis = 0;
  // auto indices = graph.create<PlaceHolder>(std::vector<int64_t>{1}, std::string{"index"});
  // auto gather = graph.create<Gather>(A, indices, axis);
  auto gelu = graph.create<ElementWise>(A, "Cast", MemorySpace::global, "int32");  // 必须使用MemorySpace::global，且int16只能转float16，etc.

  // int64_t axis_ = 1;
  // float eps=1e-5;
  // auto scale = graph.create<PlaceHolder>(std::vector<int64_t>{768, 768}, std::string{"float32"});
  // auto bias = graph.create<PlaceHolder>(std::vector<int64_t>{768, 768}, std::string{"float32"});
  // auto layernorm = graph.create<LayerNorm>(A, scale, bias, axis_, eps);

  // auto A = graph.create<PlaceHolder>(std::vector<int64_t>{16, 1, 1, 256}, std::string{"float32"});
  // auto B = graph.create<PlaceHolder>(std::vector<int64_t>{1, 128, 256}, std::string{"float32"});
  // graph.create<Binary>(A, B, "Add");

  // int m = 2048, n = 2048, k = 1024;
  // auto A = graph.create<PlaceHolder>(std::vector<int64_t>{m, k}, std::string{"float32"});
  // auto B = graph.create<PlaceHolder>(std::vector<int64_t>{k, n}, std::string{"float32"});
  // auto C = graph.create<Matmul>(A, B);

  // auto A = graph.create<PlaceHolder>(std::vector<int64_t>{256, 2048, 64}, std::string{"float32"});
  // auto B = graph.create<PlaceHolder>(std::vector<int64_t>{256, 2048, 64}, std::string{"float32"});
  // auto C = graph.create<BatchedMatmul>(A, Layout::rowMajor, B, Layout::colMajor);

  graph.dump();
  auto module = generator.optimize(graph);
  generator.dump(module);
  auto&& sourceCode = generator.codegen(module);
}

void test_matmul() {

  /* 1. Demo */
  KernelCodeGenerator generator("CUDA");
  
  auto graph = generator.createGraph("matmul_demo");
  generator.setLogMode(Log::Debug);

  int m = 4096, n = 2048, k = 1024;
  auto A = graph.create<PlaceHolder>(std::vector<int64_t>{m, k}, std::string{"float32"});
  auto B = graph.create<PlaceHolder>(std::vector<int64_t>{k, n}, std::string{"float32"});
  auto C = graph.create<Matmul>(A, B);
  auto D = graph.create<Relu>(C, MemorySpace::inplace);

  graph.dump();
  auto module = generator.optimize(graph);
  generator.dump(module);
  auto&& sourceCode = generator.codegen(module);
  generator.save(sourceCode, "../test/matmul/matmulKernel.cu");
  std::string adaptorCode = "";
  adaptorCode += "#include \"matmulKernel.cu\"\n";
  adaptorCode += "const int M = " + std::to_string(m) + ";\n";
  adaptorCode += "const int N = " + std::to_string(n) + ";\n";
  adaptorCode += "const int K = " + std::to_string(k) + ";\n";
  adaptorCode += "#define kernelFunc matmul_demo::kernel0\n";
  generator.save(adaptorCode, "../test/matmul/adaptor.cu");
  system("cd ../build && make matmul && ../bin/matmul");

  /* 2. Batch perf test.*/
  std::vector<int64_t> dims {256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384};
  for (auto dim : dims) {
    auto fileName = "Matmul_M" + std::to_string(dim) + "_N" + std::to_string(dim) + "_K" + std::to_string(dim);
    auto graph = generator.createGraph(fileName);
    generator.setLogMode(Log::Release);

    int m = dim, n = dim, k = dim;
    auto A = graph.create<PlaceHolder>(std::vector<int64_t>{m, k}, std::string{"float32"});
    auto B = graph.create<PlaceHolder>(std::vector<int64_t>{k, n}, std::string{"float32"});
    // auto C = graph.create<Matmul>(A, B, MemorySpace::global);
    auto C = graph.create<Matmul>(A, B, std::string{"float32"});

    auto module = generator.optimize(graph);
    auto&& sourceCode = generator.codegen(module);
    auto srcFile = "../test/matmul/" + fileName + ".cu";
    generator.save(sourceCode, srcFile);

    std::string adaptorCode = "";
    adaptorCode += "#include \"" + fileName + ".cu\"\n";
    adaptorCode += "const int M = " + std::to_string(m) + ";\n";
    adaptorCode += "const int N = " + std::to_string(n) + ";\n";
    adaptorCode += "const int K = " + std::to_string(k) + ";\n";
    adaptorCode += "#define kernelFunc " + fileName + "::kernel0\n";
    generator.save(adaptorCode, "../test/matmul/adaptor.cu");
    system("cd ../build && make matmul && ../bin/matmul");
  }
}

void test_flash_attention() {
  /* 1. Demo */
  KernelCodeGenerator generator("CUDA");
  
  auto graph = generator.createGraph("flash_attn_demo");
  generator.setLogMode(Log::Debug);
  // generator.opts.push_back(std::move(std::make_unique<FMHAOptimizer>()));

  int64_t hidden_dim = 2048L;
  int64_t total_token = 16 * 1024L;
  int64_t head_dim = 64L;
  int64_t seq_len = 2048L;

  int64_t batch_size = total_token / seq_len;
  int64_t head_num = hidden_dim / head_dim;
  

  auto Q = graph.create<PlaceHolder>(std::vector<int64_t>{batch_size, head_num, seq_len, head_dim}, std::string{"float32"});
  auto K = graph.create<PlaceHolder>(std::vector<int64_t>{batch_size, head_num, seq_len, head_dim}, std::string{"float32"});
  auto V = graph.create<PlaceHolder>(std::vector<int64_t>{batch_size, head_num, seq_len, head_dim}, std::string{"float32"});

  auto S = graph.create<BatchedMatmul>(Q, Layout::rowMajor, K, Layout::colMajor);
  auto P = graph.create<Softmax>(S, -1, MemorySpace::inplace);
  auto O = graph.create<BatchedMatmul>(P, Layout::rowMajor, V, Layout::rowMajor);
  

  graph.dump();
  auto module = generator.optimize(graph);
  generator.dump(module);
  auto&& sourceCode = generator.codegen(module);
  llvm::outs() << "==============";
  llvm::outs() << sourceCode ;
  generator.save(sourceCode, "/home/pangyunfei/xushilong/KernelCodeGen/bin/demo.cu");
  // std::string adaptorCode = "";
  // adaptorCode += "#include \"matmulKernel.cu\"\n";
  // adaptorCode += "const int M = " + std::to_string(m) + ";\n";
  // adaptorCode += "const int N = " + std::to_string(n) + ";\n";
  // adaptorCode += "const int K = " + std::to_string(k) + ";\n";
  // adaptorCode += "#define kernelFunc matmul_demo::kernel0\n";
  // generator.save(adaptorCode, "../test/matmul/adaptor.cu");
  // system("cd ../build && make matmul && ../bin/matmul");

}


int main(int argc, char* argv[]) {

  // test_matmul();
  test_operators();
  // test_flash_attention();

}