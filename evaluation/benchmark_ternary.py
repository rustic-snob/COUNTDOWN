import time
import torch
import torch.nn as nn
import bitblas

# # # Test parameters
input_features = 4096
output_features = 14336
lowrank_features = 512
batch_size = 1

# enabling debug output

# bitblas.set_log_level("Debug")
matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=output_features,  # N dimension
    K=input_features,  # K dimension
    A_dtype="bfloat16",  # activation A dtype
    W_dtype="int2",  # weight W dtype
    accum_dtype="float32",  # accumulation dtype
    out_dtype="float32",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)

lowrank_layer = (torch.randn(input_features, lowrank_features, device='cuda', dtype=torch.bfloat16), torch.randn(lowrank_features, output_features, device='cuda', dtype=torch.bfloat16))

def lowrank(x, predictor_u, predictor_v):
    return (x @ predictor_u @ predictor_v)

matmul = bitblas.Matmul(config=matmul_config)

# Create input matrices
input_tensor = torch.rand((1, input_features), dtype=torch.bfloat16).cuda()
weight_tensor = torch.randint(-1, 2, (output_features, input_features), dtype=torch.int8).cuda()

# Transform weight tensor to int2 data type
weight_tensor_int2 = matmul.transform_weight(weight_tensor)

start_time = time.time()
# Perform mixed-precision matrix multiplication
output_tensor = matmul(input_tensor, weight_tensor_int2)
torch.cuda.synchronize()
bitblas_time = time.time() - start_time

weight_tensor = weight_tensor.t().to(torch.bfloat16)

start_time = time.time()
# Reference result using PyTorch matmul for comparison
ref_result = torch.matmul(input_tensor, weight_tensor)
torch.cuda.synchronize()
ref_time = time.time() - start_time

start_time = time.time()
# Perform low-rank approximation
lowrank_result = lowrank(input_tensor, *lowrank_layer)
torch.cuda.synchronize()
lowrank_time = time.time() - start_time

print(f"BitBLAS Time: {bitblas_time * 1000:.3f} ms")
print(f"Ref Time: {ref_time * 1000:.3f} ms")
print(f"Lowrank Time: {lowrank_time * 1000:.3f} ms")

# benchmark latency for bitblas
latency = matmul.profile_latency()
print(f"BitBLAS Time: {latency:.3f} ms")

def profile(model, *args):

    import numpy as np

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(*args)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000 / num_repeats

    with torch.no_grad():
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        warmup_runtime = get_runtime()
        num_repeats = max(1, int(1000 / warmup_runtime))
        times = get_runtime(num_repeats)
    return np.mean(times)

torch_time = profile(torch.matmul, input_tensor, weight_tensor)

print(f"Torch Time: {torch_time:.3f} ms")

lowrank_time = profile(lowrank, input_tensor, *lowrank_layer)
print(f"Lowrank Time: {lowrank_time:.3f} ms")