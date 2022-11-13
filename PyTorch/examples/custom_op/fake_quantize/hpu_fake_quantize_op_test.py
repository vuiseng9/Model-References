import torch
import habana_frameworks.torch.core

# ensure that environment variable GC_KERNEL_PATH is set to built custom lib
# e.g. export GC_KERNEL_PATH=/home/ubuntu/Habana_Custom_Kernel/build/src/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so

# Register Op:
# 1. Activate environment
# 2. python setup.py install
# 3. find the built .so e.g. /home/ubuntu/Model-References/PyTorch/examples/custom_op/fake_quantize/build/lib.linux-x86_64-3.8/hpu_fake_quantize.cpython-38-x86_64-linux-gnu.s

# Example of test run:
# pytest --custom_op_lib /home/ubuntu/Model-References/PyTorch/examples/custom_op/fake_quantize/build/lib.linux-x86_64-3.8/hpu_fake_quantize.cpython-38-x86_64-linux-gnu.so hpu_fake_quantize_op_test.py

test_blob_path = "/home/ubuntu/nncf/examples/torch/classification/qio_rn18-8bit-wq-sym-aq-sym-per-tensor.pth"
test_tensor_key = 'ResNet/NNCFConv2d[conv1]/conv2d_0|WEIGHT'
EPS = 1e-6

def test_fakequantize_op_function(custom_op_lib_path):
    torch.ops.load_library(custom_op_lib_path)

    d = torch.load(test_blob_path)
    test_input = d[test_tensor_key]['Forward']['i']
    
    input_ = torch.from_numpy(test_input['input_']).to('hpu')
    input_low = torch.from_numpy(test_input['input_low']).to('hpu')
    input_range = torch.from_numpy(test_input['input_range']).to('hpu')
    levels = test_input['levels']

    fq_out = torch.ops.custom_op.fakequantize(input_, input_low, input_range, levels)

    test_output = torch.from_numpy(d[test_tensor_key]['Forward']['o'])

    assert( (test_output.detach().cpu() - fq_out.detach().cpu()).sum().abs() < EPS)

