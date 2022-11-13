/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

namespace ns_FakeQuantizeKernel
{
    struct Params
    {
        int64_t levels;
    };
}

bool register_fakequantize() {
    // Registering custom_op::fakequantize
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::USER_PARAMS, 3}; //shouldnt this be 
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    //What does this means
    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(ns_FakeQuantizeKernel::Params);
      params->levels = inputs[3].toInt(); //levels
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::fakequantize", //schema name
        "custom_quantize_fwd_f32", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::fakequantize\n";
    return true;
}


at::Tensor fakequantize_execute(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int64_t levels) {
  // TODO
  // Registering the custom op, need to be called only once
  static bool registered = register_fakequantize();
  TORCH_CHECK(registered, "fakequantize kernel not registered" );

  std::vector<c10::IValue> inputs{input, input_low, input_range, levels};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::fakequantize");

  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("fakequantize(Tensor input, Tensor input_low, Tensor input_range, int levels) -> Tensor");
//   m.def("custom_relu_backward(Tensor grad, Tensor self, Scalar threshold) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("fakequantize", fakequantize_execute);
//   m.impl("custom_relu_backward", custom_relu_backward_execute);
}

