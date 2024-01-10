#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import time
import numpy as np
import onnx
import onnxruntime as rt
import pickle
import io
import os
import sys
import platform
import itertools
import re

PADDING_TYPE_DICT = {"VALID": "PADDING_VALID",
                     "NOTSET": "PADDING_NOT_SET",
                     "SAME_UPPER": "PADDING_SAME_BEGIN",
                     "SAME_LOWER": "PADDING_SAME_END"}

ACTIVATION_OPS = ['Relu', 'Prelu', 'LeakyRelu']

def find_exponent(layer_name, quant_info_str):
    pattern = re.escape(layer_name) + r', .*exponent: (-?\d+)'
    match = re.search(pattern, quant_info_str)
    if match:
        return int(match.group(1))
    else:
        return None


def find_next_nodes(graph, node):
    next_nodes = []
    for output in node.output:
        for _node in graph.node:
            if output in _node.input:
                next_nodes.append(_node)
    return next_nodes


def parse_padding_and_strides(node_attrs):
    padding_type = "PADDING_VALID"
    padding = ""
    strides = "1, 1"
    kernel_shape = "2, 2"
    for attr in node_attrs:
        if attr.name == 'auto_pad':     # padding_type
            padding_type = PADDING_TYPE_DICT[attr.s]
        elif attr.name == 'pads':       # padding
            padding = ", ".join(map(str, attr.ints))
        elif attr.name == 'strides':
            strides = ", ".join(map(str, attr.ints))
        elif attr.name == 'kernel_shape':
            kernel_shape = ", ".join(map(str, attr.ints))
    return padding_type, padding, strides, kernel_shape


def parse_conv2d(graph, layer_init, layer_idx, bw, node, value_info, quant_info):
    exponent = find_exponent(node.name, quant_info)
    node_name = node.name.replace('/', '_').lower()
    padding_type, padding, strides, _ = parse_padding_and_strides(node.attribute)

    next_nodes = find_next_nodes(graph, node)
    activation_str = "NULL"
    for next_node in next_nodes:
        if next_node.op_type in ACTIVATION_OPS:
            activation_str = 'get_{}_activation()'.format(node_name)
            break
    layer_init = layer_init.format(layer_idx, bw, exponent, node_name, node_name, activation_str, padding_type, padding, strides, layer_idx)
    return layer_init


def parse_gemm(graph, layer_init, layer_idx, bw, node, value_info, quant_info):
    exponent = find_exponent(node.name, quant_info)
    node_name = node.name.replace('/', '_').lower()
    next_nodes = find_next_nodes(graph, node)
    activation_str = "NULL"
    for next_node in next_nodes:
        if next_node.op_type  in ACTIVATION_OPS:
            activation_str = 'get_{}_activation()'.format(node_name)
            break
    padding_type = "PADDING_VALID"
    padding = ""
    strides = "1, 1"
    layer_init = layer_init.format(layer_idx, bw, exponent, node_name, node_name, activation_str, padding_type, padding, strides, layer_idx)
    return layer_init


def parse_maxpool2d(graph, layer_init, layer_idx, bw, node, value_info, quant_info):
    padding_type, padding, strides, kernel_shape = parse_padding_and_strides(node.attribute)
    layer_init = layer_init.format(layer_idx, bw, kernel_shape, padding_type, padding, strides, layer_idx)
    return layer_init

def parse_reshape(graph, layer_init, layer_idx, bw, node, value_info, quant_info):
    if value_info.name.find("flatten") != -1:
        shape = ", ".join(["1", "1", str(value_info.type.tensor_type.shape.dim[-1].dim_value)])
    else:
        dims = value_info.type.tensor_type.shape.dim
        shape = ", ".join([str(dims[3].dim_value), str(dims[2].dim_value), str(dims[1].dim_value)])
    layer_init = layer_init.format(layer_idx, bw, shape, layer_idx)
    return layer_init

def parse_softmax(graph, layer_init, layer_idx, bw, node, value_info, quant_info):
    exponent = find_exponent(node.name, quant_info)
    layer_init = layer_init.format(layer_idx, bw, exponent, layer_idx)
    return layer_init

OP_PARSE_DICT = {"Conv": {"include": "conv2d",
                        "declare": "Conv2D<{}> l{};",
                        "initialize": "l{}(Conv2D<{}>({}, get_{}_filter(), get_{}_bias(), {}, {}, {{{}}}, {}, \"l{}\"))",
                        "parser": parse_conv2d},
                "Gemm": {"include": "conv2d",
                         "declare": "Conv2D<{}> l{};",
                         "initialize": "l{}(Conv2D<{}>({}, get_{}_filter(), get_{}_bias(), {}, {}, {{{}}}, {}, \"l{}\"))",
                         "parser": parse_gemm},
                'MaxPool': {"include": "max_pool2d",
                            "declare": "MaxPool2D<{}> l{};",
                            "initialize": "l{}(MaxPool2D<{}>({{{}}}, {}, {{{}}}, {}, \"l{}\"))",
                            "parser": parse_maxpool2d},
                'Relu': {},
                'Reshape': {"include": "reshape",
                            "declare": "Reshape<{}> l{};",
                            "initialize": "l{}(Reshape<{}>({{{}}}, \"l{}\"))",
                            "parser": parse_reshape},
                'Softmax': {"include": "softmax",
                            "declare": "Softmax<{}> l{};",
                            "initialize": "l{}(Softmax<{}>({}, \"l{}\"))",
                            "parser": parse_softmax},
                'Transpose': {}
            }

def parse_proto(model_proto, bw, quant_info):
    bw = bw + '_t'
    layers_inc_list = []
    layers_decl_list = []
    layers_init_list = []
    in_info = {}
    layer_idx = 1

    graph_input = model_proto.graph.input[0]
    in_info['shape'] = ", ".join(map(str, [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim][1:]))
    in_info['exponent'] = find_exponent(graph_input.name, quant_info)

    for node, info in itertools.zip_longest(model_proto.graph.node, model_proto.graph.value_info):
        op_type = node.op_type
        if op_type in OP_PARSE_DICT.keys() and OP_PARSE_DICT[op_type]:
            layer_inc = OP_PARSE_DICT[op_type]['include']
            layer_decl = OP_PARSE_DICT[op_type]['declare'].format(bw, layer_idx)
            layer_init = OP_PARSE_DICT[op_type]['parser'](model_proto.graph, OP_PARSE_DICT[op_type]['initialize'], layer_idx, bw, node, info, quant_info)
            layers_inc_list.append(layer_inc)
            layers_decl_list.append(layer_decl)
            layers_init_list.append(layer_init)
            layer_idx += 1
    return in_info, list(set(layers_inc_list)), layers_decl_list, layers_init_list


def generate_model_define(fp, model_name, bw, in_info, layers_inc_list, layers_decl_list, layers_init_list):
    bw = bw + '_t'
    layer_num = len(layers_decl_list)

    description = (
        '/**\n'
        '* @file    model_define.hpp\n'
        '* @brief   esp-dl model define header file.\n'
        '* @date    {}\n'
        '* @note    File auto generated, DO NOT edit.\n'
        '*/\n\n'
        ).format(time.strftime("%Y-%m-%d", time.localtime()))

    class_derive = (
        '#pragma once\n'
        '#include "dl_layer_model.hpp"\n'
        '#include "dl_layer_base.hpp"\n'
        '{}\n'
        '#include "{}_coefficient.hpp"\n'
        '#include <stdint.h>\n\n'
        'using namespace dl;\n'
        'using namespace layer;\n'
        'using namespace {}_coefficient;\n\n'
        'class {} : public Model<{}> // Derive the Model class in "dl_layer_model.hpp" \n'
        '{{\n'
        ).format('\n'.join('#include "dl_layer_{}.hpp"'.format(n) for n in layers_inc_list), model_name, model_name, model_name.upper(), bw)

    member_declare = (
        'private:\n\t'
            'Tensor<{}> m_input;\n\t'
            'Tensor<{}> m_output;\n\n\t'
            '// Declare layers as member variables\n\t'
            '{}\n'
        'public:\n\t'
            '{}\n\t'
        ).format(bw, bw, '\n\t'.join(layers_decl_list[:-1]), layers_decl_list[-1])

    constructor = (
        '\n\t/**\n\t'
        '* @brief Initialize layers in constructor function\n\t'
        '*/\n\t'
        '{}() : {}\n\t'
        '{{}}\n\t'
        ).format(model_name.upper(), ',\n\t\t\t'.join(layers_init_list))

    build = (
        '\n\t/**\n\t'
        '* @brief call each layers build(...) function in sequence\n\t'
        '* @param input\n\t'
        '*/\n\t'
        'void build(Tensor<{}> &input)\n\t'
        '{{\n\t\t'
            'this->l1.build(input);\n\t\t'
            '{}\n\t'
        '}}\n\t'
        ).format(bw, "\n\t\t".join(["this->l{}.build(this->l{}.get_output());".format(i + 1, i) for i in range(1, layer_num)]))

    call = (
        '\n\t/**\n\t'
        '* @brief call each layers call(...) function in sequence\n\t'
        '* @param input\n\t'
        '*/\n\t'
        'void call(Tensor<{}> &input)\n\t'
        '{{\n\t\t'
            'this->l1.call(input);\n\t\t'
            'input.free_element();\n\t\t'
            '{}\n\t'
        '}}\n'
        ).format(bw, "\n\t\t".join(["this->l{}.call(this->l{}.get_output());\n\t\tthis->l{}.get_output().free_element();".format(i + 1, i, i) for i in range(1, layer_num)]))

    inference_input = (
        '\n\t/**\n\t'
        '* @brief set model inference input\n\t'
        '* @param in_ptr\n\t'
        '*/\n\t'
        'void input({} *in_ptr)\n\t'
        '{{\n\t\t'
            'this->m_input.set_element(in_ptr).set_exponent({}).set_shape({{{}}}).set_auto_free(false);\n\t'
        '}}\n'
        ).format(bw, in_info['exponent'], in_info['shape'])

    inference_invoke = (
        '\n\t/**\n\t'
        '* @brief run model inference\n\t'
        '*/\n\t'
        'auto *invoke(void)\n\t'
        '{{\n\t\t'
            'this->forward(this->m_input);\n\t\t'
            'return this->l{}.get_output().get_element_ptr();\n\t'
        '}}\n'
        '}};\n'
        ).format(layer_num)

    fp.write(description)
    fp.write(class_derive)
    fp.write(member_declare)
    fp.write(constructor)
    fp.write(build)
    fp.write(call)
    fp.write(inference_input)
    fp.write(inference_invoke)


def quantize(esp_dl_dir, model_file, x_pkl, y_pkl, output_dir, target='esp32', bw='int8', granularity='per-tensor', calib_method='minmax', evalute=False):
    system_type = platform.system()
    quant_tool_path = f'{esp_dl_dir}/tools/quantization_tool'
    lib_path = f'{quant_tool_path}/{system_type.lower()}'
    if system_type == 'Windows':
        lib_path = lib_path.replace('/', '\\')
        quant_tool_path = quant_tool_path.replace('/', '\\')
    sys.path.append(lib_path)
    sys.path.append(quant_tool_path)

    import optimizer
    import calibrator
    import evaluator

    # Optimize the onnx model
    optimized_model_path = optimizer.optimize_fp_model(model_file)
    model_name = os.path.splitext(os.path.basename(model_file))[0].lower()

    # Calibration
    if y_pkl is None:
        with open(pickle_data, 'rb') as f:
            (test_images, test_labels) = pickle.load(f)
    else:
        with open(x_pkl, 'rb') as f:
            (test_images) = pickle.load(f)
        with open(y_pkl, 'rb') as f:
            (test_labels) = pickle.load(f)

    # test_images = test_images / 255.0

    # Prepare the calibration dataset
    data_size = len(test_images) // 2
    range_num = data_size // 100
    calib_dataset = test_images[0:data_size:range_num]

    pickle_file_path = f'{output_dir}/calib.pickle'
    model_proto = onnx.load(optimized_model_path)

    print('Generating the quantization table:')
    calib = calibrator.Calibrator(bw, granularity, calib_method)
    calib.set_providers(['CPUExecutionProvider'])

    calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    calib.export_coefficient_to_cpp(model_proto, pickle_file_path, target, output_dir, '{}_coefficient'.format(model_name), True)

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    print('{}'.format(output))
    in_info, layers_inc_list, layers_decl_list, layers_init_list = parse_proto(model_proto, bw, output)

    print('Creating {}_model.hpp'.format(model_name.lower()))
    with open(os.path.join(output_dir, "{}_model.hpp".format(model_name)), "w") as fp:
        generate_model_define(fp, model_name, bw, in_info, layers_inc_list, layers_decl_list, layers_init_list)

    # Evaluate the performance
    if evalute is True:
        print('Evaluating the performance on {}:'.format(target))
        eva = evaluator.Evaluator(bw, granularity, target)
        eva.set_providers(['CPUExecutionProvider'])
        eva.generate_quantized_model(model_proto, pickle_file_path)

        output_names = [n.name for n in model_proto.graph.output]
        providers = ['CPUExecutionProvider']
        m = rt.InferenceSession(optimized_model_path, providers=providers)

        batch_size = 100
        batch_num = int(len(test_images) / batch_size)
        res = 0
        fp_res = 0
        input_name = m.get_inputs()[0].name
        for i in range(batch_num):
            # int8_model
            [outputs, _] = eva.evalute_quantized_model(test_images[i * batch_size:(i + 1) * batch_size], False)
            res = res + sum(np.argmax(outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])

            # floating-point model
            fp_outputs = m.run(output_names, {input_name: test_images[i * batch_size:(i + 1) * batch_size].astype(np.float32)})
            fp_res = fp_res + sum(np.argmax(fp_outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])

        print('accuracy of int8 model is: %f' % (res / len(test_images)))
        print('accuracy of fp32 model is: %f' % (fp_res / len(test_images)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use esp_dl component tools to quantize model"
    )
    parser.add_argument(
        "-e",
        "--esp-dl-dir",
        help="esp_dl component tools dir",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-file",
        help="The model file to quantize",
        required=True
    )
    parser.add_argument(
        "-x",
        "--x-pickle",
        help="Pickle data for calibration",
        required=True,
    )
    parser.add_argument(
        "-y",
        "--y-pickle",
        help="Pickle data for calibration",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output of the generated files",
        required=True
    )
    parser.add_argument(
        "-t",
        "--target",
        help="Chip target",
        required=False
    )
    parser.add_argument(
        "-b",
        "--bit-width",
        help="int8|int16",
        required=False
    )
    parser.add_argument(
        "-g",
        "--granularity",
        help="'per-tensor|per-channel",
        required=False
    )
    parser.add_argument(
        "-c",
        "--calib-method",
        help="minmax|entropy",
        required=False
    )

    args = parser.parse_args()

    quantize(args.esp_dl_dir, args.model_file, args.x_pickle, args.y_pickle, args.output_dir,
            'esp32' if args.target is None else args.target,
            'int8' if args.bit_width is None else args.bit_width,
            'per-tensor' if args.granularity is None else args.granularity,
            'minmax' if args.calib_method is None else args.calib_method)