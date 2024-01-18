/**
* @file    model_define.hpp
* @brief   esp-dl model define header file.
* @date    2024-01-08
* @note    File auto generated, DO NOT edit.
*/

#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_base.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_softmax.hpp"
#include "dl_layer_conv2d.hpp"
#include "hand_rec_model_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace hand_rec_model_coefficient;

class HAND_REC_MODEL : public Model<int8_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
	Tensor<int8_t> m_input;
	Tensor<int8_t> m_output;

	// Declare layers as member variables
	Reshape<int8_t> l1;
	Conv2D<int8_t> l2;
	MaxPool2D<int8_t> l3;
	Conv2D<int8_t> l4;
	MaxPool2D<int8_t> l5;
	Conv2D<int8_t> l6;
	MaxPool2D<int8_t> l7;
	Reshape<int8_t> l8;
	Conv2D<int8_t> l9;
	Conv2D<int8_t> l10;
public:
	Softmax<int8_t> l11;

	/**
	* @brief Initialize layers in constructor function
	*/
	HAND_REC_MODEL() : l1(Reshape<int8_t>({96, 96, 1}, "l1")),
			l2(Conv2D<int8_t>(0, get_statefulpartitionedcall_sequential_conv2d_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_biasadd_activation(), PADDING_VALID, {}, 1, 1, "l2")),
			l3(MaxPool2D<int8_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l3")),
			l4(Conv2D<int8_t>(0, get_statefulpartitionedcall_sequential_conv2d_1_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_1_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_1_biasadd_activation(), PADDING_VALID, {}, 1, 1, "l4")),
			l5(MaxPool2D<int8_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l5")),
			l6(Conv2D<int8_t>(0, get_statefulpartitionedcall_sequential_conv2d_2_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_2_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_2_biasadd_activation(), PADDING_VALID, {}, 1, 1, "l6")),
			l7(MaxPool2D<int8_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l7")),
			l8(Reshape<int8_t>({1, 1, 6400}, "l8")),
			l9(Conv2D<int8_t>(0, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l9")),
			l10(Conv2D<int8_t>(-1, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_VALID, {}, 1, 1, "l10")),
			l11(Softmax<int8_t>(-6, "l11"))
	{
	}

	/**
	* @brief call each layers build(...) function in sequence
	* @param input
	*/
	void build(Tensor<int8_t> &input)
	{
		this->l1.build(input);
		this->l2.build(this->l1.get_output());
		this->l3.build(this->l2.get_output());
		this->l4.build(this->l3.get_output());
		this->l5.build(this->l4.get_output());
		this->l6.build(this->l5.get_output());
		this->l7.build(this->l6.get_output());
		this->l8.build(this->l7.get_output());
		this->l9.build(this->l8.get_output());
		this->l10.build(this->l9.get_output());
		this->l11.build(this->l10.get_output());
	}

	/**
	* @brief call each layers call(...) function in sequence
	* @param input
	*/
	void call(Tensor<int8_t> &input)
	{
		this->l1.call(input);
		input.free_element();
		this->l2.call(this->l1.get_output());
		this->l1.get_output().free_element();
		this->l3.call(this->l2.get_output());
		this->l2.get_output().free_element();
		this->l4.call(this->l3.get_output());
		this->l3.get_output().free_element();
		this->l5.call(this->l4.get_output());
		this->l4.get_output().free_element();
		this->l6.call(this->l5.get_output());
		this->l5.get_output().free_element();
		this->l7.call(this->l6.get_output());
		this->l6.get_output().free_element();
		this->l8.call(this->l7.get_output());
		this->l7.get_output().free_element();
		this->l9.call(this->l8.get_output());
		this->l8.get_output().free_element();
		this->l10.call(this->l9.get_output());
		this->l9.get_output().free_element();
		this->l11.call(this->l10.get_output());
		this->l10.get_output().free_element();
	}

	/**
	* @brief set model inference input
	* @param in_ptr
	*/
	void input(int8_t *in_ptr)
	{
		this->m_input.set_element(in_ptr).set_exponent(1).set_shape({96, 96, 1}).set_auto_free(false);
	}

	/**
	* @brief run model inference
	*/
	auto *invoke(void)
	{
		this->forward(this->m_input);
		return this->l11.get_output().get_element_ptr();
	}
};
