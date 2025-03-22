#pragma once
#ifndef MYSTL_TYPE_TRAITS_H
#define MYSTL_TYPE_TRAITS_H

// 用于处理类型属性和类型关系的模板
#include <type_traits>

// 定义了一个名为mystl的命名空间，用于封装自定义的类型特性工具，避免命名冲突
namespace mystl
{
	// T表示结构体模板的参数类型，v表示该类型的值
	template<class T,T v>
	struct m_integral_constant
	{
		static constexpr T value = v;
	};

	// using表示创建类型别名
	template <bool b>
	using m_bool_constant = m_integral_constant<bool, b>;

	// 定义别名
	typedef m_bool_constant<true> m_true_type;
	typedef m_bool_constant<false> m_false_type;

	template<class T1,class T2>
	struct pair;
	
	template<class T>
	struct is_pair :mystl::m_false_type {};

	template<class T1,class T2>
	struct is_pair<mystl::pair<T1, T2>> :mystl::m_true_type {};
}

#endif // !MYSTL_TYPE_TRAITS_H
