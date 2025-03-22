#pragma once
#ifndef MYSTL_TYPE_TRAITS_H
#define MYSTL_TYPE_TRAITS_H

// ���ڴ����������Ժ����͹�ϵ��ģ��
#include <type_traits>

// ������һ����Ϊmystl�������ռ䣬���ڷ�װ�Զ�����������Թ��ߣ�����������ͻ
namespace mystl
{
	// T��ʾ�ṹ��ģ��Ĳ������ͣ�v��ʾ�����͵�ֵ
	template<class T,T v>
	struct m_integral_constant
	{
		static constexpr T value = v;
	};

	// using��ʾ�������ͱ���
	template <bool b>
	using m_bool_constant = m_integral_constant<bool, b>;

	// �������
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
