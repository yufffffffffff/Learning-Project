#include <iostream>

// 定义一个函数用于计算两个整数的和
int add(int a, int b) {
	return a + b;
}

int main() {
	int num1 = 5;
	int num2 = 3;
	// 调用 add 函数计算两数之和
	int result = add(num1, num2);
	std::cout << "两数之和为: " << result << std::endl;
	return 0;
}