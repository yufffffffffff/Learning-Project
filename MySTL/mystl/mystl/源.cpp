#include <iostream>

// ����һ���������ڼ������������ĺ�
int add(int a, int b) {
	return a + b;
}

int main() {
	int num1 = 5;
	int num2 = 3;
	// ���� add ������������֮��
	int result = add(num1, num2);
	std::cout << "����֮��Ϊ: " << result << std::endl;
	return 0;
}