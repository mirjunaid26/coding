#include <iostream>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(10,20); // calling the function add() with arguments 10 and 20
    std::cout << "Result: " << result << std::endl;
    return 0;
}