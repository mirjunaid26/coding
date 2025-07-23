#include <iostream>
#include <stack>

int main() {
    std::stack<int> stack;

    for (int i = 1; i <= 7; i++) {
        stack.push(i);
        std::cout << "Pushed: " << i << std::endl;
            }

    std::cout << "Stack after pushing element: " << stack.size() << std::endl;
    
    return 0;

}