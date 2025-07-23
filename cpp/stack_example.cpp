#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;

    // Push 7 elemets to the stack
    for (int i = 1; i <= 7; i++) {
        s.push(i);
        std::cout << "Pushed: " << i << std::endl;
            }

    std::cout << "Stack after pushing element: " << s.size() << std::endl;
    
    // Pop and display all elements
    std::cout << "\nPopping elements:\n";
    while (!s.empty()) {
        std::cout << "Popped: " << s.top() << std::endl;
        s.pop();
    }

    return 0;

}