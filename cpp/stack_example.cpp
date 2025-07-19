#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;

    // Push 7 elemets 
    for (int i=1; i<=7; i++) {
        s.push(i);
        std::cout << "Pushed: " << i << std::endl;
    }

    std::cout << "\nStack size after pushing: " << s.size() << "\n";

    return 0;
}