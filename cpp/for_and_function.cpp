#include <iostream>

// Declare a function
int square(int x);
int main() {
    for (int i; i<=5; ++i) {
        std::cout << "Square of " << i << " is " << square(i) << std::endl;
    }
    return 0;
}

// Define the function
int square(int x) {
    return x * x;
}