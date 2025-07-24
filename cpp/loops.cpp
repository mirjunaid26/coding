#include <iostream>

int main() {
    const int maxIterations = 5;

    int iteration = 0;

    // For loop
    std::cout << "For loop:" << std::endl;
    for (iteration = 0; iteration < maxIterations; ++iteration) {
        std::cout << "iteration = " << iteration << std::endl;
    }
    std::cout << "-----------------" << std::endl;
    
    // While loop
    std::cout << "While loop:" << std::endl;
    iteration = 0;
    while (iteration < maxIterations) {
        std::cout << "iteration = " << iteration << std::endl;
        ++iteration;
    }
    std::cout << "-----------------" << std::endl;
    
    // Do-while loop
    std::cout << "Do-while loop:" ;
    iteration = 0;
    {
    do {
        std::cout << "iteration = " << iteration << std::endl;
        ++iteration;
    } while (iteration < maxIterations);
    }
    return 0;
}

