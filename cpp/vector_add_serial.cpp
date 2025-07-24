#include <iostream>
#include <vector>

int main() {
    const int N = 8;
    std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> B = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> C(N);

    // Element-wise addition
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < N; ++i){
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}