#include<iostream>

int main() {
    int a = 1;
    int b = 2;

    std::cout << a << b << std::endl;

    // let us swap values
    int temp = a;
    a = b;
    b = temp;
    
    std::cout << a << b;
    
    return 0;

}