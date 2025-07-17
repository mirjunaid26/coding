#include<iostream>

int main(){
    int a = 5;
    int b = 1;
    
    std::cout << a << b << std::endl;

    int temp = a;
    
    a = b;
    b = temp;

    std::cout << a << b << std::endl;

    return 0;
}