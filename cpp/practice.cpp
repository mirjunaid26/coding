#include<iostream>

int main(){
    // Data types
    double price = 9.99;
    float interestRate = 3.67F;
    long fileSize = 90000L;
    char letter = 'a';
    string name = "Ali";
    bool isValid = true;
    auto years = 5;

    // Number systems
    int x = 255;
    int y = 0b1111111;
    int z = 0xFF;

    // Data types size and limits
    int bytes = sizeof(int);
    int min = numeric_limits<int>::min();
    int max = numeric_limits<int>::max();

    // Arrays
    int numbers[] = {1, 2, 3, 4};
    cout << numbers[0];

    // C-style casting
    double a = 2.0;
    int b = (int) a;

    // C++ casting
    int c = static_cast<int>(a);


    return 0;
}