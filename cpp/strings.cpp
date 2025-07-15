// cpp/strings.cpp
#include<iostream>
#include<string>
std::string name = "John";

int main(){
    const char* name = "John";
    int age = 23;

    //1. C-style printf formatting (closest to Python's % operator)
    std::printf("My name is %s and I am %d years old.\n", name, age);

    //2. C++ style with cout
    std::cout << "My name is " << name << " and I am " << age << " years old.\n";

    return 0;
}