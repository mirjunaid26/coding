# 🧠 Coding Practice: C++ & Python

This repository contains C++ and Python practice programs side by side to help you learn programming fundamentals in both languages.  
Inspired by [LearnCpp.com](https://www.learncpp.com) and [LearnPython.org](https://www.learnpython.org).

---

## 📁 Repository Structure

```
coding/
├── cpp/              # C++ examples
│   ├── strings.cpp
│   └── ...
├── python/           # Python examples
│   ├── strings.py
│   └── ...
└── README.md
```

Each file in `cpp/` has a corresponding example in `python/` showing the same concept in both languages.

---

## 🚀 How to Run

### 🧰 Prerequisites

- A **C++ compiler**: `g++`, `clang++`, or MSVC
- **Python 3** installed (`python3` or `python`)

---

### 🏃 Run C++ Example

```bash
g++ cpp/strings.cpp -o strings
./strings
```

---

### 🏃 Run Python Example

```bash
python3 python/strings.py
```

---

## 📚 Topics Covered

We will include parallel examples for:

- Variables and types
- Strings and input/output
- Conditionals (`if`, `else`)
- Loops (`for`, `while`)
- Functions
- Lists/arrays
- Dictionaries/maps
- Classes and objects
- File I/O
- Error handling

---

## 📘 Learn More

- [LearnCpp.com](https://www.learncpp.com) — Great resource for mastering C++
- [LearnPython.org](https://www.learnpython.org) — Interactive tutorials for Python

---

## Why C++ is Faster than Python

### 1. Compiled vs. Interpreted
- **C++** is a compiled language. It is translated into machine code before running, allowing the CPU to execute instructions very efficiently.
- **Python** is an interpreted language. The Python interpreter executes the code at runtime, adding overhead for parsing and executing each instruction.

### 2. Static vs. Dynamic Typing
- **C++** uses static typing: variable types are known at compile time, enabling better optimization.
- **Python** uses dynamic typing: variable types are determined at runtime, which introduces extra overhead.

### 3. Memory Management
- **C++** provides manual control over memory allocation and deallocation, allowing fine-tuned performance optimizations.
- **Python** uses automatic garbage collection, which adds runtime overhead.

### 4. Low-Level Access
- **C++** allows direct access to hardware and system resources like pointers and SIMD instructions.
- **Python** abstracts low-level details, focusing on ease of use and safety.

### 5. Optimization
- C++ compilers perform extensive compile-time optimizations like inlining and loop unrolling.
- Python relies on interpreter optimizations and often requires external tools (e.g., PyPy, Cython) for speed improvements.

### Summary

| Feature            | C++                       | Python                    |
|--------------------|---------------------------|---------------------------|
| Execution          | Compiled to machine code  | Interpreted at runtime    |
| Typing             | Static (compile-time)     | Dynamic (runtime)         |
| Memory management  | Manual                    | Automatic (garbage collection) |
| Low-level control  | Yes                       | Limited                   |
| Speed              | Generally much faster     | Slower due to overhead    |

---

## 🤝 Contributing

Want to add more concepts or fix something?

1. Fork this repository  
2. Create a new branch (`feature/your-topic`)  
3. Add your examples to both `cpp/` and `python/`  
4. Open a Pull Request

Let’s build this together!

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.

---

Happy coding! 👨‍💻👩‍💻