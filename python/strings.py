name = "John"
age = 23

# 1. Old-style formatting (C-style)
print("My name is %s and I am %d years old." % (name, age))

# 2. str.format() method
print("My name is {} and I am {} years old.".format(name, age))

# 3. f-string (Python 3.6+)
print(f"My name is {name} and I am {age} years old.")