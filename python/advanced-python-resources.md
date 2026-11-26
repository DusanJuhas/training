# Advanced Python – Learning Resources

## Books

- **Fluent Python, 2nd ed. – Luciano Ramalho (O'Reilly, 2022)**
  Start here. Written for Python 3.10, it covers the data model and special methods, functions as objects, OOP idioms (protocols, static typing), control flow (context managers, generators, async/await, thread/process pools) and metaprogramming (descriptors, class decorators). Especially useful for developers coming from C/C++/Java. Caveat: the async chapters are somewhat dated now.

- **Effective Python – Brett Slatkin (latest edition)**
  Short, numbered best-practice items. Easy to read in short sittings and fills gaps that Fluent Python leaves.

- **Architecture Patterns with Python – Harry Percival & Bob Gregory**
  Design rather than syntax: domain-driven design, repository, unit of work, event-driven architecture. Free online at https://www.cosmicpython.com

- **High Performance Python – Micha Gorelick & Ian Ozsvald**
  Profiling, NumPy, multiprocessing, Cython. Relevant for data and AI work.

- **Robust Python – Patrick Viafore**
  Type hints and maintainable large codebases.

- **CPython Internals – Anthony Shaw**
  How the interpreter works: bytecode, memory management, the GIL. A C background helps.

- **Python Cookbook, 3rd ed. – David Beazley & Brian K. Jones**
  Older, but the recipes on generators, decorators and metaclasses still hold up.

## Online

- **David Beazley – "Python Mastery"** (free on GitHub: `dabeaz-course/python-mastery`)
  Material from his advanced paid workshop. One of the best hands-on options.

- **Conference talks (YouTube)**
  - James Powell – "So you want to be a Python expert?" (decorators, generators, context managers, metaclasses)
  - Raymond Hettinger – "Transforming Code into Beautiful, Idiomatic Python"
  - Raymond Hettinger – "Beyond PEP 8"

- **Official Python documentation**
  - Data Model reference: https://docs.python.org/3/reference/datamodel.html
  - Descriptor HowTo Guide: https://docs.python.org/3/howto/descriptor.html

- **Real Python** (https://realpython.com) – advanced tutorials and learning paths on async, typing and packaging.

- **O'Reilly Learning** – check whether your employer provides a subscription; most of the books above are included.

## Suggested order

1. Fluent Python + Beazley's Python Mastery for practice
2. Effective Python
3. Then choose by goal:
   - System design → Architecture Patterns with Python
   - Performance / NLP / GenAI → High Performance Python
