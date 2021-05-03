# ASP parser and solver in Python

simple parser for ASP in Python wrapping the solver `clingo`.

## dependencies

by means of conda:
```
conda install -c potassco clingo
conda install -c conda-forge antlr-python-runtime
``` 

otherwise: 
**clingo** (the ASP solver)
- download the sources from [https://github.com/potassco/clingo]
- compile the python library following the instructions
- add the library to the project dependencies 

To work with the grammar, download the ANTLR complete JAR file and the necessary Python module.
The easiest way is by using pip.

```
pip install antlr4-python2-runtime
```

