import ASPLoader
from clingo import Control

class Atom:
    # -- Fields --
    # name : String
    def __init__(self, functor):
        if functor.isalnum():
            self.functor = functor
        else:
            raise ValueError("Only alphanumeric strings are allowed for atoms. [" + functor + "]")

    def __str__(self):
        # return "[A]"
        return self.functor

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_ASP(self):
        return self.functor


class Literal:
    # -- Fields --
    # atom : Atom
    # neg : Boolean
    def __init__(self, atom, neg=False):
        self.atom = atom
        self.neg = neg

    def __str__(self):
        if self.neg:
            prefix = "-"
        else:
            prefix = ""
        # return "[L]"
        return prefix + str(self.atom)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((self.neg, self.atom.name))

    def to_ASP(self):
        if self.neg:
            prefix = "-"
        else:
            prefix = ""
        return prefix + str(self.atom)

    def __repr__(self):
        return str(self)


class ExtLiteral:
    # -- Fields --
    # literal : Literal
    # naf : Boolean
    def __init__(self, literal, naf=False):
        self.literal = literal
        self.naf = naf

    def __str__(self):
        if self.naf:
            prefix = "not "
        else:
            prefix = ""
        # return "[E]"
        return prefix + str(self.literal)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_ASP(self):
        if self.naf:
            prefix = "not "
        else:
            prefix = ""
        return prefix + str(self.literal)

    def __repr__(self):
        return str(self)


class Formula:
    # -- Fields --
    # operator : Operator
    # inputFormulas : Formula list
    # inputTerms : ExtLiteral list
    def __init__(self, operator, input_formulas=(), input_terms=()):
        self.operator = operator
        self.input_formulas = input_formulas
        self.input_terms = input_terms

    def __str__(self):
        output = "{'operator': " + str(self.operator) + ", "
        if len(self.input_formulas) > 0:
            output += "'inputFormulas': ["
            for inputFormula in self.input_formulas:
                output += str(inputFormula) + ", "
            output = output[:-2] + "]"
        elif len(self.input_terms) > 0:
            output += "'inputTerms': ["
            for inputTerm in self.input_terms:
                output += str(inputTerm) + ", "
            output = output[:-2] + "]"
        return output + "}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def extract_literals(self, filter=None):
        literals = []
        if len(self.input_formulas) > 0:
            for inputFormula in self.input_formulas:
                literals.extend(inputFormula.extract_literals(filter))
        elif len(self.input_terms) > 0:
            for ext_literal in self.input_terms:
                if filter is None:
                    literals.append(ext_literal.predicate)
                elif ext_literal.naf == filter:
                    literals.append(ext_literal.predicate)
        return literals

    def extract_asserted_literals(self):
        return self.extract_literals(False)

    def extract_naf_literals(self):
        return self.extract_literals(True)

    def to_ASP(self, head=False):
        output = ""

        n = len(self.input_terms)

        if self.operator is Operator.NONE:
            if n > 1:
                raise ValueError("there should be only an atom here")
        elif self.operator is Operator.AND:
            if head is True:
                output += str(n) + "{"
        elif self.operator is Operator.OR:
            output += "1{"
        elif self.operator is Operator.CHOICE:
            output += "{"
        elif self.operator is Operator.XOR:
            output += "1{"
        else:
            raise ValueError("Not yet implemented")

        if len(self.input_formulas) > 0:
            for formula in self.input_formulas:
                output += formula.to_ASP() + ', '
            output = output[0:-2]
        else:
            if len(self.input_terms) > 1:
                for term in self.input_terms:
                    output += term.to_ASP() + ';'
                output = output[0:-1]
            else:
                output += self.input_terms[0].to_ASP()

        if self.operator is not Operator.NONE:
            if self.operator is Operator.AND:
                if head is True:
                    output += "}" + str(n)
            elif self.operator is Operator.OR:
                output += "}"
            elif self.operator is Operator.CHOICE:
                # TODO.md: choice parameters to be implemented
                output += "}" + str(n)
            elif self.operator is Operator.XOR:
                output += "}1"
            else:
                raise ValueError("Not yet implemented")

        return output

    def __repr__(self):
        return str(self)


class Operator:
    NONE = 0
    AND = 1
    OR = 2
    XOR = 3
    CHOICE = 4


class Rule:
    # -- Fields --
    # head : Formula
    # body : Formula
    def __init__(self, head=None, body=None):
        self.head = head
        self.body = body

    def __str__(self):
        return "{'body': " + str(self.body) + ", 'head': " + str(self.head) + "}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def extract_literals(self, filter=None):
        atoms = []
        if self.head is not None:
            atoms.extend(self.head.extract_literals(filter))
        if self.body is not None:
            atoms.extend(self.body.extract_literals(filter))
        return atoms

    def extract_asserted_literals(self):
        return self.extract_literals(False)

    def extract_naf_literals(self):
        return self.extract_literals(True)

    def is_norm_rule(self):
        return self.head is not None and self.body is not None

    def is_fact(self):
        return self.head is not None and self.body is None

    def is_constraint(self):
        return self.head is None and self.body is not None

    def to_ASP(self):
        output = ""
        if self.head is not None:
            output += self.head.to_ASP()
        if self.body is not None:
            output += " :- " + self.body.to_ASP()
        return output + "."


class Program:
    # -- Fields --
    # rule_list
    def __init__(self, rule_list=(), parsing_errors=None):
        self.rule_list = rule_list
        self.parsing_errors = parsing_errors
        self.running_errors = None
        self.answer_sets = []

    def to_ASP(self):
        output = ""
        for rule in self.rule_list:
            output += rule.to_ASP() + "\n"
        print(output)
        return output

    def __on_model(self, model):
        answer_set = []
        for atom in model.symbols(atoms="True"):
            answer_set.append(ASPLoader.parse_literal(str(atom)))
        self.answer_sets.append(answer_set)

    def solve(self):
        ctl = Control()
        ctl.configuration.solve.models = 0  # to obtain all answer sets
        ctl.add("base", [], self.to_ASP())
        ctl.ground([("base", [])])
        ctl.solve(on_model=self.__on_model)
        return self.answer_sets

    ## TODO.md: iterative solver


## Minimal ASP solver with clingo
## from https://sourceforge.net/p/potassco/mailman/message/36263330/
##
## import clingo
##
## clingo_args = []
## cc = clingo.Control(clingo_args)
## # if you want to load an external asp file:
## # cc.load(filename)
## cc.add('base', [], '''
## a(X) :- not b(X), d(X).
## b(X) :- not a(X), d(X).''')
## cc.add('base', [], "d(1;2;3).")
## cc.ground([("base",[])])

## def onmodel(m):
##  print(repr(m))
## cc.solve(on_model=onmodel)