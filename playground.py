import clingo

prog1 = """
#external cnc(chemical;nonchemical).
#external hl(character;chemicalstructure;drawing;flowchart;genesequence;graph;math;programlisting;table).
#external pc(none;one;multi).

% If both cnc and hl infer chemical, the image is chemical
chemicalimage :- cnc(chemical), hl(chemicalstructure).

% If either cnc or hl infer non chemical, the image is not chemical
nonchemicalimage :- cnc(nonchemical).
nonchemicalimage :- not hl(chemicalstructure).

% If there is a chemicalimage and multiple pixel clusters, we have one chemical depiction
onechemicalstructure :- chemicalimage, pc(one).

% If there is a chemicalimage and multiple pixel clusters, we have many chemical depiction
manychemicalstructure :- chemicalimage, pc(multi).
"""

prog2 = """#external model(none;apple;banana). fruit :- model(apple)."""

prog3 = """
a(1..5).
adj(X,Y) :- a(X), a(Y), Y-X==1.
adj(X,Y) :- a(X), a(Y), Y-X==2.
trans(X,Y,Z) :- adj(X,Y), adj(Y,Z), adj(X,Z).
"""

prog4 = """
#external cnc(inference, confidence).

outcome :- cnc(inference, confidence), inference == chemical, confidence == 100.
"""

ctl = clingo.Control()
ctl.add("base", [], prog4)
ctl.ground([("base", [])])

# ctl.assign_external(clingo.Function("cnc", [clingo.Function("inference", [clingo.Function("chemical")], True)], True), True)

ctl.assign_external(clingo.Function("cnc", [clingo.Function("chemical"), clingo.Function("100")], True), True)
# ctl.assign_external(clingo.Function("cnc", [clingo.Function("chemical")], True), True)
# ctl.assign_external(clingo.Function("hl", [clingo.Function("chemicalstructure")], True), True)
# ctl.assign_external(clingo.Function("model", [clingo.Function("apple")], True), True)
ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))