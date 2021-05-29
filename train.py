import argparse

from models.hl import HL_net

"""
Program to (Re-)train models in a network with a meta-level filter using explicit rules based on the output of the models in our network
Example: Train model A with labels X and Y. Filter data from labels X if model B infers label Z on the datapoint.
More concrete example: Train model A with labels Apple and Banana. 
                       Filter data from Apples; if object in datapoint is anything besides red or green, inferred by model B.

Example ASP rules:
#external modelb(red;green).

apple :- modelb(red).
apple :- modelb(green).
banana :- not apple.

Now we can build a dataloader that loads in the dataset from modela regularly but then uses this ASP program to filter out datapoints.

Open questions:
- Can we train all models in the network at the same time?
- How 'automatic' is this training?
- How do we train model a from modelb and modelb from model a with one generic call?

Notes:
- Here we train a specialization that we have discovered from our model exploration.
- We can use the inference code, infer.py.
- Do we simply use folder names as classnames?
"""


training_program = """
#external hl(character;chemicalstructure;drawing;flowchart;genesequence;graph;math;programlisting;table).

chemichal :- hl(programlisting).
chemichal :- hl(drawing).
chemichal :- hl(genesequence).
chemichal :- hl(chemicalstructure).

nonchemical :- not chemical.
"""

if __name__ == "__main__":
    # Example option 1
    # We train an individual network with one given ASP
    # More control, more steps
    model_mapping = {"hl": HL_net()}
    orchestrator = orchestrator(training_program, model_mapping)
    orchestrator.train()

    # Example option 2
    # We map ASP's to each network that we want to train
    # Less control, more steps
    model_mapping = {"hl": HL_net()}
    training_program_mapping = {"cnc": training_program}
    orchestrator = orchestrator(model_mapping)
    orchestrator.train(training_program_mapping)

    