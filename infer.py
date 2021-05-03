import sys
sys.path.append('asp')

from dataloader import load_input

from orchestrator import Orchestrator

from models.cnc import CNC_net
from models.hl import HL_net


"""
Program to make an inference using given ASP rules and nn's
x Define program
- Load models
- Make inferences
- Parse outputs
- Retrieve SM
"""

inference_program = """
% We define our neural rules
nn(cnc, image, [chemical, nonchemical]).
nn(hl, image, ['character', 'chemicalstructure', 'drawing', 'flowchart', 'genesequence', 'graph', 'math', 'programlisting', 'table']).

% If both cnc and hl infer chemical, the image is chemical
chemicalimage :- cnc(chemical), hl(chemicalstructure).

% If either cnc or hl infer non chemical, the image is not chemical
nonchemicalimage :- cnc(nonchemical).
nonchemicalimage :- not hl(chemicalstructure).
"""

if __name__ == "__main__":
    print(f"Our program is defined as:\n{inference_program}")
    print(f"Now we load in our nn's")
    cnc = CNC_net()
    hl = HL_net()

    model_mapping = {"cnc":cnc,
                     "hl": hl}

    # Initialize our inputs dictionary and process the paths into data tensors
    inputs_dict = {"image": "example_image.tif"}
    inputs_tensor_dict = {name:load_input(name, path) for name, path in inputs_dict.items()}

    orchestrator = Orchestrator(inference_program, model_mapping)
    print(repr(orchestrator))

    answer_sets = orchestrator.infer(inputs_tensor_dict)

    print(f"Answer sets: {answer_sets}")