% We define our neural rules
cnc(chemical).
hl(chemicalstructure).

% If both cnc and hl infer chemical, the image is chemical
chemicalimage :- cnc(chemical), hl(chemicalstructure).

% If either cnc or hl infer non chemical, the image is not chemical
nonchemicalimage :- cnc(nonchemical).
nonchemicalimage :- not hl(chemicalstructure).