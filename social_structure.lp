#external cnc(chemical;nonchemical).
#external hl(character;chemicalstructure;drawing;flowchart;genesequence;graph;math;programlisting;table).
#external pc(none;one;multi).

% If both cnc and hl infer chemical, the image is chemical
chemicalimage :- cnc(chemical), hl(chemicalstructure).

% If either cnc or hl infer non chemical, the image is not chemical
nonchemicalimage :- cnc(nonchemical).
nonchemicalimage :- not hl(chemicalstructure).

% If there is a chemicalimage and multiple pixel clusters, we have one chemical depiction
onechemicalimages :- chemicalimage, pc(one).

% If there is a chemicalimage and multiple pixel clusters, we have many chemical depiction
manychemicalimages :- chemicalimage, pc(multi).