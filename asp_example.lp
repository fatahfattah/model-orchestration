bird(tux). penguin(tux).
bird(tweety). chicken(tweety).

flies(X) :- bird(X), not -flies(X).
-flies(X) :- bird(X), not flies(X).
-flies(X) :- penguin(X).
:- flies(tux), -flies(tux).
:- flies(tweety), -flies(tweety).
