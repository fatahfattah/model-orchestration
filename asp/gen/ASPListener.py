# Generated from /Users/giovanni/surfdrive/dev/pycosmos/ASP-parser/ASP.g4 by ANTLR 4.9
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ASPParser import ASPParser
else:
    from ASPParser import ASPParser

# This class defines a complete listener for a parse tree produced by ASPParser.
class ASPListener(ParseTreeListener):

    # Enter a parse tree produced by ASPParser#program.
    def enterProgram(self, ctx:ASPParser.ProgramContext):
        pass

    # Exit a parse tree produced by ASPParser#program.
    def exitProgram(self, ctx:ASPParser.ProgramContext):
        pass


    # Enter a parse tree produced by ASPParser#asprule.
    def enterAsprule(self, ctx:ASPParser.AspruleContext):
        pass

    # Exit a parse tree produced by ASPParser#asprule.
    def exitAsprule(self, ctx:ASPParser.AspruleContext):
        pass


    # Enter a parse tree produced by ASPParser#aspfact.
    def enterAspfact(self, ctx:ASPParser.AspfactContext):
        pass

    # Exit a parse tree produced by ASPParser#aspfact.
    def exitAspfact(self, ctx:ASPParser.AspfactContext):
        pass


    # Enter a parse tree produced by ASPParser#rangedef.
    def enterRangedef(self, ctx:ASPParser.RangedefContext):
        pass

    # Exit a parse tree produced by ASPParser#rangedef.
    def exitRangedef(self, ctx:ASPParser.RangedefContext):
        pass


    # Enter a parse tree produced by ASPParser#normrule.
    def enterNormrule(self, ctx:ASPParser.NormruleContext):
        pass

    # Exit a parse tree produced by ASPParser#normrule.
    def exitNormrule(self, ctx:ASPParser.NormruleContext):
        pass


    # Enter a parse tree produced by ASPParser#constraint.
    def enterConstraint(self, ctx:ASPParser.ConstraintContext):
        pass

    # Exit a parse tree produced by ASPParser#constraint.
    def exitConstraint(self, ctx:ASPParser.ConstraintContext):
        pass


    # Enter a parse tree produced by ASPParser#choice.
    def enterChoice(self, ctx:ASPParser.ChoiceContext):
        pass

    # Exit a parse tree produced by ASPParser#choice.
    def exitChoice(self, ctx:ASPParser.ChoiceContext):
        pass


    # Enter a parse tree produced by ASPParser#head.
    def enterHead(self, ctx:ASPParser.HeadContext):
        pass

    # Exit a parse tree produced by ASPParser#head.
    def exitHead(self, ctx:ASPParser.HeadContext):
        pass


    # Enter a parse tree produced by ASPParser#body.
    def enterBody(self, ctx:ASPParser.BodyContext):
        pass

    # Exit a parse tree produced by ASPParser#body.
    def exitBody(self, ctx:ASPParser.BodyContext):
        pass


    # Enter a parse tree produced by ASPParser#list_literals.
    def enterList_literals(self, ctx:ASPParser.List_literalsContext):
        pass

    # Exit a parse tree produced by ASPParser#list_literals.
    def exitList_literals(self, ctx:ASPParser.List_literalsContext):
        pass


    # Enter a parse tree produced by ASPParser#list_ext_literals_expressions.
    def enterList_ext_literals_expressions(self, ctx:ASPParser.List_ext_literals_expressionsContext):
        pass

    # Exit a parse tree produced by ASPParser#list_ext_literals_expressions.
    def exitList_ext_literals_expressions(self, ctx:ASPParser.List_ext_literals_expressionsContext):
        pass


    # Enter a parse tree produced by ASPParser#expression.
    def enterExpression(self, ctx:ASPParser.ExpressionContext):
        pass

    # Exit a parse tree produced by ASPParser#expression.
    def exitExpression(self, ctx:ASPParser.ExpressionContext):
        pass


    # Enter a parse tree produced by ASPParser#num_expression.
    def enterNum_expression(self, ctx:ASPParser.Num_expressionContext):
        pass

    # Exit a parse tree produced by ASPParser#num_expression.
    def exitNum_expression(self, ctx:ASPParser.Num_expressionContext):
        pass


    # Enter a parse tree produced by ASPParser#ext_literal.
    def enterExt_literal(self, ctx:ASPParser.Ext_literalContext):
        pass

    # Exit a parse tree produced by ASPParser#ext_literal.
    def exitExt_literal(self, ctx:ASPParser.Ext_literalContext):
        pass


    # Enter a parse tree produced by ASPParser#literal.
    def enterLiteral(self, ctx:ASPParser.LiteralContext):
        pass

    # Exit a parse tree produced by ASPParser#literal.
    def exitLiteral(self, ctx:ASPParser.LiteralContext):
        pass


    # Enter a parse tree produced by ASPParser#pos_literal.
    def enterPos_literal(self, ctx:ASPParser.Pos_literalContext):
        pass

    # Exit a parse tree produced by ASPParser#pos_literal.
    def exitPos_literal(self, ctx:ASPParser.Pos_literalContext):
        pass


    # Enter a parse tree produced by ASPParser#list_parameters.
    def enterList_parameters(self, ctx:ASPParser.List_parametersContext):
        pass

    # Exit a parse tree produced by ASPParser#list_parameters.
    def exitList_parameters(self, ctx:ASPParser.List_parametersContext):
        pass


    # Enter a parse tree produced by ASPParser#predicate.
    def enterPredicate(self, ctx:ASPParser.PredicateContext):
        pass

    # Exit a parse tree produced by ASPParser#predicate.
    def exitPredicate(self, ctx:ASPParser.PredicateContext):
        pass


    # Enter a parse tree produced by ASPParser#identifier.
    def enterIdentifier(self, ctx:ASPParser.IdentifierContext):
        pass

    # Exit a parse tree produced by ASPParser#identifier.
    def exitIdentifier(self, ctx:ASPParser.IdentifierContext):
        pass


    # Enter a parse tree produced by ASPParser#constant.
    def enterConstant(self, ctx:ASPParser.ConstantContext):
        pass

    # Exit a parse tree produced by ASPParser#constant.
    def exitConstant(self, ctx:ASPParser.ConstantContext):
        pass


    # Enter a parse tree produced by ASPParser#variable.
    def enterVariable(self, ctx:ASPParser.VariableContext):
        pass

    # Exit a parse tree produced by ASPParser#variable.
    def exitVariable(self, ctx:ASPParser.VariableContext):
        pass



del ASPParser