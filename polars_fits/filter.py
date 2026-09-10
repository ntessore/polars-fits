import io
import json

import polars as pl

from typing import Any


class CfitsioFilter:
    """
    Class for parsing Polars expressions into cfitsio filters.
    """

    FUNCTION = {
        ("Trigonometry", "Cos"): "cos",
        ("Trigonometry", "Sin"): "sin",
    }

    INFIX = {
        ("Pow", "Generic"): "**",
    }

    PREFIX = {
        "Negate": "-",
    }

    BINARY = {
        "And": "&&",
        "Eq": "==",
        "Gt": ">",
        "GtEq": ">=",
        "Lt": "<",
        "LtEq": "<=",
        "Minus": "-",
        "Multiply": "*",
        "NotEq": "!=",
        "Or": "||",
        "Plus": "+",
        "TrueDivide": "/",
    }

    @classmethod
    def from_expr(cls, expr: pl.Expr) -> str | None:
        """Convert Polars expression to cfitsio filter."""
        return cls().parse(json.loads(expr.meta.serialize(format="json")))

    def parse(self, node: dict[str, Any]) -> str | None:
        """Parse an arbitrary node."""
        key, value = next(iter(node.items()))
        result = None
        method = getattr(self, f"parse_{key}", None)
        if method is not None:
            result = method(value)
        if result is None:
            raise ValueError(f"unsupported expression: {node}")
        return result

    def parse_BinaryExpr(self, value: dict[str, Any]) -> str | None:
        """Parse a BinaryExpr node with left, op, right elements."""
        op = self.BINARY.get(value["op"])
        if op is None:
            return None
        left = self.parse(value["left"])
        if left is None:
            return None
        right = self.parse(value["right"])
        if right is None:
            return None
        return f"({left}) {op} ({right})"

    def parse_Column(self, name: str) -> str:
        """Parse a Column node with column name."""
        return f"${name}$"

    def parse_Dyn(self, value: Any) -> str | None:
        """Parse a Dyn node."""
        return self.parse(value)

    def parse_Function(self, value: Any) -> str | None:
        """Parse a Function node."""
        function = value["function"]
        if isinstance(function, dict):
            function = next(iter(function.items()))
        prefix = infix = False
        if function in self.FUNCTION:
            fn = self.FUNCTION[function]
        elif function in self.INFIX:
            infix = True
            fn = self.INFIX[function]
        elif function in self.PREFIX:
            prefix = True
            fn = self.PREFIX[function]
        else:
            return None
        args = [self.parse(arg) for arg in value["input"]]
        if any(arg is None for arg in args):
            return None
        if infix:
            if len(args) != 2:
                return None
            return args[0] + fn + args[1]
        if prefix:
            if len(args) != 1:
                return None
            return fn + args[0]
        return fn + "(" + ",".join(args) + ")"

    def parse_Int(self, value: int) -> str:
        """Parse Int node."""
        return str(value)

    def parse_Int64(self, value: int) -> str:
        """Parse Int64 node."""
        return str(value)

    def parse_Literal(self, value: Any) -> str | None:
        """Parse a Literal node."""
        return self.parse(value)

    def parse_Scalar(self, value: Any) -> str | None:
        """Parse a Scalar node."""
        return self.parse(value)
