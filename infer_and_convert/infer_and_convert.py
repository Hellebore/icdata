"""
Copyright (c) 2020 Aadit Kapoor
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import pandas as pd
import copy
import logging
import seaborn as sns
import os
from typing import List
import pandas as pd
import datetime
import numpy as np
from enum import Enum

# ---- TYPES -----

# Accepted types, feel free to add additional types
VALID_TYPES = {
    'str': np.str,
    'int': np.int,
    'float': np.float, 'object': np.object,
    "float64": np.float64,
    "float32": np.float32,
    "datetime": datetime.datetime,
    "int64": np.int64,
    "int32": np.int32,
    "byte": np.byte,
    "floating": np.float,
    "string": np.str,
    "mixed": np.str
}


class InferType(object):
    """
        InferType handles all the values that have to be inferred.
    """

    def __init__(self, value: List[object]):
        if not isinstance(value, list):
            raise TypeError("value should be an iterable.")
        self._value = value

    @property
    def value(self):
        return pd.api.types.infer_dtype(self._value)

    def validate(self) -> object:
        vk = {v: k for k, v in VALID_TYPES.items()}
        if self.value in list(VALID_TYPES.keys()):
            return VALID_TYPES[self.value]
        else:
            return None


class Type:
    """
        A type that be classified in the schema
    """

    def __init__(self, value: object, type: str):
        self._value = value
        self.type = type  # the specified type

    @property
    def value(self):
        return self._value

    def validate_type(self) -> bool:
        return self.type in VALID_TYPES.keys()

    def validate(self) -> bool:
        # object most probable: pd.DataFrame
        raise NotImplementedError

    def _is_instance(self) -> bool:
        if check := isinstance(self.value, type(VALID_TYPES.get(self.type, None))):
            if check is None:
                raise ValueError("None returned.")
            else:
                return check
        else:
            return check

    def __repr__(self) -> str:
        if self.validate_type():
            # It is a valid type
            return f"{type(self.value)}: {repr(self.value)}"
        else:
            raise TypeError("Not a valid type. Check VALID_TYPES.")

    def is_valid(self):
        check = None
        try:
            check = isinstance(self.value, type(
                VALID_TYPES.get(self.type, None)))
        except:
            print(f"Invalid type.")
        else:
            return check
# Maybe used


class StringType(Type):
    def __init__(self, value: str):
        super().__init__(value, type="str")

    def validate(self) -> bool:
        return self.is_valid()


class FloatType(Type):
    def __init__(self, value: str):
        super().__init__(value, type="float")

    def validate(self) -> bool:
        return self.is_valid()


class IntType(Type):
    def __init__(self, value: str):
        super().__init__(value, type="int")

    def validate(self) -> bool:
        return self.is_valid()


class ObjectType(Type):
    def __init__(self, value: str):
        super().__init__(value, type="object")

    def validate(self) -> bool:
        return self.is_valid()
# --- END TYPES ----------


class SchemaArgumentError(Exception):
    """Generic error for all Schema related issue"""
    pass


class Schema:
    """Defines the schema for a pandas dataframe"""

    def __init__(self, *args, columns=None):
        if not args or not columns:
            raise SchemaArgumentError("args and columns required.")
        if not all(list(map(lambda x: x in VALID_TYPES.keys(), args))) and not any(list(map(lambda x: x == "infer", args))):
            raise SchemaArgumentError(
                f"Invalid types. Check the types at {VALID_TYPES.keys()}")
        self.mapped = {}
        if len(args) == len(columns):
            for arg, col in zip(args, columns):
                # mapping column to type
                self.mapped[col] = arg
        else:
            raise SchemaArgumentError(
                "number of types does not match number of columns.")

    def __repr__(self):
        return f"Schema({self.mapped})"

    @property
    def columns(self):
        return list(self.mapped.keys())

    @property
    def types(self):
        return list(self.mapped.values())


def ic_data(df: pd.DataFrame, schema: Schema, inplace=False, verbose=False):
    """
    ic_data stands infer_and_convert a Pandas DataFrame. It works by detecting the supplied
    column name and its corresponding values. This method encapsulates the need to
    manually change the types.You can provide a datatype if you know what can of data
    is being passed or pass 'infer' to automatically infer the datatype.

    Arguments
    -----------
    df: pandas DataFrame
    schema: A valid Schema
    inplace: Preserve the datatype
    verbose: print log statements

    Returns
    -----------
    df: pandas DataFrame

    Example
    -----------
    >>> ic_data(df=sns.load_dataset("tips"), schema=schema, verbose=True)
    """

    new_df = None
    if not inplace:
        new_df = copy.copy(df)
    columns = df.columns.tolist()
    if len(columns) != len(schema.columns):
        raise SchemaArgumentError("Number of columns do not match.")
    if verbose:
        print(f"Gathered columns: {schema.columns}")
    if columns != schema.columns:
        raise SchemaArgumentError("Columns are not identical.")

    for t, c in zip(schema.types, columns):
        if verbose:
            print(f"{t}: {c}")
        if t not in VALID_TYPES.keys() and t != "infer":
            raise SchemaArgumentError(
                f"{t} not a valid type. Check {list(VALID_TYPES.keys())}")
        if t == "infer":
            # infer types and convert
            it = InferType(df[c].values.tolist()).validate()
            if inplace:
                df[c] = df[c].astype(it)
            else:
                new_df[c] = new_df[c].astype(it)
            if verbose:
                print(f"{c}: changing to inferred type: {it}: [OK]")
        elif isinstance(df[c].dtype, VALID_TYPES[t]):
            if verbose:
                print(f"{c} column: [UNCHANGED]")
        else:
            if inplace:
                df[c] = df[c].astype(VALID_TYPES[t])
            else:
                new_df[c] = new_df[c].astype(t)
            if verbose:
                print(f"{c}: [{df[c].dtype} => {VALID_TYPES[t]}: [OK]")
    if not inplace:
        return new_df


"""
EXAMPLE
---------

df = sns.load_dataset("tips")

schema = Schema("str", "str", "str", "str", "str", "str",
                "int", columns=df.columns.tolist())

ic_data(df, schema, verbose=True)
"""
