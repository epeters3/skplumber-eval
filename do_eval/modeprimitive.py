from skplumber.primitives.primitive import Primitive
from skplumber.consts import PrimitiveType
import pandas as pd


class ModePrimitive(Primitive):
    """
    Baseline classifier that always predicts the mode.
    """

    primitive_type = PrimitiveType.CLASSIFIER
    param_metas = {}

    def fit(self, X, y) -> None:
        self.pred = y.mode()[0]

    def produce(self, X) -> None:
        return pd.Series([self.pred for _ in range(len(X.index))])
