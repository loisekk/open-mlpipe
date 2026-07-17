"""DataSplitter stage — stratified or random train/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.typing import TaskType


class SplitStage(Stage):
    name = "split"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.clean_data
        config = ctx.config
        target = ctx.target_column

        X = df.drop(columns=[target])
        y = df[target]

        # Encode target if classification and string
        if ctx.task_type == TaskType.CLASSIFICATION and y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target)
            ctx._label_encoder = le

        # Determine stratification
        stratify = None
        if ctx.task_type == TaskType.CLASSIFICATION:
            stratify = y

        test_size = config.data.test_size
        if len(df) < 500:
            test_size = 0.1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config.data.random_state,
            stratify=stratify,
        )

        ctx.X_train = X_train
        ctx.X_test = X_test
        ctx.y_train = y_train
        ctx.y_test = y_test

        # Update column types after split
        ctx.numeric_columns = [c for c in ctx.numeric_columns if c in X_train.columns]
        ctx.categorical_columns = [c for c in ctx.categorical_columns if c in X_train.columns]

        return ctx
