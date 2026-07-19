"""DataLoader stage — loads data and sets up context."""

from __future__ import annotations

from open_mlpipe.config.defaults import SmartDefaults
from open_mlpipe.core.context import PipelineContext
from open_mlpipe.core.stage import Stage
from open_mlpipe.utils.io import load_data
from open_mlpipe.utils.typing import TaskType


class DataLoaderStage(Stage):
    name = "load"
    version = "1.0"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        config = ctx.config
        if config is None:
            return ctx
        df = load_data(
            config.data.path,
            sep=config.data.separator,
            encoding=config.data.encoding,
        )

        if config.data.max_rows:
            df = df.head(config.data.max_rows)

        ctx.raw_data = df

        # Warn on very small datasets
        if len(df) < 500:
            ctx.metrics["small_data_warning"] = (
                f"Dataset has only {len(df)} rows. "
                "Consider collecting more data for reliable results. "
                "Pipeline will use simpler models with strong regularization."
            )
        ctx.clean_data = df.copy()

        # Detect target
        target: str | None = config.data.target
        if target is None:
            target = SmartDefaults.detect_target_column(df)
            if target is None:
                target = str(df.columns[-1])
        ctx.target_column = target

        # Detect task
        task_str = config.task
        if task_str == "auto":
            task = SmartDefaults.detect_task(df[target])
        else:
            task = TaskType(task_str)
        ctx.task_type = task

        # Detect column types
        ctx.column_types = SmartDefaults.detect_column_types(
            df.drop(columns=[target], errors="ignore")
        )

        # Classify columns
        numeric_cols: list[str] = []
        cat_cols: list[str] = []
        datetime_cols: list[str] = []
        for col, ct in ctx.column_types.items():
            if ct.value == "numeric":
                numeric_cols.append(col)
            elif ct.value in ("categorical", "boolean"):
                cat_cols.append(col)
            elif ct.value == "datetime":
                datetime_cols.append(col)

        ctx.numeric_columns = numeric_cols
        ctx.categorical_columns = cat_cols
        ctx.datetime_columns = datetime_cols

        # Detect skewed/normal numeric columns
        if numeric_cols:
            skewed, normal = SmartDefaults.detect_skewed_columns(df[numeric_cols])
            ctx.skewed_columns = skewed
            ctx.normal_columns = normal

        # Detect high/low cardinality categoricals
        for col in cat_cols:
            nuniq = df[col].nunique()
            if nuniq > config.preprocessing.high_cardinality_threshold:
                ctx.high_cardinality_columns.append(col)
            else:
                ctx.low_cardinality_columns.append(col)

        return ctx

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.raw_data is not None
