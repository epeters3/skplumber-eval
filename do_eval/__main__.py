import typing as t
from time import time
import random
import json

from skplumber import SKPlumber, Pipeline
from skplumber.metrics import f1macro
from skplumber.primitives import classifiers
import openml
from sklearn.model_selection import train_test_split
import ray

from do_eval.modeprimitive import ModePrimitive
from do_eval.utils import NpEncoder

# Define configuration

MAX_ROWS = 5000
MAX_COLS = 100
BUDGET = 30
NUM_DATASETS = 30
TASK_WHITELIST = [
    3,
    11,
    14,
    15,
    16,
    18,
    22,
    23,
    29,
    31,
    37,
    43,
    45,
    49,
    53,
    2079,
    3021,
    3022,
    3549,
    3560,
    3902,
    3903,
    3913,
    3917,
    3918,
    9946,
    9957,
    9971,
    9978,
    10093,
    10101,
    14954,
    125920,
    146800,
    146817,
    146819,
    146820,
    146821,
    146822,
]

# Keep NUM_DATASETS random whitelisted datasets
random.shuffle(TASK_WHITELIST)
TASK_WHITELIST = set(TASK_WHITELIST[:NUM_DATASETS])

# Define baselines

rf_baseline = Pipeline()
rf_baseline.add_step(classifiers["RandomForestClassifierPrimitive"])

mode_baseline = Pipeline()
mode_baseline.add_step(ModePrimitive)

# Define helpers


def get_task_ids() -> dict:
    return [
        task_id
        for task_id in openml.study.get_suite("OpenML-CC18").tasks
        if task_id in TASK_WHITELIST
    ]


def get_pipeline_spec(pipeline: Pipeline) -> list:
    steps = []
    for step in pipeline.steps:
        prim = step.primitive
        steps.append(
            {"primitive": prim.__class__.__name__, "params": prim.get_params()}
        )
    return steps


def train_and_score(estimator, X, y) -> dict:
    """
    `estimator` can be an instance of `skplumber.Pipeline` or
    `skplumber.SKPlumber`.
    """
    start = time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    score = f1macro(y_test, y_pred)
    total_time = time() - start
    pipeline_spec = get_pipeline_spec(
        estimator.best_pipeline if isinstance(estimator, SKPlumber) else estimator
    )
    return {"score": score, "time": total_time, "pipeline": pipeline_spec}

@ray.remote
def do_ablation_test(task_id) -> dict:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(
        f"now running ablation test on dataset '{dataset.name}' (task {task_id})..."
    )
    X, y = task.get_X_and_y(dataset_format="dataframe")
    n, p = X.shape

    rf_stats = train_and_score(rf_baseline, X, y)
    mode_stats = train_and_score(mode_baseline, X, y)
    with_tuning = train_and_score(
        SKPlumber("classification", BUDGET, exit_on_pipeline_error=False, tune=True),
        X,
        y,
    )
    no_tuning = train_and_score(
        SKPlumber("classification", BUDGET, exit_on_pipeline_error=False, tune=False),
        X,
        y,
    )

    return {
        "task_id": task_id,
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.name,
        "n_instances": n,
        "n_features": p,
        "n_classes": len(y.unique()),
        "rf_baseline": rf_stats,
        "mode_baseline": mode_stats,
        "with_tuning": with_tuning,
        "no_tuning": no_tuning,
    }


if __name__ == "__main__":
    # Initialize
    ray.init()
    current_results = []
    remaining_run_ids = []

    # Figure out what's been done already
    with open("evaluation-results.json") as rf:
        for line in rf:
            current_results.append(json.loads(line))
    finished_task_ids = {result["task_id"] for result in current_results}

    # Kick off all the experiments concurrently
    for task_id in get_task_ids():
        if task_id not in finished_task_ids:
            remaining_run_ids.append(do_ablation_test.remote(task_id))

    with open("evaluation-results.json", "a+") as wf:
        # Gather up the results of all the tests.
        while remaining_run_ids:
            # Get the next finished ablation test
            done_ids, remaining_run_ids = ray.wait(remaining_run_ids)
            result = ray.get(done_ids[0])
            # Write the results of this ablation test to the results file.
            json.dump(result, wf, cls=NpEncoder)
            wf.write("\n")
