#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to read ranking objects for datasets
#######################################
from contextlib import redirect_stdout
import json
import pickle as pkl
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tsadams.datasets.dataset import Entity
from tsadams.model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES, AUTOTSAD_ENTITIES
from tsadams.utils.eval_utils import get_aggregated_ranks, get_metric_names
from tsadams.utils.model_selection_utils import rank_models
from tsadams.utils.utils import get_args_from_cmdline
from tsadams.utils.set_all_seeds import set_all_seeds


def _compute_aggregate_rank(performance_matrix: pd.DataFrame, use_all_ranks: bool, top_k: int, top_kr: int, metric: str) -> str:
    # Compute the robust rank aggregation for a dataset to choose the best model
    # (top-1 model selection). We assume here that "minimum influence metric"
    # in the paper corresponds to the "Most Reliable Metric MS" rank
    # aggregation in the code.
    ranks_by_metrics, *_ = rank_models(performance_matrix)

    # Evaluation metric should always be 'Best F-1'
    evaluation_metric = "Best F-1"
    metric_names = get_metric_names(performance_matrix.columns, evaluation_metric=evaluation_metric)
    ranks = np.concatenate([ranks_by_metrics[:8, :], ranks_by_metrics[8::3, :]],axis=0).astype(int)

    if not use_all_ranks:
        filtered_idxs = [
            i for i, mn in enumerate(metric_names)
            if ((len(mn.split('_')) == 3) and (mn.split(
                '_')[2] in ['noise', 'scale', 'cutoff', 'contextual', 'average']))
        ]
        ranks = ranks[filtered_idxs, :]

    # Rank-aggregation based Model Selection
    *_, top_reliability_metric_rank, _, _, _ = get_aggregated_ranks(ranks=ranks, metric=metric, top_k=top_k, top_kr=top_kr)
    cols = sorted([
        c for c in metric_names
        if ((len(c.split('_')) == 3) and (c.split('_')[2] in ['noise', 'scale', 'cutoff', 'contextual', 'average']))
    ])
    method_ranking = performance_matrix.index[top_reliability_metric_rank].tolist()
    print("Filtered performance matrix:")
    print(performance_matrix[cols])
    print("Most Reliable Metric MS ranking:", method_ranking)

    # Most Reliable Metric MS (return top-1 method)
    return method_ranking[0]
    

def _plot_entity(entity: Entity, ax=None) -> None:
    if ax is None:
        ax = plt.gca()

    if len(entity.Y.shape) == 1:
        data = entity.Y
    if len(entity.Y.shape) == 2:
        if entity.Y.shape[0] > 1:
            print(f"Can only plot univariate time series (shape={entity.shape})!")
            return
        
        data = entity.Y.ravel()
    ax.plot(data, label=entity.name)

    if np.any(entity.labels):
        labels = np.r_[0, entity.labels.ravel(), 0]
        anomalies = np.c_[
            np.nonzero(np.diff(labels) == 1)[0],
            np.nonzero(np.diff(labels) == -1)[0]
        ]
        y0, y1 = ax.get_ylim()
        for b, e in anomalies:
            height = y1 - y0
            ax.add_patch(
                Rectangle((b, y0), e-b, height, edgecolor="orange", facecolor="yellow", alpha=0.5)
            )


def load_model_wrapper(dataset, entity, args, plot=False):
    print()
    print(42 * "=")
    print(f"Loading results for dataset {dataset} on entity: {entity}")
    print(42 * "=")

    ro_path = Path(args["results_path"]) / dataset / f"ranking_obj_{entity}.data"
    if not ro_path.exists():
        print(f"No results for {entity} found!")
        return

    with ro_path.open('rb') as f:
        ranking_obj = pkl.load(f)
    
    entity_name = ranking_obj.test_data.entities[0].name

    autotsad_result_path = Path("scorings") / f"tsadams-{entity_name}-mim"
    autotsad_result_path.mkdir(exist_ok=True)
    scorings_path = autotsad_result_path / f"scores.csv"
    with (autotsad_result_path / "config.json").open("w") as fh:
        json.dump(args, fh)

    with (autotsad_result_path / "execution.log").open("w") as fh, redirect_stdout(fh):
        print(ranking_obj.test_data)
        print(f"Entity-name={entity_name}")

        if not hasattr(ranking_obj, "models_performance_matrix") or ranking_obj.models_performance_matrix is None:
            print("models_performance_matrix is missing, trying to recover other performance metrics")
            metrics = []
            if hasattr(ranking_obj, "models_evaluation_metrics"):
                print(f"{ranking_obj.models_evaluation_metrics=}")
                metrics.append(ranking_obj.models_evaluation_metrics)
            else:
                print("models_evaluation_metrics is missing!")
            if hasattr(ranking_obj, "models_forecasting_metrics"):
                print(f"{ranking_obj.models_forecasting_metrics=}")
                metrics.append(ranking_obj.models_forecasting_metrics)
            else:
                print("models_forecasting_metrics is missing!")
            if hasattr(ranking_obj, "models_centrality"):
                print(f"{ranking_obj.models_centrality=}")
                metrics.append(ranking_obj.models_centrality)
            else:
                print("models_centrality is missing!")
            if hasattr(ranking_obj, "models_synthetic_anomlies"):
                print(f"{ranking_obj.models_synthetic_anomlies=}")
                metrics.append(ranking_obj.models_synthetic_anomlies)
            else:
                print("models_synthetic_anomlies is missing!")

            if len(metrics) == 0:
                np.savetxt(scorings_path, np.zeros(ranking_obj.test_data.entities[0].Y.shape[1]), delimiter=",")
                print(f"\n{dataset} {entity}: Could not load ranking results because the performance matrix is missing completely!")
                print("==========================================")
                return

            ranking_obj.models_performance_matrix = pd.concat(metrics, axis=1)
        print(f"{ranking_obj.models_performance_matrix=}")

        print("\nChecking scorings")
        for m in ranking_obj.MODEL_NAMES:
            if m not in ranking_obj.predictions:
                print(f"  {m} is missing scorings!")
                continue
            scoring = ranking_obj.predictions[m]['entity_scores']
            print(f"  {m} found scorings (shape ={scoring.shape})")
            # for univariate TS (autotsad), we can do scoring.ravel()
            # print(scoring)
        best_method = _compute_aggregate_rank(ranking_obj.models_performance_matrix,
                                            use_all_ranks=args["use_all_ranks"],
                                            metric=args["metric"],
                                            top_k=args["top_k"],
                                            top_kr=args["top_kr"])
        print(f"Best method for dataset {ranking_obj.test_data.entities[0].name}: {best_method}")
        scoring = ranking_obj.predictions[best_method]['entity_scores'].ravel()
        np.savetxt(scorings_path, scoring, delimiter=",")

    if plot:
        fig, axs = plt.subplots(2, 2, sharex="col")
        axs[0, 0].set_title(f"{dataset} - {entity}")
        _plot_entity(ranking_obj.train_data.entities[0], ax=axs[0, 0])
        axs[0, 0].legend()
        _plot_entity(ranking_obj.test_data.entities[0], ax=axs[0, 1])
        axs[0, 1].legend()
        axs[1, 1].plot(scoring, label=best_method, color="orange")
        axs[1, 1].legend()
        plt.show()


def main(datasets=['anomaly_archive', 'smd'], entities=[ANOMALY_ARCHIVE_ENTITIES, MACHINES], args=None):
    if args is None:
        args = get_args_from_cmdline()
    
    set_all_seeds(args['random_seed']) # Reduce randomness

    scorings_path = Path("scorings")
    scorings_path.mkdir(exist_ok=True)

    for d_i, dataset in enumerate(datasets):
        _ = Parallel(n_jobs=args['n_jobs'])(
            delayed(load_model_wrapper)(dataset, entities, args, plot=False)
            for entities in entities[d_i]
        )


def load_pooled_results() -> pd.DataFrame:
    args = get_args_from_cmdline()
    
    set_all_seeds(args['random_seed']) # Reduce randomness

    results_path = Path(args["results_path"]) / "aggregate_stats.pkl"
    with results_path.open("rb") as fh:
        results = pkl.load(fh)

    items = []
    for dataset in results:
        for i in range(args["n_validation_splits"]):
            record = dict((k, v[i]) for k, v in results[dataset].items())
            record["Split"] = i
            record["Dataset"]= dataset
            items.append(record)

    df = pd.DataFrame(items)
    print(df.columns)
    # Assumption: "minimum influence metric" in the paper corresponds to "Most Reliable Metric MS" in the code
    cols = [c for c in df.columns if c == "Split" or c == "Dataset" or "Most Reliable" in c]
    print(df[cols])
    return df


if __name__ == '__main__':
    # main(['autotsad'], [['NASA-MSL=C-2']])
    # main(['autotsad'], [['KDD-TSAD=174_UCR_Anomaly_insectEPG2']])
    main(['autotsad'], [AUTOTSAD_ENTITIES])
    # load_pooled_results()
