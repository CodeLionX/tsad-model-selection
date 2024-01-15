import sys
from tsadams.model_trainer.entities import AUTOTSAD_ENTITIES

sys.path.append(".")

from download_data import main as download_data
from train_all_models import main as train_all_models
from evaluate_all_models import main as evaluate_all_models
from compute_pooled_results import main as compute_pooled_results


def main():
    DATASETS = ['autotsad']
    ENTITIES = [['GutenTAG=ecg-diff-count-1.semi-supervised']]
    # DATASETS = ['smd']
    # ENTITIES = [['machine-2-8']]

    # download_data(DATASETS)
    train_all_models(DATASETS, ENTITIES)
    evaluate_all_models(DATASETS, ENTITIES)
    compute_pooled_results(DATASETS)


if __name__ == "__main__":
    main()
