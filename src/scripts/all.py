import sys
from tsadams.model_trainer.entities import AUTOTSAD_ENTITIES

sys.path.append(".")

from download_data import main as download_data
from train_all_models import main as train_all_models
from evaluate_all_models import main as evaluate_all_models
from parse_ranking_results import main as parse_ranking_results


def main():
    DATASETS = ['autotsad']
    # ENTITIES = [AUTOTSAD_ENTITIES]
    ENTITIES = [['GutenTAG=ecg-diff-count-1.semi-supervised']]
    # DATASETS = ['smd']
    # ENTITIES = [['machine-2-8']]

    # download_data(DATASETS)
    train_all_models(DATASETS, ENTITIES)
    evaluate_all_models(DATASETS, ENTITIES)
    # performed within autotsad code:
    # parse_ranking_results(DATASETS, ENTITIES)


if __name__ == "__main__":
    main()
