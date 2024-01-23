import argparse
import sys
from tsadams.model_trainer.entities import AUTOTSAD_ENTITIES
from tsadams.utils.config import Config

sys.path.append(".")

from download_data import main as download_data
from train_all_models import main as train_all_models
from evaluate_all_models import main as evaluate_all_models
from parse_ranking_results import main as parse_ranking_results



def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("tsadams AutoTSAD integration",
                                     description="Execute tsadams on a single time series (entity) "
                                                 "selected by AutoTSAD.")
    parser.add_argument("--entity", type=str, required=True,
                        help="Name of the entity to process. Required!")
    parser.add_argument("--dataset", type=str, default="autotsad",
                        help="Name of the dataset (time series group). "
                             "Should be the default!")
    parser.add_argument("--config_file_path", "-c", type=str, default="../../configs/config.yml", required=False,
                        help="Path to the tsadams config file")
    return parser


def main(args: argparse.Namespace) -> None:
    datasets = [args.dataset]
    entities = [[args.entity]]

    # load config
    config = Config(config_file_path=args.config_file_path).parse()

    # download_data(DATASETS, args=config)
    train_all_models(datasets, entities, args=config)
    evaluate_all_models(datasets, entities, args=config)
    parse_ranking_results(datasets, entities, args=config)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
