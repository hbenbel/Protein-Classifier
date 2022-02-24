import argparse
from os.path import exists

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import getDataloaders
from models import ProtCNN
from utils import getData, getLabels, getVocabulary


def main(params):
    seed_everything(0)

    # Retrieve labels and ids
    train_data, train_targets = getData(dataset_path=params['dataset_path'],
                                        partition='train')

    fam2label = getLabels(targets=train_targets)
    word2id = getVocabulary(data=train_data)

    # Retrieve dataloaders
    dataloaders = getDataloaders(word2id=word2id,
                                 fam2label=fam2label,
                                 params=params,
                                 partitions=['train', 'dev', 'test'],
                                 shuffles=[True, False, False])

    # Initialize model
    model = ProtCNN(num_id=len(word2id), num_classes=len(fam2label))
    logger = TensorBoardLogger(save_dir=params['log_path'])
    trainer = Trainer(devices='auto',
                      accelerator=params['accelerator'],
                      max_epochs=params['epochs'],
                      logger=logger)

    # Launch model training (and testing)
    trainer.fit(model=model,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['dev'])

    if params['test'] is True:
        trainer.test(test_dataloaders=dataloaders['test'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Training of ProtCNN'
             )

    parser.add_argument(
        '--dataset_path',
        '-d',
        type=str,
        help='Path to the splitted dataset',
        required=True
    )

    parser.add_argument(
        '--log_path',
        '-l',
        type=str,
        help='Path to the output log directory',
        required=True
    )

    parser.add_argument(
        '--seq_max_len',
        '-s',
        type=int,
        help='Maximum length of sequences to consider for training',
        required=False,
        default=120
    )

    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        help='Batch size for training',
        required=False,
        default=512
    )

    parser.add_argument(
        '--num_workers',
        '-n',
        type=int,
        help="Number of workers to use for the dataloaders",
        required=False,
        default=8
    )

    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        help='Number of epochs for the training',
        required=False,
        default=25
    )

    parser.add_argument(
        '--accelerator',
        '-a',
        type=str,
        help='Type of device to use for the training',
        required=False,
        default='cpu',
        choices=['cpu', 'gpu', 'tpu']
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Flag to allow the testing of the model',
        required=False,
        default=False
    )

    args = parser.parse_args()
    params = {}
    params['dataset_path'] = args.dataset_path
    params['log_path'] = args.log_path
    params['seq_max_len'] = args.seq_max_len
    params['batch_size'] = args.batch_size
    params['num_workers'] = args.num_workers
    params['accelerator'] = args.accelerator
    params['epochs'] = args.epochs
    params['test'] = args.test

    assert exists(params['dataset_path']), "Dataset path doesn't exists :("

    main(params)
