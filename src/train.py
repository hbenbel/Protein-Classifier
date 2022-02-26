import argparse
from os.path import exists

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import getDataloaders
from models import ProtCNN
from utils import analyzeDataset, getData, getLabels, getVocabulary


def main(params):
    seed_everything(0)

    # Retrieve labels and ids
    train_data, train_targets = getData(dataset_path=params['dataset_path'],
                                        partition='train')

    # Launch training data analysis
    if params['analyze'] is True:
        analyzeDataset(dataset_path=params['dataset_path'],
                       log_path=params['log_path'])

    fam2label = getLabels(targets=train_targets)
    word2id = getVocabulary(data=train_data)

    # Retrieve dataloaders
    dataloaders = getDataloaders(word2id=word2id,
                                 fam2label=fam2label,
                                 params=params,
                                 partitions=['train', 'dev', 'test'],
                                 shuffles=[True, False, False])

    # Initialize model
    model = ProtCNN(num_id=len(word2id),
                    num_classes=len(fam2label),
                    params=params)

    logger = TensorBoardLogger(save_dir=params['log_path'])
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = Trainer(devices='auto',
                      accelerator=params['accelerator'],
                      max_epochs=params['epochs'],
                      logger=logger,
                      callbacks=[checkpoint_callback])

    # Launch model training (and testing)
    if params['train'] is True:
        trainer.fit(model=model,
                    train_dataloaders=dataloaders['train'],
                    val_dataloaders=dataloaders['dev'])

    if params['test'] is True:
        trainer.test(model=model if params['train'] is False else None,
                     ckpt_path=params['ckpt_path'],
                     dataloaders=dataloaders['test'])


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
        '--train',
        action='store_true',
        help='Flag to allow the training of the model',
        required=False,
        default=False
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Flag to allow the testing of the model',
        required=False,
        default=False
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Flag to allow the analysis of dataset',
        required=False,
        default=False
    )

    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        help='Learning rate for the optimizer',
        required=False,
        default=1e-2
    )

    parser.add_argument(
        '--momentum',
        '-m',
        type=float,
        help='Momentum for the optimizer',
        required=False,
        default=0.9
    )

    parser.add_argument(
        '--weight_decay',
        '-w',
        type=float,
        help='Weight decay for the optimizer',
        required=False,
        default=1e-2
    )

    parser.add_argument(
        '--gamma',
        '-g',
        type=float,
        help='Multiplicative factor of learning rate decay',
        required=False,
        default=0.9
    )

    args = parser.parse_args()

    parser.add_argument(
        '--ckpt_path',
        '-c',
        type=str,
        help='Path toward the pretrained model to use for testing only',
        required=args.test is True and args.train is False,
        default=None
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
    params['train'] = args.train
    params['test'] = args.test
    params['analyze'] = args.analyze
    params['ckpt_path'] = args.ckpt_path
    params['learning_rate'] = args.learning_rate
    params['momentum'] = args.momentum
    params['weight_decay'] = args.weight_decay
    params['gamma'] = args.gamma

    assert exists(params['dataset_path']), "Dataset path doesn't exists :("

    main(params)
