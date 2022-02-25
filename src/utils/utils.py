import glob
import os

from pandas import concat, read_csv


def getData(dataset_path, partition):
    files = glob.glob(os.path.join(dataset_path, partition, '*'))
    data = list(map(lambda x: read_csv(x,
                                       index_col=None,
                                       usecols=['sequence',
                                                'family_accession']), files))
    data = concat(data)

    return data['sequence'], data['family_accession']


def getLabels(targets):
    return {target: i for i, target in enumerate(targets.unique())}


def getVocabulary(data):
    vocabulary = set()
    for sequence in data:
        vocabulary.update(sequence)

    word2id = {w: i for i, w in enumerate(sorted(vocabulary), start=1)}
    word2id['<pad>'] = 0

    return word2id
