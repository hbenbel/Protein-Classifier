import glob
import os
from collections import Counter

from matplotlib.pyplot import savefig, subplots
from pandas import DataFrame, concat, read_csv
from seaborn import barplot, histplot
from tqdm import tqdm


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


def get_amino_acid_frequencies(data, partition):
    aa_counter = Counter()

    for sequence in data:
        aa_counter.update(sequence)

    return DataFrame({f'Amino Acid {partition}': list(aa_counter.keys()),
                     'Frequency': list(aa_counter.values())})


def family_sizes_distribution(dataset_path, saving_folder_path):
    f, ax = subplots(1, 3, figsize=(15, 5))
    for i, partition in tqdm(enumerate(['train', 'dev', 'test']),
                             desc='Generating Family Sizes Distribution'):

        _, targets = getData(dataset_path, partition)
        sorted_targets = targets.groupby(targets).size()
        sorted_targets = sorted_targets.sort_values(ascending=False)
        ax[i].set_title(partition)
        histplot(sorted_targets.values, kde=True, log_scale=True, ax=ax[i])

    f.text(0.5, 0.04, 'Family size (log scale)', ha='center', va='center')
    f.text(0.06,
           0.5,
           '# Families',
           ha='center',
           va='center',
           rotation='vertical')

    savefig(fname=os.path.join(saving_folder_path,
                               'family_sizes_distribution.png'))


def sequence_length_distribution(dataset_path, saving_folder_path):
    f, ax = subplots(1, 3, figsize=(15, 5))
    for i, partition in tqdm(enumerate(['train', 'dev', 'test']),
                             desc='Generating Sequence Length Distribution'):

        data, _ = getData(dataset_path, partition)
        sequence_lengths = data.str.len()
        median = sequence_lengths.median()
        mean = sequence_lengths.mean()

        histplot(sequence_lengths.values,
                 kde=True,
                 log_scale=True,
                 bins=60,
                 ax=ax[i])

        ax[i].set_title(partition)
        ax[i].axvline(mean,
                      color='r',
                      linestyle='-',
                      label=f"Mean = {mean:.1f}")

        ax[i].axvline(median,
                      color='g',
                      linestyle='-',
                      label=f"Median = {median:.1f}")

        ax[i].legend(loc='best')

    f.text(0.5, 0.04, 'Sequence length (log scale)', ha='center', va='center')
    f.text(0.06,
           0.5,
           '# Sequences',
           ha='center',
           va='center',
           rotation='vertical')

    savefig(fname=os.path.join(saving_folder_path,
                               'sequence_length_distribution.png'))


def amino_acid_distribution(dataset_path, saving_folder_path):
    f, ax = subplots(1, 3, figsize=(15, 5))
    for i, partition in tqdm(enumerate(['train', 'dev', 'test']),
                             desc='Generating Amino Acid Distribution'):

        data, _ = getData(dataset_path, partition)
        ax[i].set_yscale('log')
        amino_acid_counter = get_amino_acid_frequencies(data, partition)
        barplot(x=f'Amino Acid {partition}',
                y='Frequency',
                data=amino_acid_counter.sort_values(
                                            by=[f'Amino Acid {partition}'],
                                            ascending=True),
                ax=ax[i])

    savefig(fname=os.path.join(saving_folder_path,
                               'amino_acid_distribution.png'))


def analyzeDataset(dataset_path, log_path):
    saving_folder_path = os.path.join(log_path, 'analysis')

    if not os.path.exists(saving_folder_path):
        os.makedirs(saving_folder_path)

    family_sizes_distribution(dataset_path, saving_folder_path)
    sequence_length_distribution(dataset_path, saving_folder_path)
    amino_acid_distribution(dataset_path, saving_folder_path)
