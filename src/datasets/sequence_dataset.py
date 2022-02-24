import numpy as np
from torch import from_numpy
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from utils import getData


class SequenceDataset(Dataset):
    def __init__(self,
                 word2id,
                 fam2label,
                 seq_max_len,
                 dataset_path,
                 partition):

        self.word2id = word2id
        self.fam2label = fam2label
        self.seq_max_len = seq_max_len
        self.data, self.label = getData(dataset_path, partition)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.preprocess(self.data.iloc[index])
        label = self.fam2label[self.label.iloc[index]]

        return {'sequence': sequence, 'target': label}

    def preprocess(self, sequence):
        seq = list(map(lambda x: self.word2id[x], sequence[:self.seq_max_len]))
        padding = list(range(self.seq_max_len - len(seq)))
        seq += list(map(lambda _: self.word2id['<pad>'], padding))
        seq = from_numpy(np.array(seq))

        one_hot_seq = one_hot(seq, num_classes=len(self.word2id))
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq


def getDataloaders(word2id, fam2label, params, partitions, shuffles):
    dataloaders = {}

    for partition, shuffle in zip(partitions, shuffles):
        dataset = SequenceDataset(word2id=word2id,
                                  fam2label=fam2label,
                                  seq_max_len=params['seq_max_len'],
                                  dataset_path=params['dataset_path'],
                                  partition=partition)

        dataloaders[partition] = DataLoader(dataset=dataset,
                                            batch_size=params['batch_size'],
                                            shuffle=shuffle,
                                            num_workers=params['num_workers'])

    return dataloaders
