import esm
import torch
import itertools
import os
import string
from Bio import SeqIO
import numpy as np
import argparse
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
from scipy.spatial.distance import squareform, pdist, cdist


msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval().cuda()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa 
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)
    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


parser = argparse.ArgumentParser()
parser.add_argument('--a3m_data', type=str, default="../data/example/positive_train1.a3m")
parser.add_argument('--msa_composition_data', type=str, default="../data/example/positive_train_msa_composition1.npy")
args = parser.parse_args()

inputs = greedy_select(read_msa(args.a3m_data),128)
msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
msa_token_data = msa_transformer_batch_tokens
if msa_transformer_batch_tokens.shape[2] > 1024:
    msa_token_data = msa_transformer_batch_tokens[:,:,:1024]
else:
    msa_token_data = torch.cat((msa_transformer_batch_tokens,torch.full((1,msa_transformer_batch_tokens.shape[1],1024-msa_transformer_batch_tokens.shape[2]),1)),2) #add padding token
if msa_transformer_batch_tokens.shape[1]<128:
    msa_token_data = torch.cat((msa_token_data,torch.full((1,128-msa_transformer_batch_tokens.shape[1],1024),1)),1)
    for k in range(msa_transformer_batch_tokens.shape[1],128):
        msa_token_data[0][k][0] = 0      
msa_token_data = msa_token_data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


msa_embedding_data = msa_transformer(msa_token_data, repr_layers=[12],need_head_weights=False ,return_contacts=False)['representations'][12][:,0,:,:]


msa_composition_data = np.zeros([33,768])
data = msa_embedding_data.cpu().numpy()[0]
label_p = msa_token_data.cpu().numpy()[0][0]
for i in range(1024):
    msa_composition_data[label_p[i]] += data_p[i]
pad_num=0
for i in range(3):
    pad_num += list(label_p).count(i)
for i in range(24,33):
    pad_num += list(label_p).count(i) 
for i in range(4,24):
    msa_composition_data[i] = msa_composition_data[i]/(1024-pad_num)
msa_composition_data = msa_composition_data[4:24,:]

np.save(args.msa_composition_data,msa_composition_data)
