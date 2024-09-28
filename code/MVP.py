from MVP_classifier import MVP_classifier
from torch import nn
import torch as th
import torch.nn.functional as F
import numpy as np
import torch



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--msa_composition', type=str, help="msa_composition input '.npy' file", required=True)
    parser.add_argument('--seqsim', type=str, help="seqsim input '.npy' file", required=True)
    
    
    MVP_model = MVP_classifier().train().cuda()
    MVP_model.load_state_dict(torch.load('MVP_model.pt'))
    MVP_model.eval().cuda() 

    msa_composition = np.load(arg.msa_composition)
    seqsim = np.load(arg.msa_composition)

    msa_composition = torch.from_numpy(msa_composition)
    msa_composition = msa_composition.type('torch.FloatTensor')
    msa_composition = msa_composition.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    seqsim = torch.from_numpy(seqsim)
    seqsim = seqsim.type('torch.FloatTensor')
    seqsim = seqsim.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    result=MVP_model(msa_composition,seqsim)
    
    if result[0][0].round() ==0:
        print('VF')
    else:
        print('non-VF')