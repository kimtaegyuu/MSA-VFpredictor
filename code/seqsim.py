import numpy as np
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help="input query name", required=True)
    args = parser.parse_args()
    
    
    seqsim=[]
    with open('blast_db/'+args.query+'_positive') as f:
        pos = f.readlines()
    pos = pos[15:]
    for i in range(len(pos)):
        if 'Score     E' in pos[i]:
            seqsim.append(float(pos[i+3].split()[1]))

    with open('blast_db/'+args.query+'_negative') as fn:
        neg = fn.readlines()
    neg = neg[15:]
    for i in range(len(neg)):
        if 'Score     E' in neg[i]:
            seqsim.append(float(neg[i+3].split()[1]))
            
    np.save(args.query+'_seqsim',np.array(seqsim))