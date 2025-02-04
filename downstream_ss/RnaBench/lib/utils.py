import collections

import numpy as np
import pandas as pd

from Bio import AlignIO
from pathlib import Path
from collections import defaultdict

canonicals = ['AU', 'UA', 'GC', 'CG', 'UG', 'GU']

################################################################################
# Checking functions
################################################################################

def quick_check_df(row):
    assert len(row['sequence']) == len(row['structure'])
    assert len(row['pk']) == len(row['pos1id']) == len(row['pos2id'])

# is tested
def is_valid(structure):
    assert len([int(x.isupper()) for x in structure]) == len([int(x.islower()) for x in structure])
    assert structure.count('(') == structure.count(')')
    assert structure.count('[') == structure.count(']')
    assert structure.count('{') == structure.count('}')
    assert structure.count('<') == structure.count('>')

def not_seq_nc_only(row):
    seq = ''.join(row['sequence']).upper()
    if all([x not in ['A', 'G', 'C', 'U'] for x in set(seq)]):
        return False
    return True

def has_seq_nc(row):
    seq = ''.join(row['sequence']).upper()
    if any([x not in ['A', 'G', 'C', 'U'] for x in set(seq)]):
        return True
    return False

def has_nc(row):
    for p1, p2 in zip(row['pos1id'], row['pos2id']):
        if row['sequence'][p1]+row['sequence'][p2] not in canonicals:
            return True
    return False

def has_multiplet(row):
    pos_count = collections.Counter(row['pos1id'] + row['pos2id'])
    is_multi = any(i > 1 for i in pos_count.values())
    return is_multi

################################################################################
# Representation Conversion
################################################################################

def db2pairs(structure, start_index=0):
    """
    Converts dot-bracket string into a list of pairs.

    Input:
      structure <string>: A sequence in dot-bracket format.
      start_index <int>: Starting index of first nucleotide (default zero-indexing).

    Returns:
      pairs <list>: A list of tuples of (index1, index2, pk_level).

    """
    level_stacks = collections.defaultdict(list)
    closing_partners = {')': '(', ']': '[', '}': '{', '>': '<'}
    levels = {')': 0, ']': 1, '}': 2, '>': 3}

    pairs = []

    for i, sym in enumerate(structure, start_index):
        if sym == '.':
            continue
        # high order pks are alphabetical characters
        if sym.isalpha():
            if sym.isupper():
                level_stacks[sym].append(i)
            else:
                try:  # in case we have invalid preditions, we continue with next bracket
                    op = level_stacks[sym.upper()].pop()
                    pairs.append((op, i,
                                  ord(sym.upper()) - 61))  # use asci code if letter is used to asign PKs, start with level 4 (A has asci code 65)
                except:
                    continue
        else:
            if sym in closing_partners.values():
                level_stacks[sym].append(i)
            else:
                try:  # in case we have invalid preditions, we continue with next bracket
                    op = level_stacks[closing_partners[sym]].pop()
                    pairs.append([op, i, levels[sym]])
                except:
                    continue
    return sorted(pairs, key=lambda x: x[0])


def pairs2mat(pairs, length, symmetric=True, no_pk=False):
    """
    Convert list of pairs to matrix representation of structure.
    """
    # print(pairs)
    mat = np.zeros((length, length))
    if no_pk:
        for p1, p2 in pairs:
            mat[p1, p2] = 1
            if symmetric:
                mat[p2, p1] = 1
        return mat
    for p1, p2, _ in pairs:
        mat[p1, p2] = 1
        if symmetric:
            mat[p2, p1] = 1
    return mat


def db2mat(db):
    """
    Convert dot-bracket string to matrix representation of structure.
    """
    length = len(db)
    pairs = db2pairs(db)
    return pairs2mat(pairs, length)



################################################################################
# Format Conversion
################################################################################

def info_from_ct_no_gt(ct_path):
    ct = pd.read_csv(ct_path, sep='\s+', header=None, names=['1', '2', '3', '4', '5', '6'])
    seq = ct['2'].to_list()[1:]
    pairs = [[int(p1), int(p2)-1] for p1, p2 in zip(ct['3'].to_list()[1:], ct['5'].to_list()[1:]) if int(p2) != 0]

    return ''.join(seq), pairs


def pairs_from_ct_with_gt(ct_dir, ground_truth):
    pairs = []
    for i, row in ground_truth.iterrows():
        try:
            ct_path = Path(ct_dir, f"{row['Id']}.ct")
            ct = pd.read_csv(ct_path, sep='\s+', header=None, names=['1', '2', '3', '4', '5', '6'])
            pred_seq = ct['2'].to_list()[1:]
            assert ''.join(pred_seq) == ''.join(row['sequence'])
            pred_pairs = [[int(p1), int(p2)-1] for p1, p2 in zip(ct['3'].to_list()[1:], ct['5'].to_list()[1:]) if int(p2) != 0]
            pairs.append(pred_pairs)
        except AssertionError as a:
            print('### Sequence in ct file does not match original sequence.')
            print('original:', row['sequence'])
            print('ct -file:', pred_seq)
            print("### I'll use None for the prediciton.")
            pairs.append(None)
        except FileNotFoundError as f:
            print('### No file for id', row.index)
            print(f)
            print("### I'll use None for the prediciton.")
    return pairs




def fasta2df(fa_file):
    pass

def df2fasta(df, out_path):
    """
    convert dataframe to fasta file.
    """
    with open(out_path, 'w+') as f:
        for _, row in df.iterrows():
            f.write('>' + str(row['Id']) + '\n')
            f.write(''.join(row['sequence'])+'\n')


def stockholm2records(sto_file):
    align = AlignIO.read(sto_file, "stockholm")
    return align


def stockholm2df(sto_file):
    align = stockholm2records(sto_file)
    d_list = [{'Id': r.name, 'sequence': r.seq} for r in align]
    return pd.DataFrame(d_list)

def stockholm2idlist(sto_file):
    align = stockholm2records(sto_file)
    return [r.name for r in align]


def df2bpseq(df, outdir, acgu_only=False, seed=0):

    rng = np.random.default_rng(seed=seed)

    NUCS = {
        'T': 'U',
        'P': 'U',
        'R': 'A',  # or 'G'
        'Y': 'C',  # or 'T'
        'M': 'C',  # or 'A'
        'K': 'U',  # or 'G'
        'S': 'C',  # or 'G'
        'W': 'U',  # or 'A'
        'H': 'C',  # or 'A' or 'U'
        'B': 'U',  # or 'G' or 'C'
        'V': 'C',  # or 'G' or 'A'
        'D': 'A',  # or 'G' or 'U'
        'N': rng.choice(['A', 'C', 'G', 'U']),
        'A': 'A',
        'U': 'U',
        'C': 'C',
        'G': 'G',
    }

    def nuc_mapping(s, nucs=NUCS):
            return NUCS[s]


    for i, row in df.iterrows():
        bpseq_output = [[str(i), s, '0']
                        if not acgu_only
                        else [str(i), nuc_mapping(s), '0']
                        for i, s in enumerate(row['sequence'], 1)]
        outputs = []
        for p1, p2 in zip(row['pos1id'], row['pos2id']):
            bpseq_output[p1][2] = str(p2+1)
            bpseq_output[p2][2] = str(p1+1)

        outpath = Path(outdir, f"{row['Id']}.bpseq")
        with open(outpath, 'w+') as f:
            for line in bpseq_output:
                f.write('\t'.join(line)+'\n')


################################################################################
# STATISTICS
################################################################################

def get_pair_types(seq, pairs):
    wc_pairs = ['AU', 'UA', 'GC', 'CG']
    wobble_pairs = ['GU', 'UG']
    wc = []
    wobble = []
    nc = []
    for pair in pairs:
        bp = seq[pair[0]]+seq[pair[1]]
        if bp in wc_pairs:
            wc.append((pair[0], pair[1]))
        elif bp in wobble_pairs:
            wobble.append((pair[0], pair[1]))
        else:
            nc.append((pair[0], pair[1]))
    return wc, wobble, nc


################################################################################
# following code snippets originate from SPOT-RNA repo at https://github.com/jaswindersingh2/SPOT-RNA
# we use them for plotting with varna in the visualization module
################################################################################


# copy-paste from SPOT-RNA2 source code
def lone_pair(pairs):
    lone_pairs = []
    pairs.sort()
    for i, I in enumerate(pairs):
        if ([I[0] - 1, I[1] + 1] not in pairs) and ([I[0] + 1, I[1] - 1] not in pairs):
            lone_pairs.append(I)

    return lone_pairs


# copy-paste from SPOT-RNA2 source code
def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]],sequence[i[1]]] in [["A","U"], ["U","A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]],sequence[i[1]]] in [["G","C"], ["C","G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]],sequence[i[1]]] in [["G","U"], ["U","G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
        # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t



# copy-paste from SPOT-RNA2 source code
def multiplets_pairs(pred_pairs):

    pred_pair = [i[:2] for i in pred_pairs]
    temp_list = flatten(pred_pair)
    temp_list.sort()
    new_list = sorted(set(temp_list))
    dup_list = []
    for i in range(len(new_list)):
        if (temp_list.count(new_list[i]) > 1):
            dup_list.append(new_list[i])

    dub_pairs = []
    for e in pred_pair:
        if e[0] in dup_list:
            dub_pairs.append(e)
        elif e[1] in dup_list:
            dub_pairs.append(e)

    temp3 = []
    for i in dup_list:
        temp4 = []
        for k in dub_pairs:
            if i in k:
                temp4.append(k)
        temp3.append(temp4)

    return temp3


# copy-paste from SPOT-RNA2 source code
def multiplets_free_bp(pred_pairs, y_pred):
    L = len(pred_pairs)
    multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = []
    while len(multiplets_bp) > 0:
        remove_pairs = []
        for i in multiplets_bp:
            save_prob = []
            for j in i:
                save_prob.append(y_pred[j[0], j[1]])
            remove_pairs.append(i[save_prob.index(min(save_prob))])
            save_multiplets.append(i[save_prob.index(min(save_prob))])
        pred_pairs = [k for k in pred_pairs if k not in remove_pairs]
        multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = [list(x) for x in set(tuple(x) for x in save_multiplets)]
    assert L == len(pred_pairs)+len(save_multiplets)
    #print(L, len(pred_pairs), save_multiplets)
    return pred_pairs, save_multiplets


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def ct_file_output(pairs, seq, id, save_result_path, algorithm='folding_algorithm'):

    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]] = int(I[1]) + 1
        col5[I[1]] = int(I[0]) + 1
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    #os.chdir(save_result_path)
    #print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(save_result_path, (temp), delimiter='\t\t', fmt="%s", header=str(len(seq)) + '\t\t' + str(id) + '\t\t' + algorithm + ' output\n' , comments='')  # changed outpath from original SPOT-RNA code

    return


def get_multiplets(pairs):
    pair_dict = defaultdict(list)
    for p in pairs:
        pair_dict[p[0]].append(p[1])
        pair_dict[p[1]].append(p[0])

    multiplets = []

    for k, v in pair_dict.items():
        if len(v) > 1:
            for p in v:
                multiplets.append(tuple(sorted([k, p])))
    multiplets = list(set(multiplets))

    return multiplets
