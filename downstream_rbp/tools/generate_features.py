import os, sys, h5py

import numpy as np

from pretrained import load_pretrained_model


MODEL_NAME = "ProtRNA_pretrained"

protname  = sys.argv[1]
BATCH_SIZE = int(sys.argv[2])
data_dir = sys.argv[3]
feature_dir = sys.argv[4]

def decode_one_hot(one_hot_sequences):
    """Decode multiple one-hot encoded DNA/RNA sequences.
    
    Parameters:
        one_hot_sequences (numpy array): A numpy array of shape (N, 4, L) where
            N is the number of sequences,
            4 represents the one-hot encoding for 'A', 'C', 'G', 'T' (or 'U'),
            L is the sequence length.
    
    Returns:
        List of decoded sequences as strings.
    """
    nucleotide_map = {0: 'a', 1: 'c', 2: 'g', 3: 'u'}
    decoded_sequences = []

    for sequence in one_hot_sequences:
        decoded_seq = ""
        transposed_seq = sequence.T

        for position in transposed_seq:
            # Determine which nucleotide is represented
            nucleotide_index = np.argmax(position)
            decoded_seq += nucleotide_map[nucleotide_index]

        decoded_sequences.append(decoded_seq)
    
    return decoded_sequences


# Load pretrained models
base_model = load_pretrained_model(name=MODEL_NAME)
batch_converter = base_model.alphabet.get_batch_converter()

print(f"{protname} data under inference")

filename = protname + ".h5"
print('processing', filename)
file_path = os.path.join(data_dir, filename)
save_path = os.path.join(feature_dir, filename)

datafile = h5py.File(file_path, 'r')
save_datafile = h5py.File(save_path, 'w')

for key in ['Y_train', 'Y_test']:
    seq_labels = np.array(datafile[key]).astype(np.int32)
    save_dataset = save_datafile.create_dataset(key, data=seq_labels, compression="gzip")
    print(f"{key} dataset saved in {filename}")

for key in ['X_train', 'X_test']:
    seq_data = np.array(datafile[key]).astype(np.float32) # N, 5, L
    one_hots = seq_data[:, :4, :]
    seqs = decode_one_hot(one_hots)
    print('dataset length:', len(seqs))
    print('one hot shape:', one_hots.shape)
    print(seqs[0])

    feats = np.empty((seq_data.shape[0], 1280, seq_data.shape[-1])) # N, 1280, L

    for i in range(0, len(seqs), BATCH_SIZE):
        batch_tokens = batch_converter(seqs[i:i + BATCH_SIZE])
        batch_results = base_model(batch_tokens, repr_layers=[33]) # L, 1280
        batch_reprs = batch_results['representations'][33][:, 1:-1, :] # N, L, 1280
        feats[i:i + BATCH_SIZE] = np.transpose(batch_reprs, axes=(0, 2, 1)) # N, 1280, L

        if i == 0:
            print('batch tokens shape:', batch_tokens.shape)
            print(batch_tokens[0])
            print('batch reprs shape:', batch_reprs.shape)
    
    seq_and_feats = np.concatenate([one_hots, feats], axis=1) # N, 4+1280, L
    print('shape of seq_and_feats:', seq_and_feats.shape)
    save_dataset = save_datafile.create_dataset(key, data=seq_and_feats, compression="gzip")
    print(f"{key} dataset saved in {filename}")

save_datafile.close()
print(f"all seqs features saved in {filename}")