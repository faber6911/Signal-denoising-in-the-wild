import torch
import kaldi_io
import kaldiio
import numpy as np
import os

class DenoiseDataset(torch.utils.data.Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp. It returns noisy audio
    with their corresponding clean audio file.
    """
    def __init__(self, scp_file, min_length, transform = None):
        """
        Preprocess Kaldi feats.scp here
        @params:
        scp_file (path): Path to the scp file containing clean audio paths associated with
        corresponding noisy audios.
        min_length (float): Length in seconds for the desired audio files.
        """
        self.noisy_audios, self.clean_audios = [], []
        
        for line in open(scp_file):
            clean_audio_path, noisy_audio_path = line.rstrip().split() 
            self.noisy_audios.extend([noisy_audio_path])
            self.clean_audios.extend([clean_audio_path])
        
        self.noisy_audios = np.array(self.noisy_audios)
        self.clean_audios  = np.array(self.clean_audios)
        self.seq_len = min_length*16000
        self.transform = transform
        print("Totally "+str(len(self.noisy_audios))+" samples")
    
    def __len__(self):
        """Return number of samples 
        """
        return len(self.clean_audios)

    def update(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop 
        once per training iteration. 
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        noisy_audio  = self.noisy_audios[index]
        full_mat = kaldi_io.read_mat(noisy_audio).squeeze(1)
        assert len(full_mat) >= self.seq_len
        pin = np.random.randint(0, len(full_mat) - self.seq_len + 1)
        chunk_mat = full_mat[pin:pin+self.seq_len]
        y, _ = librosa.load(self.clean_audios[index], sr = 16000)
        y = y[pin:pin+self.seq_len]
        
        if self.transform:
            chunk_mat = self.transform(chunk_mat)
            y = self.transform(y)
        
        return chunk_mat.reshape(1, -1), y.reshape(1, -1)
    
    
class AudioDataset(torch.utils.data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):

        if data_type == 'train':
            data_path = '../data/serialized_train_data/'
        else:
            data_path = '../data/serialized_test_data/'
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))

        self.data_type = data_type
        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

#     def reference_batch(self, batch_size):
#         """
#         Randomly selects a reference batch from dataset.
#         Reference batch is used for calculating statistics for virtual batch normalization operation.
#         Args:
#             batch_size(int): batch size
#         Returns:
#             ref_batch: reference batch
#         """
#         ref_file_names = np.random.choice(self.file_names, batch_size)
#         ref_batch = np.stack([np.load(f) for f in ref_file_names])

#         ref_batch = emphasis(ref_batch, emph_coeff=0.95)
#         return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx])
        #pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        clean = pair[0].reshape(1, -1)
        noisy = pair[1].reshape(1, -1)
        return(torch.from_numpy(clean).type(torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor))
#         if self.data_type == 'train':
#             clean = pair[0].reshape(1, -1)
#             return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(
#                 torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
#         else:
#             return os.path.basename(self.file_names[idx]), torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)

    
class SequenceDataset(torch.utils.data.Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp
    """
    def __init__(self, scp_file, utt2spkid_file, min_length):
        """Preprocess Kaldi feats.scp here and balance the training set
        """
        self.rxfiles, self.labels, self.utt2spkid = [], [], {}
        
        # balanced training 
        id_count = {}
        for line in open(utt2spkid_file):
            utt, label = line.rstrip().split()
            self.utt2spkid[utt] = int(label)
            if not int(label) in id_count:
                id_count[int(label)] = 0
            id_count[int(label)] += 1
        max_id_count = int((max(id_count.values())+1)/2)
        
        for line in open(scp_file):
            utt, rxfile = line.rstrip().split()
            label = self.utt2spkid[utt]
            repetition = max(1, max_id_count // id_count[label])
            self.rxfiles.extend([rxfile] * repetition)
            self.labels.extend([label] * repetition)
        
        self.rxfiles = np.array(self.rxfiles)
        self.labels  = np.array(self.labels, dtype=np.int)
        self.seq_len = min_length
        print("Totally "+str(len(self.rxfiles))+" samples with at most "+
            str(max_id_count)+" samples for one class")
    
    def __len__(self):
        """Return number of samples 
        """
        return len(self.labels)

    def update(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop 
        once per training iteration. 
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        rxfile  = self.rxfiles[index]
        full_mat_clean, full_mat_noisy = kaldiio.load_mat(rxfile)
        #assert len(full_mat) >= self.seq_len
        pin = np.random.randint(0, len(full_mat_clean) - self.seq_len + 1)
        chunk_mat_clean = full_mat_clean[pin:pin+self.seq_len]
        chunk_mat_noisy = full_mat_noisy[pin:pin+self.seq_len]
        y = np.array(self.labels[index])
        chunk_mat_clean = chunk_mat_clean.reshape(1, -1)
        chunk_mat_noisy = chunk_mat_noisy.reshape(1, -1)
        
        return torch.from_numpy(chunk_mat_clean).type(torch.FloatTensor),torch.from_numpy(chunk_mat_noisy).type(torch.FloatTensor), torch.from_numpy(y)