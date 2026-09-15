from torch.utils.data import Dataset, DataLoader
from Tokenizer import BPETokenizer
import torch
import pickle
import os

def _get_cache_path(max_length, stride, cache_dir=".cache"):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"gpt_dataset_cache_ml{max_length}_s{stride}.pkl")

def _load_from_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def _save_to_cache(cache_path, inputs_ids, targets_ids):
    with open(cache_path, 'wb') as f:
        pickle.dump({'inputs_ids': inputs_ids, 'targets_ids': targets_ids}, f)

class GPTDatasetV1(Dataset):
    def __init__(self, ds, tokenizer, max_length, stride, cache_dir=".cache"):
        cache_path = _get_cache_path(max_length, stride, cache_dir)
        cached_data = _load_from_cache(cache_path)
        
        if cached_data:
            self.inputs_ids = cached_data['inputs_ids']
            self.targets_ids = cached_data['targets_ids']
        else:
            self.inputs_ids = []
            self.targets_ids = []
            
            all_text = "\n".join(ds)
            token_ids = tokenizer.encode(all_text)

            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1:i + max_length + 1]

                self.inputs_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.targets_ids.append(torch.tensor(target_chunk, dtype=torch.long))
            
            _save_to_cache(cache_path, self.inputs_ids, self.targets_ids)

    def __len__(self):
        return len(self.inputs_ids)

    def __getitem__(self, idx):
        return self.inputs_ids[idx], self.targets_ids[idx]

def create_dataloader_v1(ds, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0, cache_dir=".cache"):
    tokenizer = BPETokenizer()
    dataset = GPTDatasetV1(ds, tokenizer, max_length, stride, cache_dir=cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
