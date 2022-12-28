import json
import torch
from torch.utils.data import Dataset

class JSONDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length, transform=None):
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        category = sample['category']
        responses = sample['responses']
        if len(responses) == 0:
            responses = ['']
        # question autoregressive 
        # autoreg_question = ''.join([f'{a} {b}' for a, b in zip([question], responses)])
        if self.transform:
            question = self.transform(question)
            category = self.transform(category)
            responses = self.transform(responses)

        question_tokens = self.tokenizer.encode(question, max_length=self.max_length, padding='max_length', truncation=True)
        responses_tokens = [self.tokenizer.encode(response, max_length=self.max_length, padding='max_length', truncation=True) for response in responses][0]
        
        return {'question': torch.tensor(question_tokens), 'responses': torch.tensor(responses_tokens)}
