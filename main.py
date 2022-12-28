from transformers import BertTokenizer
from transformers import AdamW
import torch
from dpt.data import JSONDataset
from torch.utils.data import DataLoader
from x_transformers import TransformerWrapper, Decoder, XTransformer, AutoregressiveWrapper
from tqdm import tqdm, trange
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
GENERATE_LENGTH = 50
GRAD_ACCUMULATE_EVERY = 5

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
max_length = 100
dataset = JSONDataset(json_file='d.jsonl', tokenizer=tokenizer, max_length=max_length)


# create a train and test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
dataloader = DataLoader(train_dataset, batch_size=90, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)


criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


model = AutoregressiveWrapper(
    TransformerWrapper(
        num_tokens = 115000,
        attn_layers=Decoder(dim=256, depth=5, heads=3),
        max_seq_len = max_length,
    ),
    ignore_index=tokenizer.pad_token_id,
    
)


if __name__ == '__main__':

    model.load_state_dict(
        torch.load('model_latest_v2.pth')).to(DEVICE)


    num_epochs = 1000

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    
    
    for epoch in range(num_epochs):
       
        model.train()

        
        epoch_loss = 0.0
        pbar = trange(len(dataloader), desc="Loss")
        iter_dataloader = iter(dataloader)
        # Iterate through the data
        for idx in pbar:
            optimizer.zero_grad()
            batch = next(iter_dataloader)
            # Extract the inputs and labels from the batch
            inputs, labels = batch['question'], batch['responses']

            # move to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)


            # Forward pass
            outputs = model.net(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()

            # Backward pass
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            

            # Accumulate the loss
            epoch_loss += loss.item()
            pbar.set_description(f"loss: {loss.item():.3f}\n LR: {optimizer.param_groups[0]['lr']}")
            # if idx ==50:
            #     break
       
        print(f'Epoch {epoch+1} loss: {epoch_loss / len(dataloader)}')

        
        torch.save(model.state_dict(), f'./model_latest_v2.pth')
        model.eval()

      
        with torch.no_grad():
            
            for idx, batch in enumerate(test_dataloader):
                
                inputs, labels = batch['question'], batch['responses']
       
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

               
                outputs = model.generate(inputs, GENERATE_LENGTH, eos_token=tokenizer.eos_token_id)
                # decode the tokens into text
                decoded_responses = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_questions = tokenizer.decode(inputs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(f'Question: {decoded_questions}: \n\t Answer: {decoded_responses}')
                if idx == 10:
                    break
