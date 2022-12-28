import torch
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from transformers import BertTokenizer
from main import model
PATH = 'model_latest_pussy.pth'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    max_length = 64
    # so fucking hard right?
    model.load_state_dict(
        torch.load(PATH)
    )
    model = model.to(DEVICE)
    # torch.load(PATH)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    text = "Что можно подарить маме на новый год?"
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to(DEVICE)
    output = model.generate(input_ids, 200)
    # decode
    decoded_responses = tokenizer.decode(output[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(decoded_responses)
    
