import torch
from transformers import BertTokenizer
from aiogram import Bot, Dispatcher, executor, types
from main import model

API_TOKEN = ''
PATH = 'model_latest_v2.pth'
# DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

def load_model(path):

    model.load_state_dict(
        torch.load(path, map_location=DEVICE)
    )

    return model.to(DEVICE)

def generate_answer(question):
    model.eval()

    question = torch.tensor(tokenizer.encode(question, add_special_tokens=True)).unsqueeze(0).to(DEVICE)
    response = model.generate(question, 32)
    decoded_response = tokenizer.decode(response[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return decoded_response 

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = load_model(PATH)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    text = ["Привет!",
            "Я стараюсь отвечать на вопросы!"]
    await message.reply("\n".join(text))
    
@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    text = ["Для получения ответа от бота введите любой текст и отправьте его."]
    await message.reply("\n".join(text))
    
@dp.message_handler()
async def answer(message: types.Message):
    answer_message = ""
    answer_message = generate_answer(message['text'])
        
    await message.answer(answer_message)
    
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)