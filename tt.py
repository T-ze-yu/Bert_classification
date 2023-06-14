from preprocess import DataProcessor
from config import Config
from transformers import BertTokenizer
config = Config('datas')
tokenizer = BertTokenizer.from_pretrained('pretrained_bert')

dev_iterator = DataProcessor(config, tokenizer, 334, mode='dev')