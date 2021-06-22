import json
import torch
import torch.nn as nn
import numpy as np
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AdamW, BertPreTrainedModel, DistilBertModel, DistilBertTokenizerFast

def parse_args():
  parser = argparse.ArgumentParser(description='model.py')
  parser.add_argument('--train_path', type=str, default='data/train-v2.0.json',
                      help='path to train set (you should not need to modify)')
  parser.add_argument('--dev_path', type=str, default='data/dev-v2.0.json',
                      help='path to dev set (you should not need to modify)')
  # parser.add_argument('--blind_test_path', type=str, default='data/test/test.nolabels.txt',
  #                     help='path to blind test set (you should not need to modify)')
  # parser.add_argument('--test_output_path', type=str,
  #                     default='test-blind.output.txt', help='output path for test predictions')
  # Some common args have been pre-populated for you. Again, you can add more during development, but your code needs
  # to run with the default neural_sentiment_classifier for submission.
  parser.add_argument('--lr', type=float, default=3e-4,
                      help='learning rate')
  parser.add_argument('--epochs', type=int, default=10,
                      help='number of epochs to train for')
  parser.add_argument('--hidden_size', type=int,
                      default=500, help='hidden layer size')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='training batch size; 1 by default and you do not need to batch unless you want to')
  parser.add_argument("--load-path", type=str,
                      help=("Path to load a saved model from and "
                            "evaluate on test data. May not be "
                            "used with --save-dir."))
  parser.add_argument("--save-dir", action="store_true",
                      help=("Path to save model checkpoints and logs. "
                            "Required if not using --load-path. "
                            "May not be used with --load-path."))
  parser.add_argument("--output-path", type=str, default="output/output.txt",
                      help=("Path to set output file"))
  args = parser.parse_args()

  return args

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

class Model(BertPreTrainedModel):
  def __init__(self, config):
    super(Model, self).__init__(config)
    self.bert = DistilBertModel(config)
    self.dropout = nn.Dropout()
    self.classifier = nn.Linear()

def read_squad(data_path):
  path = Path(data_path)
  with open(path, 'rb') as f:
    squad_data = json.load(f)

  contexts = []
  questions = []
  answers = []
  for group in squad_data['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          questions.append(question)
          answers.append(answer)

  return contexts, questions, answers

def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two – fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

def preprocess(train_path, dev_path, tokenizer):
  train_contexts, train_questions, train_answers = read_squad(train_path)
  dev_contexts, dev_questions, dev_answers = read_squad(dev_path)

  add_end_idx(train_answers, train_contexts)
  add_end_idx(dev_answers, dev_contexts)

  train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
  dev_encodings = tokenizer(dev_contexts, dev_questions, truncation=True, padding=True)

  add_token_positions(train_encodings, train_answers)
  add_token_positions(dev_encodings, dev_answers)

  return train_encodings, dev_encodings

def setup_model():
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
  model = Model.from_pretrained('distilbert-base-uncased')
  
  return tokenizer, model

def train(model):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)

  model.train()
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  pass

def validate():
  pass

if __name__ == '__main__':
  args = parse_args()

  tokenizer, model = setup_model()

  train_encodings, dev_encodings = preprocess(args.train_path, args.dev_path, tokenizer)
  train_dataset = Dataset(train_encodings)
  dev_dataset = Dataset(dev_encodings)
  
  model = train(model)