import json
import torch
import torch.nn as nn
import numpy as np
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AdamW, BertPreTrainedModel, DistilBertModel

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

def preprocess(train_path):
  contexts, questions, answers = read_squad(train_path)

def setup_model():
  model = Model.from_pretrained('distilbert-base-uncased')

  return model

def train(model):
  model.train()
  pass

def validate():
  pass

if __name__ == '__main__':
  args = parse_args()

  train_contexts, train_questions, train_answers = preprocess(args.train_path)
  dev_contexts, dev_questions, dev_answers = preprocess(args.dev_path)

  model = setup_model()
  
  model = train(model)


