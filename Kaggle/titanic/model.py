import torch.nn as nn
import torch
import csv
import pandas as pd
import numpy as np

class MyNet(nn.Module):
  def __init__(self):
    super(MyNet, self).__init__()
    self.fc = nn.Sequential(nn.Linear(8, 6), nn.Sigmoid(), nn.Linear(6, 4), nn.Sigmoid(), nn.Linear(4, 1))
    self.opt = torch.optim.Adam(params=self.parameters(), lr=0.01)
    self.mls = nn.MSELoss()

  def forward(self, inputs):
    return self.fc(inputs)
  
  def train(self, inputs, y, epochs):
    inputs = inputs.astype(np.float32)
    inputs = torch.from_numpy(inputs)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)
    for epoch in range(epochs):
      for i in range(len(inputs)):
        out = self.forward(inputs[i])
        loss = self.mls(out, y[i])
        self.zero_grad()
        loss.backward()
        self.opt.step()
        if loss.item() < 1e-4:
          break
      print('Epoch: {}, loss: {}'.format(epoch + 1, loss.item()))
  
  def test(self, inputs):
    inputs = inputs.astype(np.float32)
    inputs = torch.from_numpy(inputs)
    return [self.forward(input) for input in inputs]
    

  def write_result(self, inputs):
    ans_y = self.test(inputs)
    with open('data/submit.csv', mode='w', newline='') as submit_file:
      csv_writer = csv.writer(submit_file)
      header = ['PassengerId', 'Survived']
      csv_writer.writerow(header)
      for i in range(len(ans_y)):
        row = [i + 892, 0 if ans_y[i].item() <= 0.5 else 1]
        csv_writer.writerow(row)
    print('write done')
