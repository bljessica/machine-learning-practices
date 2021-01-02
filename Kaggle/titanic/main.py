import pandas as pd
import torch
import model
import processData

if __name__ == "__main__":
    dp = processData.DataProcessing()
    train_x, train_y = dp.load_data('data/train.csv')
    test_x = dp.load_data('data/test.csv')

    model = model.MyNet()
    model.train(train_x, train_y, 100)
    model.write_result(test_x)


