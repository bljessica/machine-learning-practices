import pandas as pd


class DataProcessing():
  def __init__(self):
    pass

  def load_data(self, file_path):
    data = pd.read_csv(file_path)
    # 查看数据缺失情况
    # data.info()
    data = data.iloc[:, 1:]

    # 补足缺失数据
    # Cabi (数据项缺失最严重)
    data.loc[(data.Cabin.notnull()), 'Cabin'] = 1
    data.loc[(data.Cabin.isnull()), 'Cabin'] = 0
    # 年龄
    # 按称谓计算不同年龄层的均值
    Mr_age_mean = (data[data.Name.str.contains('Mr.')]['Age'].mean())
    Mrs_age_mean = (data[data.Name.str.contains('Mrs.')]['Age'].mean())
    Miss_age_mean = (data[data.Name.str.contains('Miss.')]['Age'].mean())
    Master_age_mean = (data[data.Name.str.contains('Master.')]['Age'].mean())
    # 填充缺失的年龄
    data.loc[(data.Name.str.contains('Mr.')) & data.Age.isnull(), 'Age'] = Mr_age_mean
    data.loc[(data.Name.str.contains('Mrs.')) & data.Age.isnull(), 'Age'] = Mrs_age_mean
    data.loc[(data.Name.str.contains('Miss.')) & data.Age.isnull(), 'Age'] = Miss_age_mean
    data.loc[(data.Name.str.contains('Master.')) & data.Age.isnull(), 'Age'] = Master_age_mean

    # 将特征数据进行离散化
    data['Fare'][(data.Fare <= 7.91)] = 0
    data['Fare'][(data.Fare > 7.91) & (data.Fare <= 14.454)] = 1
    data['Fare'][(data.Fare > 14.454) & (data.Fare <= 31)] = 2
    data['Fare'][(data.Fare > 31)] = 3
    data['Age'][(data.Age <= 16)] = 0
    data['Age'][(data.Age > 16) & (data.Age <= 32)] = 1
    data['Age'][(data.Age > 32) & (data.Age <= 48)] = 2
    data['Age'][(data.Age > 48) & (data.Age <= 64)] = 3
    data['Age'][(data.Age > 64)] = 4
    data.loc[data.Sex == 'male', 'Sex'] = 0
    data.loc[data.Sex == 'female', 'Sex'] = 1
    data.loc[data.Embarked == 'C', 'Embarked'] = 0
    data.loc[data.Embarked == 'Q', 'Embarked'] = 1
    data.loc[data.Embarked == 'S', 'Embarked'] = 2
    data.loc[data.Embarked.isnull(), 'Embarked'] = 3
    del data['Ticket']
    del data['Name']
    if not ('Survived' in data.columns):
      return data.iloc[:, :].to_numpy()
    return data.iloc[:, 1:].to_numpy(), data.iloc[:, 0: 1].to_numpy()
