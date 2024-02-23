import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def clean_training_dataset(dataset: pd.DataFrame):
    dataset['Transported'] = dataset['Transported'].astype(int)

    return clean_dataset(dataset)


def custom_encoder(dataset, feature, feature_values):
    for value in feature_values:
        feature_name = feature + '_' + value
        dataset[feature_name] = 0
        dataset.loc[dataset[feature] == value, feature_name] = 1
    return dataset


def clean_dataset(dataset: pd.DataFrame):
    dataset['Group'] = dataset['PassengerId'].str.split('_').str[0]

    dataset['Name'] = dataset['Name'].fillna("Unknown Unknown")

    dataset['Surname'] = dataset['Name'].str.split().str[-1]

    group_cabins=dataset.groupby(['Group'])['Cabin'].first()

    GHP_gb=dataset.groupby(['Group','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
    GHP_index = dataset[dataset['HomePlanet'].isna()][(dataset[dataset['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
    dataset.loc[GHP_index,'HomePlanet'] = dataset.iloc[GHP_index,:]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])

    columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset[columns] = dataset[columns].fillna(value=0)

    dataset.loc[(dataset['CryoSleep'].isna()) & (dataset['VIP'] == True), 'CryoSleep'] = False
    dataset.loc[(dataset['CryoSleep'].isna()) & (dataset['VIP'] == False), 'CryoSleep'] = True

    dataset['VIP'] = dataset['VIP'].fillna(0)
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(0)

    labels = ['VIP', 'CryoSleep']
    for label in labels:
        dataset[label] = dataset[label].astype(int)

    group_cabins=dataset.groupby(['Group']).first()
    group_cabins['GroupCabin'] = group_cabins['Cabin']
    group_cabins = group_cabins['GroupCabin']
    dataset = dataset.merge(group_cabins, on='Group')
    dataset.loc[(dataset['Cabin'].isna()), 'Cabin'] = dataset.loc[(dataset['Cabin'].isna()), 'GroupCabin']
    dataset = dataset.drop('GroupCabin', axis=1)

    dataset[['Deck', 'Cabin_number', 'Side']] = dataset['Cabin'].str.split('/', expand=True)
    dataset = dataset.drop('Cabin', axis=1)
    dataset['Cabin_number'] = dataset['Cabin_number'].fillna(value=0).astype(int)

    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Deck'] == 'G'), 'HomePlanet'] = 'Earth'

    dataset.loc[(dataset['HomePlanet'].isna()) & ~(dataset['Deck']=='D'), 'HomePlanet']='Earth'
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Deck']=='D'), 'HomePlanet']='Mars'

    dataset['Destination'] = dataset['Destination'].fillna('TRAPPIST-1e')

    exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['NoSpending']  = (dataset['Expenditure']==0).astype(int)

    dataset.loc[(dataset['Age'].isna()) & (dataset['NoSpending'] == 1), 'Age'] = np.random.randint(0, 18)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

    age_groups = ['0-12', '12-18', '18-25', '25-30', '30-50', '50+']
    for age_group in age_groups:
        dataset[age_group] = 0

    dataset.loc[dataset['Age']<=12, age_groups[0]] = 0
    dataset.loc[(dataset['Age']>12) & (dataset['Age'] <= 18), age_groups[1]] = 1
    dataset.loc[(dataset['Age']>18) & (dataset['Age'] <= 25), age_groups[2]] = 2
    dataset.loc[(dataset['Age']>25) & (dataset['Age'] <= 30), age_groups[3]] = 3
    dataset.loc[(dataset['Age']>30) & (dataset['Age'] <= 50), age_groups[4]] = 4
    dataset.loc[dataset['Age']>50, age_groups[5]] = 5

    dataset['Cabin_region1']=(dataset['Cabin_number']<300).astype(int)
    dataset['Cabin_region2']=((dataset['Cabin_number']>=300) & (dataset['Cabin_number']<600)).astype(int)
    dataset['Cabin_region3']=((dataset['Cabin_number']>=600) & (dataset['Cabin_number']<900)).astype(int)
    dataset['Cabin_region4']=((dataset['Cabin_number']>=900) & (dataset['Cabin_number']<1200)).astype(int)
    dataset['Cabin_region5']=((dataset['Cabin_number']>=1200) & (dataset['Cabin_number']<1500)).astype(int)
    dataset['Cabin_region6']=((dataset['Cabin_number']>=1500) & (dataset['Cabin_number']<1800)).astype(int)
    dataset['Cabin_region7']=(dataset['Cabin_number']>=1800).astype(int)

    dataset = custom_encoder(dataset, 'Side', ['P', 'S'])
    dataset = custom_encoder(dataset, 'Deck', ['A','B','C','D','E','F','G'])
    dataset = custom_encoder(dataset, 'HomePlanet', ['Earth', 'Mars', 'Europa'])
    dataset = custom_encoder(dataset, 'Destination', ['TRAPPIST-1e','PSO J318.5-22','55 Cancri e'])

    exp_feats.append('Expenditure')
    for col in exp_feats:
        dataset[col] = np.log(1+dataset[col])

    dataset = dataset.drop('PassengerId', axis=1)
    dataset = dataset.drop('Name', axis=1)
    dataset = dataset.drop('Age', axis=1)
    dataset = dataset.drop('Surname', axis=1)
    dataset = dataset.drop('Group', axis=1)
    dataset = dataset.drop('HomePlanet', axis=1)
    dataset = dataset.drop('Destination', axis=1)
    dataset = dataset.drop('Cabin_number', axis=1)
    dataset = dataset.drop('Deck', axis=1)
    dataset = dataset.drop('Side', axis=1)

    # print(dataset.info())
    # print(dataset.info())

    return dataset


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


if __name__ == '__main__':
    dataset = pd.read_csv("./train.csv")

    dataset = clean_training_dataset(dataset)

    train_ds, test_ds = split_dataset(dataset, 0.1)

    train_X = train_ds.drop("Transported", axis=1)
    train_Y = train_ds['Transported']

    test_X = test_ds.drop('Transported', axis=1)
    test_Y = test_ds['Transported']

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC, LinearSVC
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score

    # grid_params = {'n_estimators': [50, 100, 150, 200], 'max_depth': [4, 8, 12], 'learning_rate': [0.05, 0.1, 0.15]}

    # sh = GridSearchCV(estimator=LGBMClassifier(n_jobs=4, random_state=0), param_grid=grid_params).fit(train_X, train_Y)
    # print("->", sh.best_params_)
    # print('->', sh.best_score_)

    model = LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.15, num_leaves=31, n_jobs=4, random_state=0, verbose=-1)

    X = dataset.drop("Transported", axis=1)
    Y = dataset['Transported']

    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(scores, scores.mean(), scores.std())

    #model = SVC(C=50, degree=5)
    #model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.10, max_depth=4, random_state=0)
    model = model.fit(train_X, train_Y)

    test_res = model.predict(test_X)
    train_res = model.predict(train_X)

    from sklearn.metrics import accuracy_score

    print("Test accuracy :", accuracy_score(test_Y, test_res))
    print("Train accuracy :", accuracy_score(train_Y, train_res))

    sub_dataset = pd.read_csv('test.csv')

    sub_dataset = clean_dataset(sub_dataset)

    sub_results = model.predict(sub_dataset)

    submission = pd.read_csv('sample_submission.csv')
    submission['Transported'] = sub_results
    submission['Transported'] = submission['Transported'].astype(bool)
    submission.to_csv('submission.csv', index=False)