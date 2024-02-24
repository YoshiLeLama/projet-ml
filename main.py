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
    # On récupère les numéros de groupe
    dataset['Group'] = dataset['PassengerId'].str.split('_').str[0]

    # On récupère les noms de famille (inutile pour l'instant)
    dataset['Name'] = dataset['Name'].fillna("Unknown Unknown")
    dataset['Surname'] = dataset['Name'].str.split().str[-1]

    # Les membres d'un même groupe viennent tous de la même planète
    GHP_gb=dataset.groupby(['Group','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
    GHP_index = dataset[dataset['HomePlanet'].isna()][(dataset[dataset['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
    dataset.loc[GHP_index,'HomePlanet'] = dataset.iloc[GHP_index,:]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])

    # On suppose que ceux pour qui des valeurs de dépense manquent n'ont pas dépensé
    columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset[columns] = dataset[columns].fillna(value=0)

    # On ajoute une colonne contenant la somme des achats
    # et une colonne décrivant si l'individu a acheté quelque chose ou non
    exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['NoSpending']  = (dataset['Expenditure']==0).astype(int)

    # Les VIP n'ont majoritairement pas pris de CryoSleep
    dataset.loc[(dataset['CryoSleep'].isna()) & (dataset['VIP'] == True), 'CryoSleep'] = False
    dataset.loc[(dataset['CryoSleep'].isna()) & (dataset['VIP'] == False), 'CryoSleep'] = True

    # On suppose que ceux qui ne sont pas notés VIP ne le sont pas, idem pour CryoSleep
    dataset['VIP'] = dataset['VIP'].fillna(0)
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(0)

    # On change les types booléens en type entier
    labels = ['VIP', 'CryoSleep']
    for label in labels:
        dataset[label] = dataset[label].astype(int)

    # On suppose que les membres d'un groupe dont la cabin est inconnue sont
    # dans la même cabin que l'un des autres membres du groupe
    group_cabins=dataset.groupby(['Group']).first()
    group_cabins['GroupCabin'] = group_cabins['Cabin']
    group_cabins = group_cabins['GroupCabin']
    dataset = dataset.merge(group_cabins, on='Group')
    dataset.loc[(dataset['Cabin'].isna()), 'Cabin'] = dataset.loc[(dataset['Cabin'].isna()), 'GroupCabin']
    dataset = dataset.drop('GroupCabin', axis=1)

    # On divise les identifiants de cabine en deck/numéro/côté
    dataset[['Deck', 'Cabin_number', 'Side']] = dataset['Cabin'].str.split('/', expand=True)
    dataset = dataset.drop('Cabin', axis=1)
    dataset['Cabin_number'] = dataset['Cabin_number'].fillna(value=0).astype(int)

    # Les Terriens sont majoritairement au deck G, ceux d'Europa aux decks C et B et les martiens au deck F
    dataset.loc[(dataset['Deck'].isna()) & (dataset['HomePlanet'] == 'Earth'), 'Deck'] = 'G'
    dataset.loc[(dataset['Deck'].isna()) & (dataset['HomePlanet'] == 'Europa') & (dataset['NoSpending'] == 0), 'Deck'] = 'C'
    dataset.loc[(dataset['Deck'].isna()) & (dataset['HomePlanet'] == 'Europa'), 'Deck'] = 'B'
    dataset['Deck'] = dataset['Deck'].fillna('F')

    # Les decks A, B, C et T ne contiennent que individus venant d'Europe
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
    # Le deck G ne contient que des Terriens
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Deck'].isin(['E', 'F', 'G'])), 'HomePlanet'] = 'Earth'
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Deck'] == 'D'), 'HomePlanet'] = 'Mars'

    # Ceux allant vers TRAPPIST et PSO sont très majoritairement Terriens, et ceux allant vers Cancri sont majoritairement d'Europe
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Destination'].isin(['TRAPPIST-1e', 'PSO J318.5-22'])), 'HomePlanet'] = 'Earth'
    dataset['HomePlanet'] = dataset['HomePlanet'].fillna('Europa')

    # La majorité va à TRAPPIST
    dataset['Destination'] = dataset['Destination'].fillna('TRAPPIST-1e')

    # On suppose que les individus qui n'ont pas dépensé sont des enfants (0-18)
    dataset.loc[(dataset['Age'].isna()) & (dataset['NoSpending'] == 1), 'Age'] = np.random.randint(0, 18)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

    # On divise la colonne âge en groupes d'âge
    age_groups = ['0-12', '12-18', '18-25', '25-30', '30-50', '50+']
    for age_group in age_groups:
        dataset[age_group] = 0

    dataset.loc[dataset['Age']<=12, age_groups[0]] = 1
    dataset.loc[(dataset['Age']>12) & (dataset['Age'] <= 18), age_groups[1]] = 1
    dataset.loc[(dataset['Age']>18) & (dataset['Age'] <= 25), age_groups[2]] = 1
    dataset.loc[(dataset['Age']>25) & (dataset['Age'] <= 30), age_groups[3]] = 1
    dataset.loc[(dataset['Age']>30) & (dataset['Age'] <= 50), age_groups[4]] = 1
    dataset.loc[dataset['Age']>50, age_groups[5]] = 1

    # On divise les numéros de cabine en région, car chaque région voit une tendance différente
    # concernant la probabilité d'avoir été transporté ou non
    dataset['Cabin_region1']=(dataset['Cabin_number']<300).astype(int)
    dataset['Cabin_region2']=((dataset['Cabin_number']>=300) & (dataset['Cabin_number']<600)).astype(int)
    dataset['Cabin_region3']=((dataset['Cabin_number']>=600) & (dataset['Cabin_number']<900)).astype(int)
    dataset['Cabin_region4']=((dataset['Cabin_number']>=900) & (dataset['Cabin_number']<1200)).astype(int)
    dataset['Cabin_region5']=((dataset['Cabin_number']>=1200) & (dataset['Cabin_number']<1500)).astype(int)
    dataset['Cabin_region6']=((dataset['Cabin_number']>=1500) & (dataset['Cabin_number']<1800)).astype(int)
    dataset['Cabin_region7']=(dataset['Cabin_number']>=1800).astype(int)

    # On divise les colonnes de chaînes de caractères en plusieurs colonnes d'entiers 
    # (1 si correspond à la valeur de la chaîne, 0 sinon)
    dataset = custom_encoder(dataset, 'Side', ['P', 'S'])
    dataset = custom_encoder(dataset, 'Deck', ['A','B','C','D','E','F','G','T'])
    dataset = custom_encoder(dataset, 'HomePlanet', ['Earth', 'Mars', 'Europa'])
    dataset = custom_encoder(dataset, 'Destination', ['TRAPPIST-1e','PSO J318.5-22','55 Cancri e'])

    # On passe les dépenses au logarithme pour réduire la variance
    exp_feats.append('Expenditure')
    for col in exp_feats:
        dataset[col] = np.log(1+dataset[col])

    # On supprime les features inutiles
    dataset = dataset.drop('PassengerId', axis=1)
    dataset = dataset.drop('Name', axis=1)
    dataset = dataset.drop('Surname', axis=1)
    dataset = dataset.drop('Age', axis=1)
    dataset = dataset.drop('Group', axis=1)
    dataset = dataset.drop('HomePlanet', axis=1)
    dataset = dataset.drop('Destination', axis=1)
    dataset = dataset.drop('Cabin_number', axis=1)
    dataset = dataset.drop('Deck', axis=1)
    dataset = dataset.drop('Side', axis=1)


    # print(dataset.info())
    # print(dataset.describe())

    return dataset


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


if __name__ == '__main__':
    dataset = pd.read_csv("./train.csv")

    #exit(0)

    train_ds, test_ds = split_dataset(dataset, 0.1)
    train_ds = train_ds.reset_index(drop=True)
    test_ds = test_ds.reset_index(drop=True)

    train_ds = clean_training_dataset(train_ds)
    test_ds = clean_training_dataset(test_ds)

    train_X = train_ds.drop("Transported", axis=1)
    train_Y = train_ds['Transported']

    test_X = test_ds.drop('Transported', axis=1)
    test_Y = test_ds['Transported']

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC, LinearSVC
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score

    dataset = clean_training_dataset(dataset)

    X = dataset.drop("Transported", axis=1)
    Y = dataset['Transported']

    # grid_params = {'n_estimators': [50, 100, 150], 'max_depth': [4, 8, 12], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]}
    # clf = GridSearchCV(estimator=LGBMClassifier(n_jobs=8, random_state=0, verbose=-1), param_grid=grid_params).fit(X, Y)
    # print("->", clf.best_params_)
    # print('->', clf.best_score_)

    # model = clf.best_estimator_

    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31, n_jobs=8, random_state=0, verbose=-1)
    # model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=0)
    # model = RandomForestClassifier(n_estimators=200, max_depth=4)

    print(model.get_params())

    #model = SVC(C=100, degree=5)

    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(scores, scores.mean(), scores.std())

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