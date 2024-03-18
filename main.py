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

    # On ajoute une colonne contenant la somme des achats
    # et une colonne décrivant si l'individu a acheté quelque chose ou non
    exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset[exp_feats] = dataset[exp_feats].fillna(value=0)
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['NoSpending']  = (dataset['Expenditure']==0).astype(int)

    # Remplissage des données manquantes

    # Les membres d'un même groupe viennent tous de la même planète
    GHP_gb=dataset.groupby(['Group','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
    GHP_index = dataset[dataset['HomePlanet'].isna()][(dataset[dataset['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
    dataset.loc[GHP_index,'HomePlanet'] = dataset.iloc[GHP_index,:]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])

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

    # Ceux qui n'ont rien dépensé sont en CryoSleep
    dataset.loc[(dataset['CryoSleep'].isna()) & (dataset['NoSpending'] == 1), 'CryoSleep'] = True

    # Les enfants de moins de 12 ans ne dépensent pas, donc on suppose qu'une personne qui ne dépense pas 
    # en n'étant pas en CryoSleep est un enfant (-12 ans)
    dataset.loc[(dataset['Age'].isna()) & (dataset['CryoSleep'] == 0) & (dataset['NoSpending'] == 1), 'Age'] = 6
    # la valeur exacte de l'âge n'a pas d'importance, le tout est qu'il soit dans [0,12] 

    # On suppose que ceux qui ne sont pas notés VIP ne le sont pas, idem pour CryoSleep
    dataset['VIP'] = dataset['VIP'].fillna(0)
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(0)

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

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

    # Optimisation

    # On regarde qui voyage seul ou non
    groups_size = dataset['Group'].value_counts().reset_index()
    groups_size.columns = ['Group', 'GroupSize']
    dataset = dataset.merge(groups_size, on='Group')
    dataset.loc[(dataset['GroupSize'] == 1), 'Solo'] = 1
    dataset['Solo'] = dataset['Solo'].fillna(0).astype(int)

    # Adaptation des données pour le modèle 

    # On change les types booléens en type entier
    labels = ['VIP', 'CryoSleep']
    for label in labels:
        dataset[label] = dataset[label].astype(int)

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
    sides = ['P', 'S']
    dataset = custom_encoder(dataset, 'Side', sides)
    decks = ['A','B','C','D','E','F','G','T']
    dataset = custom_encoder(dataset, 'Deck', decks)
    planets = ['Earth', 'Mars', 'Europa']
    dataset = custom_encoder(dataset, 'HomePlanet', planets)
    destinations = ['TRAPPIST-1e','PSO J318.5-22','55 Cancri e']
    dataset = custom_encoder(dataset, 'Destination', destinations)

    # On passe les dépenses au logarithme pour réduire la variance
    exp_feats.append('Expenditure')
    for col in exp_feats:
        dataset[col] = np.log(1+dataset[col])
        
    # dataset['NoSpaSpending'] = (dataset['Spa'] <= 0).astype(int)
    # dataset['SpaPlus5Spending'] = (dataset['Spa'] > 5).astype(int)
    # dataset = dataset.drop('Spa', axis=1)
    
    # dataset['NoVRDeckSpending'] = (dataset['VRDeck'] <= 0).astype(int)
    # dataset['VRDeckPlus6Spending'] = (dataset['VRDeck'] >= 6).astype(int)
    # dataset = dataset.drop('VRDeck', axis=1)
    
    # dataset['NoRoomServiceSpending'] = (dataset['RoomService'] <= 0).astype(int)
    # dataset['RoomServicePlus6Spending'] = (dataset['RoomService'] >= 6).astype(int)
    # dataset = dataset.drop('RoomService', axis=1)
    
    # dataset['NoShoppingMallSpending'] = (dataset['ShoppingMall'] <= 0).astype(int)
    # dataset['ShoppingMall0to6'] = ((dataset['ShoppingMall'] > 0) & (dataset['ShoppingMall'] <= 6)).astype(int)
    # dataset = dataset.drop('ShoppingMall', axis=1)
    
    # dataset['NoFoodCourtSpending'] = (dataset['FoodCourt'] <= 0).astype(int)
    # dataset['FoodCourt0to6'] = ((dataset['FoodCourt'] > 0) & (dataset['FoodCourt'] <= 6)).astype(int)
    # dataset = dataset.drop('FoodCourt', axis=1)
    
    # dataset['ExpenditurePlus6Spending'] = ((dataset['Expenditure'] >= 6)).astype(int)
    # dataset = dataset.drop('Expenditure', axis=1)
        
    # On supprime les features inutiles
    dataset = dataset.drop('PassengerId', axis=1)
    dataset = dataset.drop('Name', axis=1)
    dataset = dataset.drop('Age', axis=1)
    dataset = dataset.drop('Group', axis=1)
    dataset = dataset.drop('GroupSize', axis=1)
    dataset = dataset.drop('Solo', axis=1)
    dataset = dataset.drop('HomePlanet', axis=1)
    dataset = dataset.drop('Destination', axis=1)
    dataset = dataset.drop('Cabin_number', axis=1)
    dataset = dataset.drop('Deck', axis=1)
    dataset = dataset.drop('Side', axis=1)
    dataset = dataset.drop('Side_P', axis=1)
    dataset = dataset.drop('VIP', axis=1)
    # dataset = dataset.drop(exp_feats, axis=1)
    # dataset = dataset.drop('NoSpending', axis=1)
    #for i in [3, 5, 6, 7]:
    #    dataset = dataset.drop('Cabin_region'+str(i), axis=1)
    dataset = dataset.drop('Deck_T', axis=1)
    # for deck in ['A', 'D', 'G', 'T']:
    #     dataset = dataset.drop('Deck_' + deck, axis=1)
    # for planet in ['Mars']:
    #     dataset = dataset.drop('HomePlanet_' + planet, axis=1)
    # for destination in ['PSO J318.5-22']:
    #     dataset = dataset.drop('Destination_' + destination, axis=1)
    # for age_group in ['25-30', '30-50', '50+']:
    #     dataset = dataset.drop(age_group, axis=1)
    
    print(dataset.info())
    # print(dataset.describe())

    return dataset


def split_dataset(dataset, test_ratio=0.1):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

def evaluate_bool_feature(dataset: pd.DataFrame, feature: str):
    if (dataset[feature].dtype != 'int64'):
        return 1
    
    true_transported = dataset[(dataset['Transported'] == 1) & (dataset[feature] == 1)].count().iloc[0]
    true_not_transported = dataset[(dataset[feature] == 1)].count().iloc[0] - true_transported
    
    return abs(true_transported - true_not_transported) / len(dataset)

if __name__ == '__main__':
    dataset = pd.read_csv("./train.csv")

    dataset = clean_training_dataset(dataset)
        
    kept_features = []
    features = []
    
    for feature in dataset:
        if feature == 'Transported':
            continue
        
        eval = evaluate_bool_feature(dataset, feature)
        
        print(feature, eval)
        
        features.append((feature, eval))
    
    features.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(22):
        kept_features.append(features[i][0])
            
    print(kept_features)
    
    dataset = dataset[kept_features + ['Transported']]
    
    train_ds, test_ds = split_dataset(dataset, 0.05)

    train_X = train_ds.drop("Transported", axis=1)
    train_Y = train_ds['Transported']

    test_X = test_ds.drop('Transported', axis=1)
    test_Y = test_ds['Transported']

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

    X = dataset.drop("Transported", axis=1)
    Y = dataset['Transported']

    # grid_params = {'n_estimators': [50, 100, 150], 'max_depth': [4, 8, 12], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]}
    # clf = GridSearchCV(estimator=LGBMClassifier(n_jobs=8, random_state=0, verbose=-1), param_grid=grid_params).fit(X, Y)
    # print("->", clf.best_params_)
    # print('->', clf.best_score_)

    # model = clf.best_estimator_

    # lgb_params = {
    #     'n_estimators': 100,
    #     'max_depth': 7,
    #     'learning_rate': 0.05,
    #     'subsample': 0.2,
    #     'colsample_bytree': 0.56,
    #     'reg_alpha': 0.25,
    #     'reg_lambda': 5e-08,
    #     'objective': 'binary',
    #     'metric': 'accuracy',
    #     'boosting_type': 'gbdt',
    #     'device': 'cpu',
    #     'random_state': 1,
    # }

    # model = LGBMClassifier(**lgb_params, verbose=-1)

    # param_dist = {
    #     'n_estimators': np.arange(50, 1000,50),
    #     'max_depth': np.arange(3, 15,2),
    #     'learning_rate': np.arange(0.001, 0.02,0.002),
    #     'subsample': [0.1,0.3,0.5,0.7,0.9],
    #     'colsample_bytree': [0.1,0.3,0.5,0.7,0.9],
    # }
    
    # random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=3, n_iter=20, random_state=1, n_jobs=-1)
    # random_search.fit(train_X, train_Y)
    # print("Best hyperparameters: ", random_search.best_params_)
    # print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))
    # model = random_search.best_estimator_
    # lgb_params=random_search.best_params_

    model = CatBoostClassifier(n_estimators=200, max_depth=4)
    # model = LGBMClassifier(n_estimators=150, learning_rate=0.15, max_depth=4, n_jobs=8, random_state=0, verbose=-1)
    # model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=0)
    # model = RandomForestClassifier(n_estimators=200, max_depth=4)
    # model = SVC()
    # model = GaussianNB()

    print(model.get_params())


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
    
    sub_dataset = sub_dataset[kept_features]
    
    print(sub_dataset.info())

    sub_results = model.predict(sub_dataset)

    submission = pd.read_csv('sample_submission.csv')
    submission['Transported'] = sub_results
    submission['Transported'] = submission['Transported'].astype(bool)
    submission.to_csv('submission.csv', index=False)
