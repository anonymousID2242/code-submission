import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

test_coverage =[]
test_precision=[]
test_acc=[]

column_names = [
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"]
data = pd.read_csv("datasets/adult.data", names=column_names)
print("Number of samples: %d" % len(data))
data.head()



def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


data.head()


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders
encoded_data, _ = number_encode_features(data)


encoded_data, encoders = number_encode_features(data)
y = encoded_data.Target.values
number_of_values=200
del encoded_data["Target"] 
X_train, X_test, y_train, y_test = model_selection.train_test_split(encoded_data[encoded_data.columns], y, train_size=0.75)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("float"))
x_new=X_train[0:number_of_values]


filename = 'saved_models/adult-nn.pickle'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_train, y_train)
result = loaded_model.predict(X_train)
result1 = loaded_model.predict(X_test)
y_new=result[0:number_of_values]
accuracy_score(result,y_train)


from sklearn.tree import _tree
column_names = [
        "Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
        "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
        "Hoursperweek", "Country", "Target"]

def tree_to_code(tree,i, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def a"+str(i)+"tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


get_ipython().run_cell_magic('capture', 'cap --no-stderr', "from sklearn import tree\nfor i in range(2,11):\n    clf = tree.DecisionTreeClassifier(max_depth=i)\n    clf = clf.fit(x_new, y_new)\n    #result1 = clf.predict(X_test)\n    y_pred = clf.predict(X_test)\n    #print(precision_score(y_pred,result1))\n    test_precision.append(precision_score(y_pred,result1))\n    test_acc.append(accuracy_score(y_pred,result1))\n    #tree_to_code(clf,i,column_names)\nwith open('code_decision_tree_adult.py', 'w') as out:\n    out.write(cap.stdout)")



import code_decision_tree_adult
y_pred



# %load code_decision_tree_adult.py
def a2tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if EducationNum <= 0.9426454603672028:
    if CapitalLoss <= 4.0046234130859375:
      return 0
    else:  # if CapitalLoss > 4.0046234130859375
      return 1
  else:  # if EducationNum > 0.9426454603672028
    if Relationship <= -0.590617448091507:
      return 1
    else:  # if Relationship > -0.590617448091507
      return 1



def a3tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if EducationNum <= 0.9426454603672028:
    if CapitalLoss <= 4.0046234130859375:
      if CapitalGain <= 0.4230954647064209:
        return 0
      else:  # if CapitalGain > 0.4230954647064209
        return 1
    else:  # if CapitalLoss > 4.0046234130859375
      if fnlwgt <= -0.839801013469696:
        return 1
      else:  # if fnlwgt > -0.839801013469696
        return 1
  else:  # if EducationNum > 0.9426454603672028
    if Relationship <= -0.590617448091507:
      if Occupation <= -0.7295635640621185:
        return 1
      else:  # if Occupation > -0.7295635640621185
        return 1
    else:  # if Relationship > -0.590617448091507
      if Hoursperweek <= 0.1663991790264845:
        return 1
      else:  # if Hoursperweek > 0.1663991790264845
        return 1

count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a2tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 1 :
            count = count+1
print(count)
k = len(X_test)
print((count)/k)
test_coverage.append((count)/k)


count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a3tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 0 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)



def a4tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if EducationNum <= 0.9426454603672028:
    if CapitalLoss <= 4.0046234130859375:
      if CapitalGain <= 0.4230954647064209:
        if fnlwgt <= 2.0990901589393616:
          return 1
        else:  # if fnlwgt > 2.0990901589393616
          return 0
      else:  # if CapitalGain > 0.4230954647064209
        return 0
    else:  # if CapitalLoss > 4.0046234130859375
      if fnlwgt <= -0.839801013469696:
        return 0
      else:  # if fnlwgt > -0.839801013469696
        return 0
  else:  # if EducationNum > 0.9426454603672028
    if Relationship <= -0.590617448091507:
      if Occupation <= -0.7295635640621185:
        return 0
      else:  # if Occupation > -0.7295635640621185
        if Workclass <= 1.1168428361415863:
          return 0
        else:  # if Workclass > 1.1168428361415863
          return 0
    else:  # if Relationship > -0.590617448091507
      if Hoursperweek <= 0.1663991790264845:
        return 0
      else:  # if Hoursperweek > 0.1663991790264845
        if fnlwgt <= 0.5877707041800022:
          return 0
        else:  # if fnlwgt > 0.5877707041800022
          return 0




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a4tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 1 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)




def a5tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if Relationship <= -0.5898788422346115:
    if EducationNum <= 0.5544191002845764:
      if Hoursperweek <= 1.788733184337616:
        if CapitalLoss <= 4.6921374797821045:
          if CapitalGain <= 0.7456265091896057:
            return 1
          else:  # if CapitalGain > 0.7456265091896057
            return 0
        else:  # if CapitalLoss > 4.6921374797821045
          return 0
      else:  # if Hoursperweek > 1.788733184337616
        return 0
    else:  # if EducationNum > 0.5544191002845764
      if Hoursperweek <= -1.2509534526616335:
        return 0
      else:  # if Hoursperweek > -1.2509534526616335
        if Age <= -0.3005167432129383:
          return 0
        else:  # if Age > -0.3005167432129383
          if Occupation <= 1.3972039818763733:
            return 0
          else:  # if Occupation > 1.3972039818763733
            return 0
  else:  # if Relationship > -0.5898788422346115
    if CapitalGain <= 0.7835536599159241:
      if Relationship <= 1.9042499661445618:
        if Hoursperweek <= 0.734975129365921:
          return 1
        else:  # if Hoursperweek > 0.734975129365921
          if EducationNum <= 0.9416324496269226:
            return 1
          else:  # if EducationNum > 0.9416324496269226
            return 0
      else:  # if Relationship > 1.9042499661445618
        if Hoursperweek <= 0.7755043338984251:
          if Country <= -0.2189738005399704:
            return 0
          else:  # if Country > -0.2189738005399704
            return 0
        else:  # if Hoursperweek > 0.7755043338984251
          return 0
    else:  # if CapitalGain > 0.7835536599159241
      return 1




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a5tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 1 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)




def a6tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if Relationship <= -0.5898788422346115:
    if EducationNum <= 0.5544191002845764:
      if Hoursperweek <= 1.788733184337616:
        if CapitalLoss <= 4.6921374797821045:
          if CapitalGain <= 0.7456265091896057:
            if EducationNum <= -0.22000757604837418:
              return 1
            else:  # if EducationNum > -0.22000757604837418
              return 1
          else:  # if CapitalGain > 0.7456265091896057
            return 0
        else:  # if CapitalLoss > 4.6921374797821045
          return 1
      else:  # if Hoursperweek > 1.788733184337616
        return 0
    else:  # if EducationNum > 0.5544191002845764
      if Hoursperweek <= -1.2509534526616335:
        return 0
      else:  # if Hoursperweek > -1.2509534526616335
        if Age <= -0.3005167432129383:
          return 1
        else:  # if Age > -0.3005167432129383
          if Occupation <= 1.3972039818763733 :
            return 1
          else:  # if Occupation > 1.3972039818763733
            return 0
  else:  # if Relationship > -0.5898788422346115
    if CapitalGain <= 0.7835536599159241:
      if Relationship <= 1.9042499661445618:
        if Hoursperweek <= 0.734975129365921:
          return 0
        else:  # if Hoursperweek > 0.734975129365921
          if EducationNum <= 0.9416324496269226:
            return 1
          else:  # if EducationNum > 0.9416324496269226
            if Hoursperweek <= 1.383441686630249:
              return 0
            else:  # if Hoursperweek > 1.383441686630249
              return 0
      else:  # if Relationship > 1.9042499661445618
        if Workclass <= 1.126318633556366:
          if Country <= -0.2189738005399704:
            return 0
          else:  # if Country > -0.2189738005399704
            if Occupation <= -0.607715617865324:
              return 1
            else:  # if Occupation > -0.607715617865324
              return 1
        else:  # if Workclass > 1.126318633556366
          return 0
    else:  # if CapitalGain > 0.7835536599159241
      return 0




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a6tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 1 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)


def a7tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if Relationship <= -0.5898788422346115:
    if EducationNum <= 0.5544191002845764:
      if Hoursperweek <= 1.788733184337616:
        if CapitalGain <= 0.7456265091896057:
          if CapitalLoss <= 4.6921374797821045:
            if EducationNum <= -0.22000757604837418:
              return 0
            else:  # if EducationNum > -0.22000757604837418
              if Age <= 0.28393687680363655:
                return 1
              else:  # if Age > 0.28393687680363655
                return 0
          else:  # if CapitalLoss > 4.6921374797821045
            return 0
        else:  # if CapitalGain > 0.7456265091896057
          return 0
      else:  # if Hoursperweek > 1.788733184337616
        return 0
    else:  # if EducationNum > 0.5544191002845764
      if Occupation <= -1.4332707524299622:
        return 0
      else:  # if Occupation > -1.4332707524299622
        if Occupation <= 1.3972039818763733:
          if Age <= -0.3005167432129383:
            return 1
          else:  # if Age > -0.3005167432129383
            return 0
        else:  # if Occupation > 1.3972039818763733
          return 0
  else:  # if Relationship > -0.5898788422346115
    if CapitalGain <= 0.7835536599159241:
      if Relationship <= 1.9042499661445618:
        if Hoursperweek <= 0.734975129365921:
          return 1
        else:  # if Hoursperweek > 0.734975129365921
          if EducationNum <= 0.9416324496269226:
            return 0
          else:  # if EducationNum > 0.9416324496269226
            if EducationNum <= 1.3288457989692688:
              if Country <= -2.1318808794021606:
                return 1
              else:  # if Country > -2.1318808794021606
                return 0
            else:  # if EducationNum > 1.3288457989692688
              return 0
      else:  # if Relationship > 1.9042499661445618
        if Country <= -0.2189738005399704:
          return 0
        else:  # if Country > -0.2189738005399704
          if Hoursperweek <= 0.7755043338984251:
            if Occupation <= -0.607715617865324:
              if Workclass <= -0.5892939791083336:
                return 1
              else:  # if Workclass > -0.5892939791083336
                return 1
            else:  # if Occupation > -0.607715617865324
              return 1
          else:  # if Hoursperweek > 0.7755043338984251
            return 1
    else:  # if CapitalGain > 0.7835536599159241
      return 0
test_coverage.append((k-count)/k)




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a7tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 0 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)



def a8tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if Relationship <= -0.5898788422346115:
    if EducationNum <= 0.5544191002845764:
      if Hoursperweek <= 1.788733184337616:
        if CapitalLoss <= 4.6921374797821045:
          if CapitalGain <= 0.7456265091896057:
            if EducationNum <= -0.22000757604837418:
              return 0
            else:  # if EducationNum > -0.22000757604837418
              if Age <= 0.28393687680363655:
                return 0
              else:  # if Age > 0.28393687680363655
                if Hoursperweek <= -0.8456618953496218:
                  return 0
                else:  # if Hoursperweek > -0.8456618953496218
                  return 0
          else:  # if CapitalGain > 0.7456265091896057
            return 0
        else:  # if CapitalLoss > 4.6921374797821045
          return 1
      else:  # if Hoursperweek > 1.788733184337616
        return 1
    else:  # if EducationNum > 0.5544191002845764
      if Hoursperweek <= -1.2509534526616335:
        return 1
      else:  # if Hoursperweek > -1.2509534526616335
        if Age <= -0.3005167432129383:
          return 1
        else:  # if Age > -0.3005167432129383
          if Occupation <= 1.3972039818763733:
            return 0
          else:  # if Occupation > 1.3972039818763733
            return 0
  else:  # if Relationship > -0.5898788422346115
    if CapitalGain <= 0.7835536599159241:
      if Relationship <= 1.9042499661445618:
        if Hoursperweek <= 0.734975129365921:
          return 1
        else:  # if Hoursperweek > 0.734975129365921
          if EducationNum <= 0.9416324496269226:
            return 0
          else:  # if EducationNum > 0.9416324496269226
            if Education <= 0.05883470177650452:
              if fnlwgt <= 1.1698413789272308:
                return 1
              else:  # if fnlwgt > 1.1698413789272308
                return 0
            else:  # if Education > 0.05883470177650452
              return 1
      else:  # if Relationship > 1.9042499661445618
        if Country <= -0.2189738005399704:
          return 1
        else:  # if Country > -0.2189738005399704
          if Hoursperweek <= 0.7755043338984251:
            if Occupation <= -0.607715617865324:
              if fnlwgt <= 2.1113431453704834:
                return 1
              else:  # if fnlwgt > 2.1113431453704834
                return 1
            else:  # if Occupation > -0.607715617865324
              return 1
          else:  # if Hoursperweek > 0.7755043338984251
            return 0
    else:  # if CapitalGain > 0.7835536599159241
      return 1




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a8tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 0 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)





def a9tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if Relationship <= -0.5898788422346115:
    if EducationNum <= 0.5544191002845764:
      if Hoursperweek <= 1.788733184337616:
        if CapitalLoss <= 4.6921374797821045:
          if CapitalGain <= 0.7456265091896057:
            if EducationNum <= -0.22000757604837418:
              return 1
            else:  # if EducationNum > -0.22000757604837418
              if Age <= 0.28393687680363655:
                return 0
              else:  # if Age > 0.28393687680363655
                if Education <= 0.31677794456481934:
                  return 0
                else:  # if Education > 0.31677794456481934
                  if fnlwgt <= -0.170230470597744:
                    return 1
                  else:  # if fnlwgt > -0.170230470597744
                    return 1
          else:  # if CapitalGain > 0.7456265091896057
            return 1
        else:  # if CapitalLoss > 4.6921374797821045
          return 1
      else:  # if Hoursperweek > 1.788733184337616
        return 1
    else:  # if EducationNum > 0.5544191002845764
      if Hoursperweek <= -1.2509534526616335:
        return 1
      else:  # if Hoursperweek > -1.2509534526616335
        if Occupation <= 1.3972039818763733:
          if Age <= -0.3005167432129383:
            return 1
          else:  # if Age > -0.3005167432129383
            return 1
        else:  # if Occupation > 1.3972039818763733
          return 1
  else:  # if Relationship > -0.5898788422346115
    if CapitalGain <= 0.7835536599159241:
      if Relationship <= 1.9042499661445618:
        if Hoursperweek <= 0.734975129365921:
          return 0
        else:  # if Hoursperweek > 0.734975129365921
          if EducationNum <= 0.9416324496269226:
            return 0
          else:  # if EducationNum > 0.9416324496269226
            if Age <= 0.9049188494682312:
              if Country <= -2.1318808794021606:
                return 0
              else:  # if Country > -2.1318808794021606
                return 1
            else:  # if Age > 0.9049188494682312
              return 1
      else:  # if Relationship > 1.9042499661445618
        if Workclass <= 1.126318633556366:
          if Country <= -0.2189738005399704:
            return 0
          else:  # if Country > -0.2189738005399704
            if Occupation <= -0.607715617865324:
              if Hoursperweek <= -1.0483076740056276:
                return 0
              else:  # if Hoursperweek > -1.0483076740056276
                return 0
            else:  # if Occupation > -0.607715617865324
              return 0
        else:  # if Workclass > 1.126318633556366
          return 0
    else:  # if CapitalGain > 0.7835536599159241
      return 1




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a9tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 1:
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)




def a10tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target):
  if Relationship <= -0.5898788422346115:
    if EducationNum <= 0.5544191002845764:
      if Hoursperweek <= 1.788733184337616:
        if CapitalGain <= 0.7456265091896057:
          if CapitalLoss <= 4.6921374797821045:
            if EducationNum <= -0.22000757604837418:
              return 1
            else:  # if EducationNum > -0.22000757604837418
              if Age <= 0.28393687680363655:
                return 0
              else:  # if Age > 0.28393687680363655
                if Hoursperweek <= -0.8456618953496218:
                  return 0
                else:  # if Hoursperweek > -0.8456618953496218
                  if Age <= 1.3797874450683594:
                    if CapitalLoss <= 2.136118598282337:
                      return 1
                    else:  # if CapitalLoss > 2.136118598282337
                      return 1
                  else:  # if Age > 1.3797874450683594
                    return 1
          else:  # if CapitalLoss > 4.6921374797821045
            return 1
        else:  # if CapitalGain > 0.7456265091896057
          return 1
      else:  # if Hoursperweek > 1.788733184337616
        return 1
    else:  # if EducationNum > 0.5544191002845764
      if Age <= -0.3005167432129383:
        return 0
      else:  # if Age > -0.3005167432129383
        if Occupation <= -1.4332707524299622:
          return 0
        else:  # if Occupation > -1.4332707524299622
          if Occupation <= 1.3972039818763733:
            return 1
          else:  # if Occupation > 1.3972039818763733
            return 1
  else:  # if Relationship > -0.5898788422346115
    if CapitalGain <= 0.7835536599159241:
      if Relationship <= 1.9042499661445618:
        if Hoursperweek <= 0.734975129365921:
          return 0
        else:  # if Hoursperweek > 0.734975129365921
          if EducationNum <= 0.9416324496269226:
            return 1
          else:  # if EducationNum > 0.9416324496269226
            if EducationNum <= 1.3288457989692688:
              if Country <= -2.1318808794021606:
                return 0
              else:  # if Country > -2.1318808794021606
                return 0
            else:  # if EducationNum > 1.3288457989692688
              return 0
      else:  # if Relationship > 1.9042499661445618
        if EducationNum <= -0.22000757604837418:
          return 0
        else:  # if EducationNum > -0.22000757604837418
          if fnlwgt <= -0.8777312934398651:
            return 1
          else:  # if fnlwgt > -0.8777312934398651
            if Hoursperweek <= -1.696774184703827:
              return 0
            else:  # if Hoursperweek > -1.696774184703827
              return 0
    else:  # if CapitalGain > 0.7835536599159241
      return 1




count = 0
for p in X_test :
        Age = p[0]
        Workclass=p[1]
        fnlwgt=p[2]
        Education=p[3]
        EducationNum=p[4]
        MartialStatus=p[5]
        Occupation=p[6]
        Relationship=p[7]
        Race=p[8]
        Sex=p[9]
        CapitalGain=p[10]
        CapitalLoss=p[11]
        Hoursperweek =p[12]
        Country =p[13]
        Target = p[13]
        m=a10tree(Age, Workclass, fnlwgt, Education, EducationNum, MartialStatus, Occupation, Relationship, Race, Sex, CapitalGain, CapitalLoss, Hoursperweek, Country, Target)
        #print(m)
        if m == 1 :
            count = count+1
print(count)
k = len(X_test)
print((k-count)/k)
test_coverage.append((k-count)/k)

test_coverage
test_precision

import pandas as pd
my_df = pd.DataFrame(test_coverage)
my_df.to_csv('results_csv/test_coverage_adult.csv', index=False, header=False)
my_df = pd.DataFrame(test_precision)
my_df.to_csv('results_csv/test_precision_adult.csv', index=False, header=False)




