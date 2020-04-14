from knn_bank import Train

#45,"blue-collar","divorced","primary","no",844,"no","no","unknown",5,"jun",1018,3,-1,0,"unknown","yes"
data = {}
data['age'] = 45
data['job'] = "blue-collar"
data['marital'] = "divorced"
data['education'] = "primary"
data['default'] = "no"
data['balance'] = 844
data['housing'] = "no"
data['loan'] = "no"
data['contact'] = "unknown"
data['day'] = 5
data['month'] = "jun"
data['duration'] = 1018
data['campaign'] = 3
data['pdays'] = -1
data['previous'] = 0
data['poutcome'] = "unknown"

train = Train().predict(data)
#train = Train().startTrain(21)
print(train)