from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
from pandas.api.types import CategoricalDtype

class Train:
	def __init__(self):
		self.job_category = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
		self.marital_category = ["married","divorced","single"]
		self.education_category = ["unknown","secondary","primary","tertiary"]
		self.default_category = ["yes","no"]
		self.housing_category = ["yes","no"]
		self.loan_category = ["yes","no"]
		self.contact_category = ["unknown","telephone","cellular"]
		self.month_category = ["jan","feb","mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
		self.poutcome_category = ["unknown","other","failure","success"]

	def startTrain(self, jumlah_k):
		# Read Dataset
		bank_data = pd.read_csv('dataset/bank-full.csv', encoding='utf-8')

		# Separate dataset to x and y
		x = bank_data.iloc[:,0:16]
		x.columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
		y = bank_data.iloc[:,16]
		y.columns = ['target']

		# normalize data to categorical
		x['job'] = x.job.astype('category', CategoricalDtype(self.job_category)).cat.codes
		x['marital'] = x.marital.astype('category', CategoricalDtype(self.marital_category)).cat.codes
		x['education'] = x.education.astype('category', CategoricalDtype(self.education_category)).cat.codes
		x['default'] = x.default.astype('category', CategoricalDtype(self.default_category)).cat.codes
		x['housing'] = x.housing.astype('category', CategoricalDtype(self.housing_category)).cat.codes
		x['loan'] = x.loan.astype('category', CategoricalDtype(self.loan_category)).cat.codes
		x['contact'] = x.contact.astype('category', CategoricalDtype(self.contact_category)).cat.codes
		x['month'] = x.month.astype('category', CategoricalDtype(self.month_category)).cat.codes
		x['poutcome'] = x.poutcome.astype('category', CategoricalDtype(self.poutcome_category)).cat.codes

		# category_columns = x.select_dtypes(['category']).columns
		# x[category_columns] = x[category_columns].apply(lambda l: l.cat.codes)
		# #print(x.head())

		# Split data to data train and data test
		(trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.20, random_state=42)

		# Join x and y data to one array
		train_data = trainX.join(trainY)
		test_data = testX.join(testY)

		# save data train and data test to csv
		train_data.to_csv('dataset/train_data.csv', sep=',', header=False, index=False)
		test_data.to_csv('dataset/test_data.csv', sep=',', header=False, index=False)

		# train knn
		knn = KNeighborsClassifier(n_neighbors=jumlah_k)
		knn.fit(trainX, trainY)

		# save model to pkl file
		joblib.dump(knn, 'model/my_model.pkl', compress=9)

		# calculate accuracy
		report_csv = {}
		report = classification_report(testY, knn.predict(testX))
		'''report_csv['no'] = report['no']
		report_csv['yes'] = report['yes']
		report_csv['weighted avg'] = report['weighted avg']
		
		report_csv = pd.DataFrame.from_dict(report_csv)
		report_csv.to_csv('akurasi.csv', sep=',', header=False, index=False)'''

		# jumlah_k = pd.DataFrame([jumlah_k])
		# jumlah_k.to_csv('jumlah_k.csv', sep=',', header=False, index=False)

		print(report)
		print(type(testX))
	
	def predict(self, dataform):
		x = pd.DataFrame(dataform, index=[0])
		#x.columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']

		# normalize data to categorical
		x['job'] = x.job.astype('category', CategoricalDtype(self.job_category)).cat.codes
		x['marital'] = x.marital.astype('category', CategoricalDtype(self.marital_category)).cat.codes
		x['education'] = x.education.astype('category', CategoricalDtype(self.education_category)).cat.codes
		x['default'] = x.default.astype('category', CategoricalDtype(self.default_category)).cat.codes
		x['housing'] = x.housing.astype('category', CategoricalDtype(self.housing_category)).cat.codes
		x['loan'] = x.loan.astype('category', CategoricalDtype(self.loan_category)).cat.codes
		x['contact'] = x.contact.astype('category', CategoricalDtype(self.contact_category)).cat.codes
		x['month'] = x.month.astype('category', CategoricalDtype(self.month_category)).cat.codes
		x['poutcome'] = x.poutcome.astype('category', CategoricalDtype(self.poutcome_category)).cat.codes

		#category_columns = x.select_dtypes(['category']).columns
		#x[category_columns] = x[category_columns].apply(lambda l: l.cat.codes)

		loaded_model = joblib.load('model/my_model.pkl')
		y = loaded_model.predict(x)
		print(type(x))
		return y
