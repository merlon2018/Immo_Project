###########   IMPORTS   ###########
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import numpy as np
import pandas as pd
import operator
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import GPy
import GPyOpt
import xgboost as xgb
import pandas_profiling
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy.stats import uniform
import unidecode
from mpl_toolkits.mplot3d import Axes3D
###################################


#Set diplay options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
###################

#Read sqlite database in pandas dataframe and clean dataframe

list_of_str_features_to_transform = ['Surface_habitable', 'Surface_living', 'Surface_cuisine',
									 'Chambres', 'Surface_chambre_1', 'Surface_chambre_2', 'Surface_chambre_3', 'Salles_de_bain',
									 'Toilettes', 'Surface_terrasse', 'Prix_de_vente_demande',
									 'Revenu_cadastral', 'CPEB_-_Consommation_energetique', 'Emission_CO2']


def load_and_process_database(DBname, list_of_str_features_to_transform):
	conn = sqlite3.connect(
			f'/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/{DBname}.db')
	query = f'SELECT * FROM {DBname}'
	df = pd.read_sql(query, conn)
	df.columns = [unidecode.unidecode(colname).replace("'", "_")
			   for colname in df.columns]
	df.columns = df.columns.str.replace(' ', '_')
	df['Prix_de_vente_demande'] = df['Prix_de_vente_demande'].str.replace(' ', '')
	df.fillna(value=pd.np.nan, inplace=True)  # fill None with np.nan
	df.dropna(subset=['Prix_de_vente_demande'], axis=0,
		   inplace=True)  # drop missed prices
	rows_to_drop = df['Prix_de_vente_demande'].str.contains(
			'\d+[€-]+\d+[,]*\d+[€]')
	df = df[~rows_to_drop]  # drop rows that satisfy "rows_to_drop"
	for col in list_of_str_features_to_transform:
		df[col] = df[col].str.replace(' ', '')
		df[col] = df[col].map(lambda x: np.nan if pd.isnull(
					x) else re.compile("[0-9]+").search(str(x)).group())
		df[col] = df[col].astype(float)
	df.drop(labels=['Quartier_ou_lieu-dit', 'Annee_de_construction',
				 'Parkings_exterieurs', 'Parkings_interieurs', 'Attestation_as-built'], axis=1, inplace=True)
	return df


df = load_and_process_database(
	'BruxellesDBV2', list_of_str_features_to_transform)
#df.to_csv('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/complete_df.csv', index=False)
old_df = load_and_process_database(
	'BruxellesDB', list_of_str_features_to_transform)
#df.to_csv('/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/complete_old_df.csv', index=False)

df['Code_postal'] = df['Code_postal'].map(
	lambda x: re.compile("[0-9]{4}").search(str(x)).group())
#Calculate and create a new feature of average prices per municipality
df['Prix_code_postal'] = df.groupby('Code_postal')[
	'Prix_de_vente_demande'].transform(lambda x: x.mean()).astype(int)
#New feature representing the average price per square meter
df['Prix_au_m_carre'] = df['Prix_de_vente_demande']/df['Surface_habitable']
df['Prix_au_m_carre'] = df.groupby(
	'Code_postal')['Prix_au_m_carre'].transform(lambda x: x.mean()).astype(int)

#df.profile_report(style={'full_width': True})

###################################################################


def convert_num_vals(dataframe):
	for col in dataframe.columns:
		try:
			dataframe[col] = dataframe[col].astype(float)
		except ValueError:
			pass
	return dataframe


df = convert_num_vals(df)


def find_correlated_feats(dataframe):
	""" function that finds columns with missing values,
	their correlation with other columns to fit
	linear regression """
	dataframe = dataframe._get_numeric_data()
	for col in dataframe.columns:
		missing_count = dataframe[col].isna().sum()
		dict_of_corr = dict()
		for col_ in dataframe.columns:
			if col_ != col and missing_count > 0:
				corr_val = dataframe[col].corr(dataframe[col_])
				dict_of_corr[col_] = corr_val
		if bool(dict_of_corr):
			col_to_choose = max(dict_of_corr.items(), key=lambda k: abs(k[1]))
			print('I chose {} for this column {} and the correlation coeff is : {}'.format(
				col_to_choose[0], col, round(col_to_choose[1], 2)))


find_correlated_feats(df)

######################## In this part of code we replace missing values    ########################
######################## by studying the correlation between features and  ########################
######################## apply linear regression to fill in missing values ########################


def apply_model(ML_model, features, output, criteria, dataf):
	""" ML_model : sklearn regression model
	features : LIST of features even if it is one feature
	output : value to predict
	dataf : dataframe  """
	dataf = dataf.copy(deep=True)
	feats_to_drop = features + output
	dataf.dropna(axis=0, subset=feats_to_drop, inplace=True)
	indices_ = list(dataf[dataf[output[0]] > criteria *
					   dataf['Surface_habitable']].index)
	dataf.drop(axis=0, index=indices_, inplace=True)
	feats_to_concat = []
	for elem in features:
		feats_to_concat.append(dataf[elem])
	if len(features) == 1:
		X = dataf[features].values.reshape(-1, 1)
	else:
		X = pd.concat(feats_to_concat, axis=1)
	y = dataf[output].values.reshape(-1, 1)
	ML_model.fit(X, y)
	return ML_model


def replace_vals(input_cols, output_col, criteria, df, model):
	""" input_cols : input columns which we use in order to predict missing values
	output_col : the column to predict
	criteria : the coefficient used to compare with 'Surface_habitable'
	df : dataframe
	model : sklearn model to use for prediction"""
	df[output_col] = df.apply(lambda x: x[output_col] if not np.isnan(
		x[output_col]) and x[output_col] < criteria*x['Surface_habitable']
		else model.predict(np.array(x[input_cols]).reshape(1, -1))[0][0], axis=1)
	return df


lr_1 = apply_model(LinearRegression(), [
	'Surface_habitable'], ['Surface_cuisine'], 0.5, df)

df = replace_vals('Surface_habitable', 'Surface_cuisine', 0.5, df, lr_1)

lr_2 = apply_model(LinearRegression(), [
	'Surface_habitable', 'Surface_cuisine'], ['Surface_living'], 0.9, df)

df = replace_vals(['Surface_habitable', 'Surface_cuisine'],
				  'Surface_living', 0.9, df, lr_2)


###################################################################################################
###################################################################################################

###################     Data visualization      ###################
ax = sns.scatterplot(
	x="Emission_CO2", y="CPEB_-_Consommation_energetique", data=df)
ax = sns.scatterplot(x="Prix_de_vente_demande",
					 y="CPEB_-_Consommation_energetique", data=df)
# detect outliers in consommation energetique
ax = sns.boxplot(x="CPEB_-_Consommation_energetique",
				 data=df)  # detecting outlier with boxplot
df1 = df.copy(deep=True)
df1 = df1.dropna(subset=['CPEB_-_Consommation_energetique'])
ax = sns.distplot(df["CPEB_-_Consommation_energetique"])
#ax = sns.distplot(df["Prix_de_vente_demande"])
###########################################@
ax = sns.scatterplot(x="Surface_habitable", y="Surface_cuisine", data=df)
ax = sns.scatterplot(
	x="Emission_CO2", y="CPEB_-_Consommation_energetique", data=df2)
ax = sns.scatterplot(
	x="Emission_CO2", y="CPEB_-_Consommation_energetique", data=df)
ax = sns.scatterplot(x="Surface_habitable", y="Surface_living", data=df)
ax = sns.scatterplot(x="Surface_habitable", y="Revenu_cadastral", data=df)
plt.plot(df["Surface_habitable"], lr_1.predict(
	np.array(df["Surface_habitable"]).reshape(-1, 1)), color='r')

predicted = lr_2.predict(df[["Surface_habitable", "Surface_cuisine"]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(df["Surface_habitable"], df["Surface_cuisine"])
XX = xx.reshape(-1, 1)
YY = yy.reshape(-1, 1)
ZZ = lr_2.predict(np.concatenate((XX, YY), axis=1))
ax.scatter(xs=df["Surface_habitable"], ys=df["Surface_cuisine"],
		   zs=df["Surface_living"], zdir='z', s=20, c=None, depthshade=True)
ax.plot_surface(xx, yy, ZZ.reshape(xx.shape), color='r')
plt.show()

#########################################################################
###################    END of Data Visualization      ###################

#Create a new dataframe which is the concatenation of two dataframes in order to capture
#additional data to predict revenu cadastral
df['Revenu_cadastral'] = df.loc[df['Revenu_cadastral']
								> 100, ['Revenu_cadastral']].replace(np.nan)
old_df['Revenu_cadastral'] = old_df.loc[old_df['Revenu_cadastral']
										> 100, ['Revenu_cadastral']].replace(np.nan)
new_df = df.copy(deep=True)
new_df.drop(labels=['Code_postal', 'Prix_code_postal',
					'Prix_au_m_carre'], axis=1, inplace=True)
new_df = pd.concat(objs=[new_df, old_df], axis=0)
new_df.dropna(axis=0, subset=['Revenu_cadastral'], inplace=True)
new_df.drop_duplicates(inplace=True)

######### Apply linear regression model to fill
######### missed values of 'Revenu_cadastral'

X_reg = new_df["Surface_habitable"].values.reshape(-1, 1)
y_reg = new_df["Revenu_cadastral"].values.reshape(-1, 1)
#reg = LinearRegression()
#reg.fit(X_reg,y_reg)
reg = sm.OLS(y_reg, X_reg)
lm = reg.fit()
#print(lm.summary())
#### replace values predicted
df['Revenu_cadastral'] = df.apply(lambda x: lm.predict(x['Surface_habitable'])[0]
								  if pd.isnull(x['Revenu_cadastral']) else x['Revenu_cadastral'], axis=1)

df['CPEB_-_Consommation_energetique'] = df.loc[df['CPEB_-_Consommation_energetique'] < 3000,
											   ['CPEB_-_Consommation_energetique']].replace(np.nan)  # replace outlier with NaN

# df_clus = df[['CPEB_-_Consommation_energetique', 'Surface_habitable']]
# df_clus = df_clus.dropna(axis=0)
# # Code for finding the optimal number of clusters which is 5 in our case
# distortions = []
# for i in range(1,11):
# 	kmean = KMeans(n_clusters=i, init='k-means++',
# 					n_init=10, max_iter=300, random_state=1)
# 	kmean.fit(df_clus)
# 	distortions.append(kmean.inertia_)
# plt.plot(range(1,11), distortions, marker = 'o')
# plt.show()
########################################################################

# Fill missing values for consommation energetique and Emission CO2
df1 = df.copy()
# df1['CPEB_per_commune'] = df1.groupby(
# 	'Code_postal')['CPEB_-_Consommation_energetique'].transform(lambda x: x.mean())

# def surface_cluster(x):
# 	if x <= 100:
# 		return "A"
# 	elif 100 < x <= 200:
# 		return "B"
# 	elif 200 < x <= 300:
# 		return "C"
# 	else:
# 		return "D"

# df1['group_surface'] = df1['Surface_habitable'].map(surface_cluster)
# df1['CPEB_per_surface'] = df1.groupby(
# 							'group_surface')['CPEB_-_Consommation_energetique'].transform(lambda x: x.mean())
subset_ = ['Prix_de_vente_demande', 'Prix_code_postal', 'Prix_au_m_carre']
#subset_ = ['Prix_de_vente_demande', 'Prix_code_postal', 'Prix_au_m_carre', 'group_surface']
df1.dropna(subset=['CPEB_-_Consommation_energetique'], inplace=True)
df1.dropna(axis=1, inplace=True)
df1.drop(labels=subset_, axis=1, inplace=True)
X = df1.drop('CPEB_-_Consommation_energetique', axis=1).values
y = df1[['CPEB_-_Consommation_energetique']].values.reshape((-1,))
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=1)
model = RFR(n_estimators=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test).reshape(-1,)
#model.score(X_test, y_test)  # equal to r2_score
#mean_absolute_error(y_test, y_pred)
cols_for_pred = ['Code_postal', 'Surface_habitable',
				 'Surface_living', 'Surface_cuisine', 'Revenu_cadastral']
df['CPEB_-_Consommation_energetique'] = df.apply(lambda x: model.predict(np.array(x[cols_for_pred]).reshape(1, -1))[0]
												 if pd.isnull(x['CPEB_-_Consommation_energetique']) else x['CPEB_-_Consommation_energetique'], axis=1)

df_reg = df.copy()
df_reg.dropna(subset=['CPEB_-_Consommation_energetique',
					  'Emission_CO2'], inplace=True)
X = df_reg['CPEB_-_Consommation_energetique'].values.reshape(-1, 1)
Y = df_reg['Emission_CO2'].values.reshape(-1, 1)
l_reg = LinearRegression()
l_reg.fit(X, Y)
# Y = Y.astype(float)
# X = X.astype(float)
# l_reg = sm.OLS(Y,X, missing='drop')
# results = l_reg.fit()
# test_pred = l_reg.predict(X[0])
# print(results.summary())
df['Emission_CO2'] = df.apply(lambda x: l_reg.predict(np.array(x['CPEB_-_Consommation_energetique']).reshape(-1, 1))[0][0]
							  if pd.isnull(x['Emission_CO2']) else x['Emission_CO2'], axis=1)
###################################################################################################
#	Fill in missing data in 'Etage_du_bien' and 'Nombre_d_etages'
f_val = df['Etage_du_bien'].value_counts().index[0]

df['Etage_du_bien'] = df.apply(lambda x: f_val if pd.isnull(
	x['Etage_du_bien']) and pd.isnull(x['Nombre_d_etages']) else x['Etage_du_bien'], axis=1)

df['Etage_du_bien'] = df.apply(lambda x: x['Nombre_d_etages']
							   if pd.isnull(x['Etage_du_bien']) or (x['Etage_du_bien'] > x['Nombre_d_etages']) else x['Etage_du_bien'], axis=1)

df['Nombre_d_etages'] = df.apply(lambda x: x['Etage_du_bien']
								 if pd.isnull(x['Nombre_d_etages']) else x['Nombre_d_etages'], axis=1)

df['Surface_chambre_1'] = df.apply(lambda x: x['Surface_habitable']-x['Surface_living']-x['Surface_cuisine']-x['Surface_chambre_2'] -
								   x['Surface_chambre_3'] if x['Surface_chambre_1'] > x['Surface_habitable']
								   or pd.isnull(x['Surface_chambre_1']) else x['Surface_chambre_1'], axis=1)

cols_with_na = ['Cave', 'Terrasse', 'Ascenseur',
				'Acces_handicape', 'Porte_blindee', 'Double_vitrage']
for col in cols_with_na:
	df[col] = df[col].replace(np.nan, "non")

###################################################################################################


def fill_with_freq_val(dataframe):
	""" fill dataframe columns with their most 
	frequent values """
	for col in dataframe.columns:
		if dataframe[col].dtype == object:
			md = dataframe[col].mode()[0]
			dataframe[col].fillna(md, inplace=True)
		else:
			mfv = dataframe[col].value_counts().index[0]
			dataframe[col].fillna(mfv, inplace=True)
	return dataframe


df = fill_with_freq_val(df)

###############		Encoding categorical variables		###############
dict_of_conf = {'non': 0, 'oui, conforme': 1, 'oui, non conforme': 0}

df['Attestation_de_conformite_de_l_installation_electrique'] = df[
	'Attestation_de_conformite_de_l_installation_electrique'].map(dict_of_conf)

dict_bool = {"oui": 1, "non": 0}

cols_to_trans = ['Cave', 'Terrasse', 'Ascenseur',
				 'Acces_handicape', 'Porte_blindee', 'Double_vitrage']

for col in cols_to_trans:
	df[col] = df[col].map(dict_bool)


def show_cols_with_nan(dataframe):
	""" Function that returns columns with 
	nan and the number of missing values """
	dict = {}
	row, columns = dataframe.shape
	print(f'Number of rows in dataframe : {row}')
	for col in dataframe.columns:
		nans = dataframe[col].isna().sum()
		if nans > 0:
			dict[col] = nans
	for keys, values in dict.items():
		print(keys, " : ", values)


df.drop(labels=['Environnement', 'Living', 'Surface_chambre_1', 'Surface_chambre_2', 'Surface_chambre_3', 'Salles_de_douche',
				'Surface_terrasse', 'Alarme'], axis=1, inplace=True)

cat_columns = df.select_dtypes(['object']).columns
cat_subset = pd.get_dummies(df[cat_columns])
df = pd.concat([df, cat_subset], axis=1)
df.drop(labels=cat_columns, axis=1, inplace=True)
#df = df.select_dtypes(['number'])
#Split data into train & test
X = df.drop('Prix_de_vente_demande', axis=1)
y = df.pop('Prix_de_vente_demande').values.reshape((-1,))
#y_log = np.log10(y)

# plt.style.use('fivethirtyeight')
# plt.hist(y, bins = 100, edgecolor = 'k');
# plt.xlabel('Price'); plt.ylabel('Number of Buildings')
# plt.title('Real Estate Price Distribution')

# quadratic = PolynomialFeatures(degree=4)
# X_quad = quadratic.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=1)

# X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
# 	X_quad, y, test_size=0.2, random_state=1)

rf_model = RFR(n_estimators=100)
rf_model.fit(X_train, y_train)
y_rf = rf_model.predict(X_test).reshape(-1,)
# y_mean = y.mean()
# y_sigma = np.sqrt(y.var())
#model.score(X_test, y_test)
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
y_lgbm = lgb_model.predict(X_test)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

xgb_model.fit(X_train, y_train)
y_xgb = xgb_model.predict(X_test)
# y_xgb = 10**y_xgb
# y_rf = 10**y_rf
# y_test = 10**y_test
# lr_model = LinearRegression()
# lr_model.fit(X_train_q, y_train_q)
# y_qlr = lr_model.predict(X_test_q)

# print("The root mean squared error is : ", int(
# 	np.sqrt(mean_squared_error(y_test_q, y_qlr))))

print("The root mean squared error is : ", int(
	np.sqrt(mean_squared_error(y_test, y_lgbm))))

print("The root mean squared error is : ", int(
	np.sqrt(mean_squared_error(y_test, y_rf))))

print("The root mean squared error is : ", int(
	np.sqrt(mean_squared_error(y_test, y_xgb))))


# Cross Validation & Hyper-parameter tuning
param_dist = {"learning_rate": uniform(0, 1),
			  "gamma": uniform(0, 5),
			  "max_depth": range(1,50),
			  "n_estimators": range(1,300),
			  "min_child_weight": range(1,10)}

rs = RandomizedSearchCV(xgb_model, param_distributions=param_dist, 
						scoring='neg_mean_squared_error', cv=10, n_iter=25)

rs.fit(X_train, y_train)

bs = rs.best_score_
print(rs.best_score_)
print(rs.best_estimator_)

hyperparameters_space = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
		{'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
		{'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
		{'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
		{'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]

hyperparameters_design_region = GPyOpt.Design_space(space=hyperparameters_space)

def evaluate_model(model_parameters_list):
	test_score_list = []
	for model_parameters in model_parameters_list:
		classification_model = xgb.XGBRegressor(objective='reg:squarederror')

		dict_model_parameters = dict(zip([element['name'] for element in hyperparameters_space],
										 model_parameters))

		# transform types to int for discrete variables
		for key in dict_model_parameters:
			hyperparameter_description = [x for x in hyperparameters_space if x['name'] == key][0]
			if hyperparameter_description['type'] == 'discrete':
				dict_model_parameters[key] = int(dict_model_parameters[key])
		classification_model.set_params(**dict_model_parameters)
		# test_score = -np.mean(cross_validate(classification_model,
		#                       X, y, scoring='neg_mean_squared_error', cv =10)['test_score'])
		classification_model.fit(X_train,y_train)
		test_score = np.sqrt(mean_squared_error(y_test, classification_model.predict(X_test)))
		test_score_list.append(test_score)
	return test_score


hyperparameters_optimization_problem = GPyOpt.methods.BayesianOptimization(evaluate_model,  # function to optimize       
										  domain=hyperparameters_space,         # box-constrains of the problem
										  acquisition_type='EI')   # Exploration exploitation
hyperparameters_optimization_problem.run_optimization(max_iter=50)
best_params = hyperparameters_optimization_problem.x_opt
best_params = array([1.30790235e-01, 4.06825044e+00, 5.00000000e+01, 3.00000000e+02,
       1.00000000e+00])

#Test model with parameters tuned
xgb_model = xgb.XGBRegressor(learning_rate = best_params[0], gamma = best_params[1],
							max_depth = int(best_params[2]), n_estimators = int(best_params[3]), min_child_weight = int(best_params[4]), objective='reg:squarederror')

xgb_model.fit(X_train, y_train)
y_xgb = xgb_model.predict(X_test)
print("The root mean squared error is : ", int(
	np.sqrt(mean_squared_error(y_test, y_xgb))))
