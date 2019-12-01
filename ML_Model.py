###########   IMPORTS   ###########
from mpl_toolkits.mplot3d import Axes3D
import unidecode
from sklearn.linear_model import LinearRegression
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import pandas_profiling
import re
import sqlite3
import operator
import pandas as pd
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
###################################


#Set diplay options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
###################

#Read sqlite database in pandas dataframe and clean dataframe
conn = sqlite3.connect(
    '/Users/macbookpro_anas/Desktop/Machine-learning/scrapper_immo/BruxellesDB.db')
query = 'SELECT * FROM BruxellesDB'
df = pd.read_sql(query, conn)
# Replace spaces in column names with underscore
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

""" remove spaces from feature values, 
replace np.nan values with 00000 to avoid errors with the map function, 
extract digits from features and finally put back the np.nan values
"""

list_of_str_features_to_transform = ['Surface_habitable', 'Surface_living', 'Surface_cuisine',
                                     'Chambres', 'Surface_chambre_1', 'Surface_chambre_2', 'Surface_chambre_3', 'Salles_de_bain',
                                     'Toilettes', 'Surface_terrasse', 'Prix_de_vente_demande',
                                     'Revenu_cadastral', 'CPEB_-_Consommation_energetique', 'Emission_CO2']
for col in list_of_str_features_to_transform:
    df[col] = df[col].str.replace(' ', '')
    # This is made in order to avoid AttributeError: 'NoneType' object has no attribute
    # 'group' and Nonetype object is not callable
    df[col].fillna(value="00000", inplace=True)
    df[col] = df[col].map(lambda x: re.compile(
        "[0-9]+").search(str(x)).group())
    df[col] = df[col].astype(int)
    df[col].replace(to_replace=[00000], value=np.nan, inplace=True)

###################################################################
""" function that finds columns with missing values, 
    their correlation with other columns and fits 
    linear regression """


def find_correlated_feats(dataframe):
    dataframe = dataframe._get_numeric_data()
    dict_of_corr = dict()
    for col in dataframe:
        missing_count = dataframe[col].isna().sum()
        for col_ in dataframe:
            if col_ != col and missing_count > 0:
                corr_val = dataframe[col].corr(dataframe[col_])
                dict_of_corr[col_] = corr_val
        col_to_choose = max(dict_of_corr.items(),
                            key=operator.itemgetter(1))[0]
        print('I chose {} for this column {} and the correlation coeff is : {}'.format(
            col_to_choose, col, round(corr_val, 2)))


indices_ = list(df[df['Surface_cuisine'] > 0.5*df['Surface_habitable']].index)


def drop_rows(list_of_feats, indices, dataf):
    feats_to_concat = []
    for elem in list_of_feats:
        feats_to_concat.append(dataf[elem])
    df_ = pd.concat(feats_to_concat, axis=1)
    df_.drop(axis=0, index=indices, inplace=True)
    df_.dropna(axis=0, inplace=True)
    return df_


rows = drop_rows(['Surface_habitable', 'Surface_living',
                  'Surface_cuisine'], indices_, df)


def apply_model(ML_model, features, output, dataf):
    """ ML_model : sklearn regression model
    features : list of features even if it is one feature
    output : value to predict
    dataf : dataframe  """
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
    df[output_col] = df.apply(lambda x: x[output_col] if not np.isnan(
        x[output_col]) and x[output_col] < criteria*x['Surface_habitable']
        else model.predict(np.array(x[input_cols]).reshape(1, -1))[0][0], axis=1)
    return df


lr_1 = apply_model(LinearRegression(), [
    'Surface_habitable'], 'Surface_cuisine', rows)

df = replace_vals('Surface_habitable', 'Surface_cuisine', 0.5, df, lr_1)

lr_2 = apply_model(LinearRegression(), [
                   'Surface_habitable', 'Surface_cuisine'], 'Surface_living', rows)

df = replace_vals(['Surface_habitable', 'Surface_cuisine'],
                  'Surface_living', 0.9, df, lr_2)


###################################################################
###################     Data visualization      ###################
ax = sns.scatterplot(x="Surface_habitable", y="Surface_cuisine", data=df)
ax = sns.scatterplot(x="Surface_habitable", y="Surface_living", data=df)
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
# TUPLE + STRING  ==>   IMMUTABLE
# LIST ==> MUTABLE
""" Function that returns columns with nan and the number of missing values """


def show_cols_with_nan(dataframe):
    dict = {}
    row, columns = dataframe.shape
    print(f'Number of rows in dataframe : {row}')
    for col in dataframe.columns:
        nans = dataframe[col].isna().sum()
        if nans > 0:
            dict[col] = nans
    return dict


df.drop(labels=['Parkings_interieurs',
                'Parkings_exterieurs', 'Parkings_interieurs'], axis=1, inplace=True)
# series_to_concat = [df['Surface_habitable'], df['Revenu_cadastral']]
# mini_df1 = pd.concat(series_to_concat, axis=1)

# """ Apply linear regression model to fill
#     missed values of 'Revenu_cadastral'"""

# mini_df = pd.concat(series_to_concat, axis=1)
# indices = list(df[df['Revenu_cadastral']<100].index)
# mini_df.drop(axis = 0, index = indices, inplace = True)
# mini_df = mini_df.dropna(axis=0)
# X_reg = mini_df["Surface_habitable"].values.reshape(-1, 1)
# y_reg = mini_df["Revenu_cadastral"].values.reshape(-1, 1)
# reg = LinearRegression()
# reg.fit(X_reg,y_reg)

#### replace values predicted
mini_df1["state"] = mini_df1["Revenu_cadastral"].map(
    lambda x: np.isnan(x) or x < 100)
df['Revenu_cadastral'] = mini_df1.T.apply(lambda x: x['Revenu_cadastral'] if not x["state"]
                                          else reg.predict(np.array(x['Surface_habitable']).reshape(-1, 1)))
df['Revenu_cadastral'] = [int(x) for x in df['Revenu_cadastral']]
###################################################################################################
#Split data into train & test
X = df.drop('Prix_de_vente_demande', axis=1)
y = df.pop('Prix_de_vente_demande')
