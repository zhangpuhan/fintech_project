import pandas as pd
import datetime
import numpy as np
from datetime import datetime
import xgboost as xgb
import pickle

class XgbModel():
    def __init__(self):
        super().__init__()

    def predict(self):
        df_current = pd.read_json("current_list.txt")

        columnNameNow = df_current.columns.values
        columnName2015 = pd.read_csv("2015columnname.txt", header=None).values.reshape(-1)
        
        columnNameNow_mlt = columnNameNow.copy()
        columnName2015_mlt = columnName2015.copy()

        for i in range(len(columnName2015_mlt)):
            columnName2015_mlt[i] = (columnName2015_mlt[i]).replace("_", "")

        for i in range(len(columnNameNow_mlt)):
            columnNameNow_mlt[i] = (columnNameNow_mlt[i]).lower()

        for i in range(len(columnName2015_mlt)):
            columnName2015_mlt[i] = (columnName2015_mlt[i]).replace("amnt", "amount")

        for i in range(len(columnNameNow_mlt)):
            columnNameNow_mlt[i] = (columnNameNow_mlt[i]).replace("amnt", "amount")

        for i in range(len(columnNameNow_mlt)):
            columnNameNow_mlt[i] = (columnNameNow_mlt[i]).replace("addrzip", "zipcode")

        for i in range(len(columnNameNow_mlt)):
            columnNameNow_mlt[i] = (columnNameNow_mlt[i]).replace("isincv", "verificationstatus")

        df_current.columns = list(columnNameNow_mlt)


        commom_features = pd.read_csv("2015featuresdetermined.txt", header=None).values.reshape(-1)

        df_current_loan = df_current[commom_features]

        empty_columns = ['memberid',
                         'secappopenacc',
                         'revolbaljoint',
                         'secapprevolutil',
                         'secappcollections12mthsexmed',
                         'secappficorangelow',
                         'secappopenactil',
                         'secappmortacc',
                         'secappinqlast6mths',
                         'secappchargeoffwithin12mths',
                         'secappmthssincelastmajorderog',
                         'secappnumrevaccts',
                         'secappearliestcrline',
                         'secappficorangehigh']

        for column in empty_columns:
            df_current_loan.drop(column, axis=1, inplace=True)

        df_current_loan.shape

        df_current_loan["intrate"] = df_current_loan["intrate"] / 100.0

        df_current_loan = df_current_loan[df_current_loan.term == 36]

        df_current_loan.drop("term", axis=1, inplace=True)

        df_current_loan["revolutil"] = df_current_loan["revolutil"].astype("float") / 100.0
        df_current_loan.drop("disbursementmethod", axis=1, inplace=True)

        df_current_loan["desc"].fillna(0, inplace=True)
        
        df_current_loan.loc[df_current_loan["desc"] != 0, "desc"] = 1

        df_current_loan["earliestcrline"] = pd.to_datetime(df_current_loan["earliestcrline"], format = "%Y-%m-%d")
        FMT = '%Y-%m-%d'
        df_current_loan["earliestcrline"] = ((datetime.strptime("2018-12-01", FMT) - df_current_loan["earliestcrline"]) / np.timedelta64(1, 'M')).astype("int")

        df_current_loan['emplength'] = df_current_loan['emplength'] / 12

        df_current_loan["emplength"][df_current_loan['emplength'] >= 10] = 10

        df_current_loan['emplength'].replace('n/a', np.nan, inplace=True)
        df_current_loan.emplength.fillna(value=-999,inplace=True)
        df_current_loan['emplength'] = df_current_loan['emplength'].astype(float)

        df_current_loan['emplength']

        Dic_grade = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

        df_current_loan.grade = df_current_loan.grade.map(Dic_grade)

        df_current_loan.subgrade = df_current_loan.subgrade.apply(lambda x: (Dic_grade[x[0]] - 1) * 5 + int(x[1]))

        df_current_loan.zipcode = df_current_loan.zipcode.apply(lambda x: int(x[0:3]))

        zipcode_freq = pd.read_csv("zipcode_freq.txt", index_col=0)


        zipcode_freq.columns = ["zipcode", "zipcode_freq"]
        df_current_loan=pd.merge(df_current_loan, zipcode_freq, how="left", on="zipcode")

        df_current_loan["emptitle"] = df_current_loan["emptitle"].apply(lambda x: str(x).lower())

        emptitle_freq = pd.read_csv("emptitle_freq.txt", index_col=0)

        df_current_loan["emptitle"] = "8"

        emptitle_freq.columns = ["emptitle", "emptitle_freq"]
        df_current_loan=pd.merge(df_current_loan, emptitle_freq, how="left", on="emptitle")

        dummy_feature = ["homeownership", "verificationstatus", "applicationtype", "purpose", "initialliststatus"]


        training_after_dummy_cols = list(pd.read_csv("featuresafterohcoding.txt", header=None).values.reshape(-1))

        df_current_loan["initialliststatus"] = df_current_loan["initialliststatus"].str.lower()
        df_current_loan["verificationstatus"] = df_current_loan["verificationstatus"].str.replace("_", " ").str.title()
        df_current_loan["applicationtype"] = df_current_loan["applicationtype"].str.title()

        df_dummy = pd.get_dummies(df_current_loan[dummy_feature])

        OHE_feature=list(df_dummy.columns.values)

        df_current_loan=pd.concat([df_current_loan, df_dummy], axis=1)

        df_current_loan.shape

        missing_cols = set(training_after_dummy_cols) - set(df_current_loan.columns)

        for c in missing_cols:
            df_current_loan[c] = 0
        df_current_loan = df_current_loan[training_after_dummy_cols]

        df_current_loan.drop(dummy_feature, axis=1, inplace=True)

        df_current_loan.drop("verificationstatusjoint", inplace=True, axis=1)

        from sklearn.preprocessing import LabelEncoder
        encoder_addrstate = LabelEncoder()
        encoder_addrstate.classes_ = np.load('addrstate_classes.npy')
        encoder_emptitle = LabelEncoder()
        encoder_emptitle.classes_ = np.load('emptitle_classes.npy')

        df_current_loan["addrstate"] = encoder_addrstate.transform(df_current_loan["addrstate"])

        df_current_loan["emptitle"] = encoder_emptitle.transform(df_current_loan["emptitle"])

        df_current_loan.drop("id", inplace=True, axis=1)
        xgbmodel = pickle.load(open("model1207.pkl", 'rb'))
        dtest = xgb.DMatrix(df_current_loan, missing = np.NAN)
        y_pred = xgbmodel.predict(dtest)
        return y_pred

