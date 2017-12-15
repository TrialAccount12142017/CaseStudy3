# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import csv 
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from preprocess import clean_DF, sales_POS_exist_Within, extr_city, build_feature_DF

class POSclassifier:
    
    def __init__(self,
                 salesfile,
                 tfrom,
                 tto,
                 surroundingfile,
                 bigcities,
                 modelinputfile):
        
        self.salesfile = salesfile
        self.surroundingfile = surroundingfile
        self.modelinputfile = modelinputfile
        self.tfrom = tfrom
        self.tto = tto
        self.bigcities = bigcities

    def PrepareFeatureNLabel(self):
        print "--------start reading sales data--------"         
        df_sales_raw = pd.read_csv(self.salesfile, delimiter = ",", dtype = str, header=None).T        
        df_sales = clean_DF(df_sales_raw)
        print str(len(df_sales.columns))+" POS have sales data"
        
        df_sales_label = sales_POS_exist_Within(df_sales, timefrom=self.tfrom, timeto=self.tto)        
        df_sales_label_M = df_sales_label['2017-01':'2017-06'].sum()
        print str(len(df_sales_label_M)) + " POS satisfy this time period"
        

        print "--------start reading surroundings--------"
        
        df_org = pd.read_json(self.surroundingfile,encoding='UTF-8')
        print str(len(df_org))+" POS have surrounding data"
        
        allCities = extr_city(df_org)
        print "POS locate in "+ str(len(list(set(allCities))))+" different cities in Switzerland"
        print "List of cities with population more than 100,000 are : "+str(self.bigcities)
        #addr =  list(set(allCities))
        found = [c for c in allCities if c in self.bigcities]
        print "there are "+str(set(Counter(found).values()))+" POS"
        print "locate in "+str(set(Counter(found).keys()))+" respectively"
        
        df_sur = build_feature_DF(df_org, self.bigcities)

        print "-------- merging into model input dataset --------"
        df_sales_final =pd.DataFrame({'store_code':df_sales_label_M.index,
                                      'Ysales':df_sales_label_M.values})
        df_final = pd.merge(df_sur, df_sales_final, how='inner', on='store_code')
        #----create classes(label as 0 or 1)----
        classes = []
        for i in df_final['Ysales']:
            if i>=20000:
                classes.append(1)
            else: 
                classes.append(0)

        df_final['label'] = classes
        self.df_final = df_final.drop('Ysales', axis=1)
        print str(len(self.df_final))+" POS in final dataset."
        print self.df_final.head()
        self.df_final.to_csv(self.modelinputfile, sep=',',encoding='UTF-8')
        print "Model input data were written to "+str(self.modelinputfile)
        print self.df_final.info()

    def XgboostingClassifier(self,
                             max_depth= [3,5,7],
                             min_child_weight = [1,3,5],
                             learning_rate = [0.1, 0.01],
                             subsample = [0.7,0.8,0.9],
                             #max_depth= [7],
                             #min_child_weight = [1],
                             #learning_rate = [0.1],
                             #subsample = [0.8],
                             fromfile = 1):
        # read data, split into training and testing
        print "--- reading data ---"
        if fromfile == 1:
            print "reading from "+ self.modelinputfile
            df_raw = pd.read_csv(self.modelinputfile,encoding='UTF-8')
            df_clean = df_raw[df_raw.columns[1:]]
        else:
            df_clean = self.df_final.astype(float)
        print df_clean.head()
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            df_clean[df_clean.columns[:-1]].as_matrix(), 
            list(df_clean[df_clean.columns[-1]]), test_size=0.2, random_state=42)
        print "training and testing data sets prepared."
        print X_train, X_test, y_train, y_test
        df_train = pd.DataFrame(X_train, columns=df_clean.columns[:-1])
        df_test = pd.DataFrame(X_test, columns=df_clean.columns[:-1])
        
        # CV to find best parameters 
        cv_params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 
                     'learning_rate': learning_rate, 'subsample': subsample}

        ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
                     'objective': 'binary:logistic', 'max_depth': 7, 'min_child_weight': 1}

        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                             cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
        optimized_GBM.fit(df_train[df_train.columns[1:]], y_train)

        bestparameterset = optimized_GBM.best_estimator_.get_xgb_params()
        print "CV best score: "+str(optimized_GBM.best_score_)
        print "best parameter set: \n"+str(bestparameterset)

        # train and output features importance
        xgdmat = xgb.DMatrix(df_train[df_train.columns[1:]], y_train)
        final_gb = xgb.train(bestparameterset, xgdmat, num_boost_round = 500) # arbitrarily

        importances = final_gb.get_fscore()
        print "feature importances: "+str(importances)
        importance_frame = pd.DataFrame({'Importance': list(importances.values()), 
                                         'Feature': list(importances.keys())})
        importance_frame.sort_values(by = 'Importance', inplace = True)
        importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,40), color = 'blue')
        plt.savefig('FeatureImportance.eps', bbox_inches='tight',format='eps', dpi=1000)

        # test and output result
        testdmat = xgb.DMatrix(df_test[df_test.columns[1:]], y_test)
        from sklearn.metrics import accuracy_score
        y_pred = final_gb.predict(testdmat) # Predict using our testdmat
        self.y_prob = y_pred.copy()
        df_prob2code =pd.DataFrame({'probability': self.y_prob, 'store_code':df_test[df_test.columns[0]]})
        df_prob2code.to_csv('predprob_2_code.csv', sep=',',encoding='UTF-8')
        y_pred = [round(value) for value in y_pred]
        print "accuracy: "+ str(accuracy_score(y_pred, y_test))
        print "classification report: "
        print classification_report(y_test, y_pred, target_names=['low-sales POS','high-sale POS'])
        self.y_test = y_test

    def GroundTruthDistrInPredGrids(self, distrfile):
        #true_low_count = [61.0,7.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        #true_high_count = [14.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0]
        true_low_count, true_high_count = self.__ground_truth()
        fit = plt.figure(figsize=(6,5))
        n_groups = len(true_low_count)
        index = np.arange(n_groups)
        print index
        bar_width = 0.4
        opacity = 0.4
        rects1 = plt.bar(index, true_low_count, bar_width,
                         alpha=opacity,
                         color='b')
        
        rects2 = plt.bar(index + bar_width, true_high_count, bar_width,
                         alpha=opacity,
                         color='g')
        plt.xlabel('Predicted POS high-sales probability')
        plt.ylabel('#POS')
        plt.legend((rects1[0], rects2[0]), ('True low sales POS', 'True high sales POS'))
        plt.xticks(index + bar_width/2, ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
        
        #plt.tight_layout()
        plt.savefig(distrfile, bbox_inches='tight',format='eps', dpi=1000)

    def __ground_truth(self):
        proba = self.y_prob
        truth = list(self.y_test)
        true_high_count = []
        true_low_count = []
        for a in range(10):
            tmp = 0.0
            sale_truth = 0.0
            for i in range(len(proba)):
                if proba[i]>=0.1*a and proba[i]<0.1*(a+1):
                    tmp += 1.0
                    if truth[i] == 1.0: sale_truth += 1.0
            true_low_count.append(tmp-sale_truth)
            true_high_count.append(sale_truth)
        print true_low_count, true_high_count
        return true_low_count, true_high_count

if __name__=='__main__':
    model = POSclassifier(
        salesfile = 'sales_granular.csv',
        surroundingfile = 'Surroundings.json',
        tfrom = '2017-01',
        tto = '2017-06',
        bigcities = [u'Basel', u'Bern', u'Lausanne', u'Winterthur', u'Z\xfcrich'],
        modelinputfile = 'clas_model_input_halfY.csv'
        )
    model.PrepareFeatureNLabel()
    model.XgboostingClassifier(fromfile = 0)
    model.GroundTruthDistrInPredGrids(distrfile = 'TruePOSDistrubutionInProbGrids.eps')
