# -*- coding: utf-8 -*-
import pandas as pd
import datetime
from collections import defaultdict
from collections import OrderedDict

# read sales
def clean_DF(df_raw):
    header = [str(x) for x in df_raw.iloc[0]]
    df_raw = df_raw[1:]
    
    df_raw.columns = header
    col2date = header[0]
    df_raw = df_raw.T.drop_duplicates().T
    df_raw.index = pd.to_datetime(df_raw[col2date])
    df_raw = df_raw.drop(col2date, axis=1)
    df_raw = df_raw.astype(float)
    df_raw = df_raw.fillna(0)
    return df_raw


def sales_POS_exist_Within(df_input, **timelmt):
    df_full = df_input.copy()
    print "Extracting sales data "+ str(timelmt)
    if 'timefrom' in timelmt.keys():
        checkend = pd.to_datetime(timelmt['timefrom']) + pd.DateOffset(months=-1)
        df_sales_before = df_full[:timelmt['timefrom']].sum()
        df_keep_b = df_sales_before[df_sales_before!=0]
        df_full = df_full[df_keep_b.index]
        
    if 'timeto' in timelmt.keys():
        #"-".join(str(at).split()[0].split("-")[:2])
        checkend = pd.to_datetime(timelmt['timeto']) + pd.DateOffset(months=-1)
        checkt = str(checkend).split()[0]#; print checkt
        df_sales_after = df_full[checkt:].sum()
        df_keep_a = df_sales_after[df_sales_after!=0]
        df_full = df_full[df_keep_a.index]
    return df_full


#  read sourroundings

def extr_city(df):
    allcity = []
    amni = df.surroundings[0].keys()
    for index, row in df.iterrows():
        checkC = 0
        for a in amni:
            lst = row['surroundings'][a]
            if len(lst)>0:
                for substore in lst:
                    if ('formatted_address' in substore.keys()) and checkC == 0:
                        allcity.extend(substore['formatted_address'].split(',')[-2].split()[1:])
                        checkC += 1
                        break
    return allcity


def build_feature_DF(df_srd, bigcities):
    
    amni = df_srd.surroundings[0].keys()
    col = ['store_code']+bigcities

    for a in amni:
        col.extend([a+"_count", a+"_rating_avg", a+"_rating_nsum"])
        
    amenities = {}
    for c in col:
        amenities[c] = []
    
    for index, row in df_srd.iterrows():
        amenities['store_code'].append(str(row['store_code']))
        checkC = 0
        for a in amni:
            lst = row['surroundings'][a]
            amenities[a+"_count"].append(len(lst))
            Rating = 0
            Nrater = 0
            if len(lst)>0:
                rating = []
                nrater = []
                for substore in lst:
                    if ('formatted_address' in substore.keys()) and checkC == 0:
                        try:
                            poscity = substore['formatted_address'].split(',')[-2].split()[1:][0]
                        except IndexError:
                            print poscity
                        checkC += 1
                        for atcity in bigcities:
                            if poscity == atcity:
                                amenities[atcity].append(1)
                            else: 
                                amenities[atcity].append(0)
                    if ('rating' in substore.keys()) and ('user_ratings_total' in substore.keys()):
                        rating.append(substore['rating'])
                        nrater.append(substore['user_ratings_total'])
                if len(rating)>0:
                    Rating = sum(rating)/len(rating)
                    Nrater = sum(nrater)
            amenities[a+"_rating_avg"].append(Rating)
            amenities[a+"_rating_nsum"].append(Nrater)
        if checkC == 0:
            for atcity in bigcities:
                amenities[atcity].append(0)
    return pd.DataFrame(amenities, columns=col)

