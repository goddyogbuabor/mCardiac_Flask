class Feature_extraction():
    # Import Libraries
    from flask import Flask, render_template, request
    import pandas as pd
    import numpy as np
    import math
    import os
    import glob
    import sys
    #import dataset
    dataDir = '../mCardiac_flask/'
               
    COLUMNS = ['activity', 'position', 'xAxis','yAxis', 'zAxis', 'timestamp', 'time']   
           
    def __init__(self):
        pass
    
    def join_train_data(self, indir=os.path.abspath('..\\mCardiac_flask\\static\\TrainData'), outfile=os.path.abspath('..\\mCardiac_flask\\JoinActivity.csv')):
           
        self.os.chdir(indir)
        extension = 'csv'
        all_filenames = [i for i in self.glob.glob('*.{}'.format(extension))]
        #combine all files in the list
        combined_csv = self.pd.concat([self.pd.read_csv(f) for f in all_filenames ])
        #export to csv
        return combined_csv.to_csv(outfile, index=False, encoding='utf-8-sig') 
    
    def join_classify_data(self, indir=os.path.abspath('..\\mCardiac_flask\\static\\ClassifyData'), outfile=os.path.abspath('..\\mCardiac_flask\\TestActivity.csv')):
           
        self.os.chdir(indir)
        extension = 'csv'
        all_filenames = [i for i in self.glob.glob('*.{}'.format(extension))]
        #combine all files in the list
        combined_csv = self.pd.concat([self.pd.read_csv(f) for f in all_filenames ])
        #export to csv
        return combined_csv.to_csv(outfile, index=False, encoding='utf-8-sig') 
    
    # calculate magnitude
    def magnitude(self,axis):
        import math
        x2 = axis['xAxis'] * axis['xAxis']
        y2 = axis['yAxis'] * axis['yAxis']
        z2 = axis['zAxis'] * axis['zAxis']
        m2 = x2 + y2 + z2
        m = m2.apply(lambda x: math.sqrt(x))
        return m
        
    # Mean value
    def mean_val(self,df):
        return df.resample('4S').mean()

    # Standard Deviation value
    def std_val(self,df):
        return df.resample('4S').std()

    # Minimum Value
    def min_val(self,df):
        return df.resample('4S').min()

    # Maximum Value
    def max_val(self,df):
        return df.resample('4S').max()
    # variance Value
    def var_val(self,df):
        return df.resample('4S').var()
        
    # standard error of the mean of groups
    def sem_val(self,df):
        return df.resample('4S').sem()
    # median Value
    def median_val(self,df):
        return df.resample('4S').median()
    # get labels
    def label_val(self,df):
        return df.resample('4S').min()

    # Extract Feature for Algorithm Training
    def train_feature(self):
        COLUMNS = ['activity', 'position', 'xAxis','yAxis', 'zAxis', 'timestamp', 'time']   
        dataDir2 = '../mCardiac_flask/JoinActivity.csv'
        data = self.pd.read_csv(dataDir2, header=None, names=COLUMNS, skiprows=[0], index_col='timestamp', parse_dates=True)
     # convert timestamp to datetime
        data.index = self.pd.to_datetime(data.index/1000, unit='s')
        data['mag'] = self.magnitude(data)
        # get class label for each activity
        CLASS = data['activity']
        train = data[['xAxis', 'yAxis', 'zAxis', 'mag']]
        features = self.pd.concat([self.mean_val(train), self.std_val(train), self.min_val(train), self.max_val(train), self.var_val(train),
                            self.sem_val(train), self.median_val(train), self.label_val(CLASS)], axis=1)
        # drop nan values
        features.dropna(inplace=True)
       
        columns = ['xmean', 'ymean', 'zmean', 'mmean', 'xstd', 'ystd', 'zstd', 'mstd', 'xmin', 'ymin', 'zmin', 'mmin',
                    'xmax', 'ymax', 'zmax', 'mmax', 'xvar', 'yvar', 'zvar', 'mvar', 'xsem', 'ysem', 'zsem', 'msem',
                    'xmedian', 'ymedian', 'zmedian', 'mmedian', "Activity"]

        #features = train
        features.columns = columns
        return features.to_csv(self.dataDir+'TrainData.csv', index=None)

    # Extract Feature for Activity Classification
    def Classify_feature(self):
        classifydata = self.pd.read_csv(self.dataDir+'TestActivity.csv', skiprows=[0], header=None, names=self.COLUMNS,
                        index_col='timestamp', parse_dates=True)
        # convert timestamp to datetime
        classifydata.index = self.pd.to_datetime(classifydata.index, unit='s')
        
        classifydata['mag'] = self.magnitude(classifydata)
        dataset = classifydata[['xAxis', 'yAxis', 'zAxis', 'mag']]

        classifyfeatures = self.pd.concat([self.mean_val(dataset), self.std_val(dataset), self.min_val(dataset), self.max_val(dataset),
                                            self.var_val(dataset), self.sem_val(dataset), self.median_val(dataset)], axis=1)
        # drop nan values
        classifyfeatures.dropna(inplace=True)
        columns = ['xmean', 'ymean', 'zmean', 'mmean', 'xstd', 'ystd', 'zstd', 'mstd', 'xmin', 'ymin', 'zmin', 'mmin',
                    'xmax', 'ymax', 'zmax', 'mmax', 'xvar', 'yvar', 'zvar', 'mvar', 'xsem', 'ysem', 'zsem', 'msem',
                    'xmedian', 'ymedian', 'zmedian', 'mmedian']
               
        classifyfeatures.columns = columns
        return classifyfeatures.to_csv(self.dataDir+'TestData.csv')

    