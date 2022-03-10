class Classification():
    from flask import Flask, render_template, request 
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # Import Libraries
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
    
    def __init__(self):
        pass
    
    dataDir ='../mCardiac_flask/'
    
      
    def train_algorithm(self):  
        Train = self.pd.read_csv(self.dataDir+'TrainData.csv')
        #Import Dataset
        X = Train.iloc[:,:28].values
        Y = Train.iloc[:,28].values
        seed =0
        # Split Dataset into trianing and testing
       # X_train, X_test, y_train, y_test = self.train_test_split(X, Y, test_size=0.30, random_state=seed)

        clf =self.RandomForestClassifier(n_estimators=100, random_state=seed)
        model = clf.fit(X,Y)
    
        # Save model as pickle file
        return self.joblib.dump(model, self.dataDir+"model.pkl")
    
    def classify(self):
        # Load model from file
        classifer=self.joblib.load(self.dataDir+"model.pkl")
        Test = self.pd.read_csv(self.dataDir+'TestData.csv')
        X_test = Test.iloc[:,1:29].values
        y_pred = classifer.predict(X_test)
        Test['Activity'] = y_pred
        Test = Test[['Activity', 'timestamp' ]]
        return Test.to_csv(self.dataDir+'ActivityData.csv')
    
    def view_activity(self):
        import os
        if os.path.exists(self.dataDir+'static/images/activity.png'):
            os.remove(self.dataDir+'static/images/activity.png')
        spacing = 2
        dataset = self.pd.read_csv(self.dataDir+'ActivityData.csv')
        #ax = self.plt.axis
        #dataset.set_index("timestamp")
        #ax =self.plt.scatter(dataset.index.values, dataset["Activity"], marker='o')
        #ax =self.sns.relplot(x='timestamp', y="Activity", hue='Activity', data=dataset, legend=False)
        ax= self.sns.scatterplot(x='timestamp', y='Activity', hue='Activity', data=dataset, legend=False)
        #self.plt.xticks(self.np.arange(min(dataset['timestamp']), max(dataset['timestamp'])+1, 1.0))
        #ax.set_xticks(dataset['timestamp'])
        ax.xaxis.set_minor_formatter(self.mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        ax.xaxis.set_minor_locator(self.mdates.SecondLocator())
        for label in ax.xaxis.get_ticklabels()[::spacing]:
            label.set_visible(False)
        self.plt.gcf().autofmt_xdate()
        #self.plt.savefig(self.dataDir+'static/images/activity.png')
        return self.plt.savefig(self.dataDir+'static/images/activity.png')
        
    def search_activity(self,start_date,end_date):
        dataset = self.pd.read_csv(self.dataDir+'ActivityData.csv')
        # Set index
        #df = df.set_index(df['date'])
        # Select observations between two datetimes
        #df.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']
       # df['timestamp'] = self.pd.to_datetime(df['timestamp'])
        #start_date = '2020-02-25 14:08:20'
        #end_date= '2020-02-25 15:01:00'
       # mask = (df['timestamp'] > start_date) & (df['timestamp'] <= end_date)
        df = dataset[(dataset['timestamp'] >start_date) & (dataset['timestamp'] <= end_date)]
        ax = self.plt.axes()
        ax =self.sns.scatterplot(x='timestamp', y='Activity', hue='Activity', data=df, legend=False)
        #plt.xticks(np.arange(min(dataset['curdate']), max(dataset['curdate'])+1, 1.0))
        #ax.set_xticks(df['timestamp'])
        ax.xaxis.set_minor_formatter(self.mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        ax.xaxis.set_minor_locator(self.mdates.MinuteLocator())
        self.plt.gcf().autofmt_xdate()
        return self.plt.savefig(self.dataDir+'static/images/activity.png')   
    