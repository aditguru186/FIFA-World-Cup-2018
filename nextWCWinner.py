#Predict the next WC winner
import pandas as pd
import nltk
from nltk.classify import apply_features
import numpy as np
import re
import sklearn
from sklearn import preprocessing, svm,neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

DF=pd.read_csv('C:\Python27\ML programs\FIFA\WorldCupMatchesProcessed3.csv',sep=',',usecols=[0,5,6,7,8,2,9,10])
newDF=pd.read_csv('C:\Python27\ML programs\FIFA\FIFAWorldCup2018.csv',sep=',',usecols=[3,4,5])
DF=DF[:835]
newDF=newDF[:48]
#print(DF.head())
results=[]
Team_Name={}# Dictionary for a team num as key to name as value
Team_Num={} # Dictionary for a team name as key to num as value
key=0
X_train=[]
Y_train=[]
Group_A,Group_B,Group_C,Group_D,Group_E,Group_F,Group_G,Group_H=[],[],[],[],[],[],[],[]
Temp_A,Temp_B,Temp_C,Temp_D,Temp_E,Temp_F,Temp_G,Temp_H=[],[],[],[],[],[],[],[]
X_test,Y_score=[],[]
for i in range(835):
    results.append('')
def getLabel(df):
    for i in range(835):
        if(df[i][3]>df[i][4]):
            results[i]=1
        elif(df[i][3]<df[i][4]):
            results[i]=2
        elif(df[i][3]==df[i][4]):
            if(df[i][1]):
                tempo=df[i][1].split()
                if(tempo[0]=='Group' or tempo[0]==''):
                    results[i]=0
                    continue
                temp=df[i][6]
                temp=temp.split(' ')
                #print(temp)
                if(temp[0]==df[i][2]):
                    results[i]=1
                elif(temp[0]==df[i][5]):
                    results[i]=2
    results[639]=2
    results[833]=1
        
                            
def NumberTheTeam(df,key):
    for i in range(835):
        if(not(df[i][2] in Team_Num.keys())):
            Team_Name[key]=df[i][2]
            Team_Num[df[i][2]]=key
            key=key+1
        if(not(df[i][5] in Team_Num.keys())):
            Team_Name[key]=df[i][5]
            Team_Num[df[i][5]]=key
            key=key+1
    #print(key)
    key=key+1
    Team_Num['Korea']=key
    Team_Name[key]='Korea'
    key=key+1
    Team_Num['Costa']=key
    Team_Name[key]='Costa'


def NumberTheTeam1(df,key):
    for i in range(48):
        if(not(df[i][0] in Team_Num.keys())):
            Team_Name[key]=df[i][0]
            Team_Num[df[i][0]]=key
            key=key+1 
        if(not(df[i][1] in Team_Num.keys())):
            Team_Name[key]=df[i][1]
            Team_Num[df[i][1]]=key
            key=key+1
    Team_Num['Draw']=-1
    Team_Name[-1]='Draw'
    
    

def MakeTrainingSet(df):
    for i in range(835):
        X_train.append([Team_Num[df[i][2]],Team_Num[df[i][5]]])
        Y_train.append(results[i])

def MakeTestSet(df):
    for i in range(48):
        X_test.append([Team_Num[df[i][0]],Team_Num[df[i][1]]])
      
def ShowResults(Y_predict,df12):
    print("The Group Stage result of the Upcoming 2018 FIFA World Cup :-")
    for i in range(48):
        temp=""
        if(Y_predict[i]==0):
            temp=df12[i][0]+" vs "+df12[i][1]+" result :- DRAW"
        else:
            temp=df12[i][0]+" vs "+df12[i][1]+" result :- "+ df12[i][Y_predict[i]-1]
        print(temp)
            
def NumberTheMatches(df1):
    for i in range(48):
        if(df1[i][2]=='Group A'):
            Temp_A.append(i)
        elif(df1[i][2]=='Group B'):
            Temp_B.append(i)
        elif(df1[i][2]=='Group C'):
            Temp_C.append(i)
        elif(df1[i][2]=='Group D'):
            Temp_D.append(i)
        elif(df1[i][2]=='Group E'):
            Temp_E.append(i)
        elif(df1[i][2]=='Group F'):
            Temp_F.append(i)
        elif(df1[i][2]=='Group G'):
            Temp_G.append(i)
        elif(df1[i][2]=='Group H'):
            Temp_H.append(i)

def UpdateGroupTable(df,Group_Name,Temp_Name,clf):
    for mid in Temp_Name:
        temporary=[[Team_Num[df[mid][0]],Team_Num[df[mid][1]]]]
        #print(temporary)
        temporary_result=clf.predict(temporary)
        #print(temporary_result)
        result=temporary_result[0]
        #X_train.append([Team_Num[df[mid][0]],Team_Num[df[mid][1]]])
        #Y_train.append(result)
        #print(result)
        for team in Group_Name:
            if(team[0]==df[mid][0]):
                if(result==1):
                    team[1]+=3
                    break
                if(result==0):
                    team[1]+=1
            if(team[0]==df[mid][1]):
                if(result==2):
                    team[1]+=3
                    break
                if(result==0):
                    team[1]+=1

def SortGroupStageResults(Groups):
    for g in Groups:
        t_t,t_p='',0
        for i in range(4):
            for j in range(4-i-1):
                if(g[j][1]<g[j+1][1]):
                    t_t=g[j][0]
                    g[j][0]=g[j+1][0]
                    g[j+1][0]=t_t
                    t_p=g[j][1]
                    g[j][1]=g[j+1][1]
                    g[j+1][1]=t_p
        #print(g)
    

def GroupStageMatches(df,Groups):
    print("Using K nearest Neighbors Classifier for training the Datas:\n")
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,Y_train)
    UpdateGroupTable(df,Group_A,Temp_A,clf)
    UpdateGroupTable(df,Group_B,Temp_B,clf)
    UpdateGroupTable(df,Group_C,Temp_C,clf)
    UpdateGroupTable(df,Group_D,Temp_D,clf)
    UpdateGroupTable(df,Group_E,Temp_E,clf)
    UpdateGroupTable(df,Group_F,Temp_F,clf)
    UpdateGroupTable(df,Group_G,Temp_G,clf)
    UpdateGroupTable(df,Group_H,Temp_H,clf)
    SortGroupStageResults(Groups)
    t=Group_A[0][0];Group_A[0][0]=Group_A[2][0];Group_A[2][0]=t
    print('Group A result:-',Group_A)
    print('Group B result:-',Group_B)
    print('Group C result:-',Group_C)
    print('Group D result:-',Group_D)
    print('Group E result:-',Group_E)
    print('Group F result:-',Group_F)
    print('Group G result:-',Group_G)
    print('Group H result:-',Group_H)
    
def KnockOutStage():
    #clf_KNN=neighbors.KNeighborsClassifier()
    #clf_KNN=tree.DecisionTreeClassifier()
    #clf_KNN=RandomForestClassifier()
    clf_SVM=svm.SVC()
    clf_KNN=svm.SVC()
    print('\nUsing Support Vector Classifier for the Knock out games prediction\nThe Knock Out Draw is following:-\n')
    clf_KNN.fit(X_train,Y_train)
    clf_SVM.fit(X_train,Y_train)
    result_QF=[]
    #Round Of 16
    for i in range(8):
        temp=[]
        if(i==0):
            temp=[[Team_Num[Group_A[0][0]],Team_Num[Group_B[1][0]]]]
        elif(i==1):
            temp=[[Team_Num[Group_C[0][0]],Team_Num[Group_D[1][0]]]]
        elif(i==2):
            temp=[[Team_Num[Group_E[0][0]],Team_Num[Group_F[1][0]]]]
        elif(i==3):
            temp=[[Team_Num[Group_G[0][0]],Team_Num[Group_H[1][0]]]]
        elif(i==4):
            temp=[[Team_Num[Group_B[0][0]],Team_Num[Group_A[1][0]]]]
        elif(i==5):
            temp=[[Team_Num[Group_D[0][0]],Team_Num[Group_C[1][0]]]]
        elif(i==6):
            temp=[[Team_Num[Group_F[0][0]],Team_Num[Group_E[1][0]]]]
        elif(i==7):
            temp=[[Team_Num[Group_H[0][0]],Team_Num[Group_G[1][0]]]]
        
        match1=clf_KNN.predict(temp)
        if(match1==0):
            match1=clf_SVM.predict(temp)
            #print('Draw')
        if(match1==1):
            result_QF.append(Team_Name[temp[0][0]])
        elif(match1==2):
            result_QF.append(Team_Name[temp[0][1]])
            
    #Quarter Finals
    result_SF=[]
    for i in range(4):
        temp=[]
        if(i==0):
            temp=[[Team_Num[result_QF[0]],Team_Num[result_QF[1]]]]
        elif(i==1):
            temp=[[Team_Num[result_QF[2]],Team_Num[result_QF[3]]]]
        elif(i==2):
            temp=[[Team_Num[result_QF[4]],Team_Num[result_QF[5]]]]
        elif(i==3):
            temp=[[Team_Num[result_QF[6]],Team_Num[result_QF[7]]]]
        
        match1=clf_KNN.predict(temp)
        if(match1==0):
            match1=clf_SVM.predict(temp)
            #print('Draw')
        if(match1==1):
            result_SF.append(Team_Name[temp[0][0]])
        elif(match1==2):
            result_SF.append(Team_Name[temp[0][1]])
    #Semi Finals
    result_F=[]
    match1=clf_KNN.predict([[Team_Num[result_SF[0]],Team_Num[result_SF[1]]]])
    if(match1==0):
        match1=clf_KNN.predict([[Team_Num[result_SF[0]],Team_Num[result_SF[1]]]])
    if(match1==1):
        result_F.append(Team_Name[Team_Num[result_SF[0]]])
    elif(match1==2):
        result_F.append(Team_Name[Team_Num[result_SF[1]]])
    
    match1=clf_KNN.predict([[Team_Num[result_SF[2]],Team_Num[result_SF[3]]]])
    if(match1==0):
        match1=clf_KNN.predict([[Team_Num[result_SF[2]],Team_Num[result_SF[3]]]])
    if(match1==1):
        result_F.append(Team_Name[Team_Num[result_SF[2]]])
    elif(match1==2):
        result_F.append(Team_Name[Team_Num[result_SF[3]]])
    result_Final=''
    
    #Finals
    match1=clf_KNN.predict([[Team_Num[result_F[0]],Team_Num[result_F[1]]]])
    if(match1==0):
        match1=clf_SVM.predict([[Team_Num[result_SF[0]],Team_Num[result_SF[1]]]])
    if(match1==1):
        result_Final=Team_Name[Team_Num[result_SF[0]]]
    elif(match1==2):
        result_Final=Team_Name[Team_Num[result_SF[1]]]
    
    print(Group_A[0][0]+'\n\tvs\t'+result_QF[0]+'\n'+Group_B[1][0]+'\n\t\t\tvs\t'+result_SF[0])
    print(Group_C[0][0]+'\n\tvs\t'+result_QF[1]+'\n'+Group_D[1][0]+'\n\t\t\t\t\tvs\t'+result_F[0])
    print(Group_E[0][0]+'\n\tvs\t'+result_QF[2]+'\n'+Group_F[1][0]+'\n\t\t\tvs\t'+result_SF[1])
    print(Group_G[0][0]+'\n\tvs\t'+result_QF[3]+'\n'+Group_H[1][0]+'\n\t\t\t\t\t\t\tvs\t'+result_Final)
    print(Group_B[0][0]+'\n\tvs\t'+result_QF[4]+'\n'+Group_A[1][0]+'\n\t\t\tvs\t'+result_SF[2])
    print(Group_D[0][0]+'\n\tvs\t'+result_QF[5]+'\n'+Group_C[1][0]+'\n\t\t\t\t\tvs\t'+result_F[1])
    print(Group_F[0][0]+'\n\tvs\t'+result_QF[6]+'\n'+Group_E[1][0]+'\n\t\t\tvs\t'+result_SF[3])
    print(Group_H[0][0]+'\n\tvs\t'+result_QF[7]+'\n'+Group_G[1][0]+'\n')
    

df=np.array(DF)
getLabel(df)
#print(results)
key=0
NumberTheTeam(df,key)
df1=np.array(newDF)
NumberTheTeam1(df1,key)
MakeTrainingSet(df)
MakeTestSet(df1)
#print(Y_train)
#print(Team_Num)
#print(Team_Name)
Group_A=[['Russia',0],['Saudi Arabia',0],['Egypt',0],['Uruguay',0]]
Group_B=[['Portugal',0],['Spain',0],['Morocco',0],['Iran',0]]
Group_C=[['France',0],['Australia',0],['Peru',0],['Denmark',0]]
Group_D=[['Argentina',0],['Croatia',0],['Iceland',0],['Nigeria',0]]
Group_E=[['Brazil',0],['Switzerland',0],['Costa Rica',0],['Serbia',0]]
Group_F=[['Germany',0],['Mexico',0],['Sweden',0],['Korea Republic',0]]
Group_G=[['Belgium',0],['Panama',0],['Tunisia',0],['England',0]]
Group_H=[['Poland',0],['Senegal',0],['Colombia',0],['Japan',0]]
NumberTheMatches(df1)
Groups=[Group_A,Group_B,Group_C,Group_D,Group_E,Group_F,Group_G,Group_H]
GroupStageMatches(df1,Groups)
x,x_1=['A','B','C','D','E','F','G','H'],0
print('Teams Qualifying For Round-of-16 :\n')
for i in Groups:#displays the round of 16 line ups:-
    t='Group '+x[x_1]
    print(t)
    print('1-'+i[0][0])
    print('2-'+i[1][0]+'\n')
    x_1+=1
KnockOutStage()

'''#Using SVM model for prediction
clf=svm.SVC()
clf.fit(X_train,Y_train)
Y_score=clf.predict(X_test)
#print('Using SVM,we got the following results',Y_score)
#ShowResults(Y_score,df1)
#Shows a bit inaccurate results

#Using KNN model for prediction
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)
Y_score=clf.predict(X_test)
#print('Using KNN,we got the following results',Y_score)
#ShowResults(Y_score,df1)
#Still this is inaccurate, and false result as it predicts the team not playing to have won in a match! Thats not possible

#using NB Classifier
gnb=GaussianNB()
gnb.fit(X_train,Y_train)
Y_score=gnb.predict(X_test)
#print('Using NB Classifier,we got the following results',Y_score)
#ShowResults(Y_score,df1)
#This works By showing no draw results!!

clf=tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
Y_score=clf.predict(X_test)
#print('Using Decision Tree,we got the following results',Y_score)
#ShowResults(Y_score,df1)

clf=RandomForestClassifier()
clf.fit(X_train,Y_train)
Y_score=clf.predict(X_test)
print('Using Random Forest,we got the following results',Y_score)
ShowResults(Y_score,df1)
'''