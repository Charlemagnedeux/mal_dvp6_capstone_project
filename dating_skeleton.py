
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# In[2]:


#Create your df here:
df = pd.read_csv("profiles.csv")


# In[3]:


df.income.value_counts()


# In[4]:


df.job.value_counts()


# In[5]:


df.status.value_counts()


# In[6]:


df.essay3.head()


# In[7]:


df.orientation.value_counts()


# 1) One graph containing exploration of the dataset

# In[8]:


sns.set(style="darkgrid")
sns.countplot(x="job", data=df)
plt.xticks(rotation=90)
plt.show()


# 2) Second graph containing exploration of the dataset

# In[9]:


sns.set(style="darkgrid")
sns.countplot(x="orientation", data=df)
plt.xticks(rotation=90)
plt.show()


# 3) Formulate a question!
# 
# I'm asking myself the question if there's a correlation between certain data types which predicts the sexual orientation. As a gay guy I want to know in which environment I'll find my future husband or vice versa as a lesbian woman. Should I go for artistic jobs with a certain income or is military better? Let's check it out with classification.
# 
# For regression I'm interested in the money: Does writing makes you rich? I want to count all the words of the essays and plot it against the income

# 4) Augment your data

# 4.1) Alter Income

# In[10]:


income_mapping = {-1: 1, 20000: 2, 30000: 3, 40000: 4, 50000: 5, 60000: 6, 70000: 7, 100000: 8, 150000: 9, 250000: 10, 500000: 11, 1000000: 12}
# NaN cannot be solved by mapping here, don't know why?
temp = df.income.map(income_mapping)
df["income_code"] = temp.where(pd.notnull(temp), 0)
df.income_code.value_counts()


# 4.2) Alter Job

# In[11]:


job_mapping = {np.nan: 0, "other": 1, "student": 2, "science / tech / engineering": 3, "computer / hardware / software": 4, "artistic / musical / writer": 5, "sales / marketing / biz dev": 6, "medicine / health": 7, "education / academia": 8, "executive / management": 9, "banking / financial / real estate": 10, "entertainment / media": 11, "law / legal services": 12, "hospitality / travel": 13, "construction / craftsmanship": 14, "clerical / administrative": 15, "political / government": 16, "rather not say": 17, "transportation": 18, "unemployed": 19, "retired": 20, "military": 21}
df["job_code"] = df.job.map(job_mapping)
df.job_code.value_counts()


# 4.3) Alter Status

# In[12]:


status_mapping = {"single": 0, "seeing someone": 1, "available": 2, "married": 3, "unknown": 4}
df["status_code"] = df.status.map(status_mapping)
df.status_code.value_counts()


# 4.4) Alter Essay3

# In[13]:


df["essay3_code"] = df.essay3.map(lambda x: 0 if pd.isnull(x) else (1 if 'smile' in x or 'laugh' in x else 2))
df.essay3_code.value_counts()


# 4.5) Alter Orientation

# In[14]:


orientation_mapping = {"straight": 0, "gay": 1, "bisexual": 2}
df["orientation_code"] = df.orientation.map(orientation_mapping)
df.orientation_code.value_counts()


# 4.6) Count words in essay

# In[15]:


#Preparation
def return_word_len(text):
    value_as_string = str(text)
    
    for char in '-.,\n':
        value_as_string = value_as_string.replace(char,' ')
    value_as_string = value_as_string.lower()
    word_list = value_as_string.split()
    length = len(word_list)
    return length

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df['essay_count_words'] = df.apply(lambda row: return_word_len(row), axis=1)


# 5) Normalize data

# In[16]:


feature_data = df[["income_code", "job_code", "status_code", "essay3_code"]]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


# 6.1) Classification - KNearest Neighbor 

# In[17]:


training_data, validation_data, training_labels, validation_labels = train_test_split(feature_data, df.orientation_code, test_size = 0.2, random_state = 100)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data, training_labels)


# In[18]:


print(classifier.score(validation_data, validation_labels))


# 6.2) Predict the orientation with normalized data

# In[19]:


# 1 High Income, Artist, Available, Likes to smile 2 Average Income, Academia, seeing someone, likes somethin at least, 3 Low Income, Military, married, nothing particular about me 
predict_data = np.array([[0.916667,0.238095,0.50,0.5], [0.583333,0.380952,0.25,1.0], [0.333333,1.000000,0.75,0.0]])
print(predict_data)


# In[20]:


print(classifier.predict(predict_data))


# Well it predicts, that every person is straight. This seems not so far fetched due to the high amount of straight people in OKCupid. Gay people can be found at Grinder :-)

# 6.3) Accuracy of Nearest Neighbors

# In[21]:


accuracies = []
for k in range(1,20):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data,training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))
k_list = list(range(1,20))
plt.plot(k_list,accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Show Orientation")
plt.show()


# 6.3) Classification - SVM

# In[22]:


classifier =()
classifier = SVC(kernel = "rbf", gamma =0.1, C=2)
classifier.fit(training_data, training_labels)


# 6.4) Predict the orientation

# In[23]:


print(classifier.score(validation_data, validation_labels))
print(classifier.predict(predict_data))


# Same result as with the Nearest Neighbor approach, but a tiny better result of two percent.

# 6.5) Accuracy of SVM

# In[ ]:


accuracies = []
accuracy = 0
highest_accuracy = 0
highest_g = 0
highest_c = 0

for c in range(1,10):
    g = 0.1
    while g < 1.1:
        classifier = SVC(kernel = "rbf", gamma = g, C=c)
        classifier.fit(training_data, training_labels)
        accuracy = classifier.score(validation_data, validation_labels)
        accuracies.append(accuracy)
        print(accuracy)
        g += 0.1
        if highest_accuracy > accuracy:
            highest_g = g
            highest_c = c
            highest_accuracy = accuracy

print("Best Accuracy is: " + str(highest_accuracy) + " with g =" + str(highest_g) +" and c = " + str(highest_c))


# 7.1) Regression - Linear Regression

# In[23]:


income =df['income'].values
income = income.reshape(-1, 1)
essay_word_len = df['essay_count_words'].values
essay_word_len = essay_word_len.reshape(-1, 1)
#print(income)
#print(essay_word_len)
line_fitter = LinearRegression().fit(essay_word_len, income)
income_predict = line_fitter.predict(essay_word_len)
plt.plot(essay_word_len, income, 'o')
plt.plot(essay_word_len, income_predict)
plt.show()


# I've come to the conclusion that the number of words in the essay doesn't correlate with the income at all. The use of many words isn't a signal for a better income.

# 7.2) Regression - Multi Regression

# Maybe we get a better input with multiple regression parameters. So maybe there's a correlation if someone uses the words "successful", "me" or "I".

# In[24]:


#Preparation
def return_word_type(text):
    value_as_string = str(text)
    counter = 0
    for char in '-.,\n':
        value_as_string = value_as_string.replace(char,' ')
    value_as_string = value_as_string.lower()
    word_list = value_as_string.split()
    for word in word_list:
        if 'successful' in word or 'me' in word or 'I' in word:
            counter +=1            
    return counter

essay_cols_2 = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays_2 = df[essay_cols_2].replace(np.nan, '', regex=True)
all_essays_2 = all_essays_2[essay_cols_2].apply(lambda x: ' '.join(x), axis=1)
df['essay_word_type'] = df.apply(lambda row: return_word_type(row), axis=1)


# 7.3) Normalize the data for regression

# In[25]:


feature_data = df[["essay_word_type", "essay_count_words"]]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


# Let's say we want to test the accuracy of using several combinations. What's the outcome going to be?

# In[35]:


accuracy_data = [[50, 100], [70, 250], [30, 80]]
scaler = MinMaxScaler()
print(scaler.fit(df[["essay_word_type", "essay_count_words"]]))
print(scaler.data_max_)
print(scaler.transform(accuracy_data))


# In[29]:


regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
regressor.fit(feature_data, df['income'])


# In[36]:


print(regressor.predict([[0.69117647,0.14673913], [0.98529412,0.96195652], [0.39705882,0.03804348]]))


# The result is the same. Rising "me" and "I" don't correlate with a high income. Very reassuring at least
