import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics



#get the bath to the dataset
path = 'D:\mydata.txt'

#read the data and give the name to each column
data = pd.read_csv(path, header=None,names=['Reclamation', 'Categorie'],sep='\t')

#print data
print('My Dataset : \n', data)
#Checking for the label counts in the categorical parameters 
print(data['Reclamation'].value_counts())
print(data['Categorie'].value_counts())


#data description (count, mean,std,min, 25%, 50%, 75%, max)
print('My data describtion : \n')
print(data.describe())#"
#data['char_length'] = data['Reclamation'].apply(lambda x : len(x))
#data['token_length'] = data['Reclamation'].apply(lambda x : len(x.split(" ")))

avg_df = data.groupby('Categorie').agg({'char_length':'mean', 'token_length':'mean'})
fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
ax1.bar(avg_df.index, avg_df['char_length'])
ax2.bar(avg_df.index, avg_df['token_length'], color='green')
ax1.set_title('Avg number of characters')
ax2.set_title('Avg number of token(words)')
ax1.set_xticklabels(avg_df.index, rotation = 45)
ax2.set_xticklabels(avg_df.index, rotation = 45)
plt.show()
"

##################
########33
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=XX, 
                      y=XX[1].astype(np.int_),
                      clf=clf, 
                      legend=2)
#############""
#centers2D = pca.transform(XX)
#plt.scatter(data2D[:,6], data2D[:,6], c=newsgroups_train.target)
#plt.scatter(centers2D[:,0], centers2D[:,1], 
 #           marker='x', s=2, linewidths=1, c='r')
#plt.show()

newsgroups_train = fetch_20newsgroups(subset='train', 
                                      categories=['yellow', 
                                                  'yellow','green', 
                                                  'red', 'blue'
                                                  , 'red']
                                      )

newsgroups_test = fetch_20newsgroups(subset='test', 
                                      categories=['talk.religion.misc', 
                                                  'comp.graphics','alt.atheism', 
                                                  'sci.space', 'soc.religion.christian'
                                                  , 'comp.windows.x'] 
                                     )
XTRAIN = vectorizer.fit_transform(newsgroups_train.data).todense()
XTEST = vectorizer.fit_transform(newsgroups_test.data).todense()

pca = PCA(n_components=6).fit(XTRAIN)
data2D = pca.transform(XTRAIN)

newsgroups_train = fetch_20newsgroups(subset='train', 
                                      categories=['talk.religion.misc', 
                                                  'comp.graphics','alt.atheism', 
                                                  'sci.space', 'soc.religion.christian'
                                                  , 'comp.windows.x'])

XX = vectorizer.fit_transform(newsgroups_train.data).todense()


centers2D = pca.transform(XX)
plt.scatter(data2D[:,0], data2D[:,1], c=newsgroups_train.target)
plt.scatter(centers2D[:,0], centers2D[:,1], 
            marker='x', s=2, linewidths=1, c='r')
plt.show()

pca = PCA(n_components=6).fit(XX)
data2D = pca.transform(XX)
plt.scatter(data2D[:,0], data2D[:,1], s=50, c=newsgroups_train.target)
plt.show()



###########3###33#


# Select ONLY 2 features
X = np.array(X)
y = newsgroups_train.target
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:], X[:]
ax.scatter(X0, X1, c='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()








#Draw the data#
col = 'Categorie'
fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(10,8))
explode = list((np.array(list(data[col].dropna().value_counts()))/sum(list(data[col].dropna().value_counts())))[::-1])[:]
labels = list(data[col].dropna().unique())[:]
sizes = data[col].value_counts()[:]
ax2.pie(sizes,  explode=explode, startangle=360, labels=labels,autopct='%1.0f%%', pctdistance=0.6)
ax2.add_artist(plt.Circle((0,0,15),0.35,fc='white'))
sns.countplot(y =col, data = data, ax=ax1)
ax1.set_title("Count of each categorie")
ax2.set_title("Percentage of each categorie")
plt.show()



#################################################"
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import inflect
import contractions
from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
    
def denoise_text(text):
        # Strip html if any. For ex. removing <html>, <p> tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        # Replace contractions in the text. For ex. didn't -> did not
        text = contractions.fix(text)
        return text
    
    ## Next step is text-normalization
    
    # Text normalization includes many steps.
    
    # Each function below serves a step.
    
    
def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    
def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words
    
    
def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    
def replace_numbers(words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    
def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    
def stem_words(words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems
    
    
def lemmatize_verbs(words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas
    
    

    
data = remove_non_ascii(data)
data =to_lowercase(data)
data =remove_punctuation(data)
data =replace_numbers(data)
data =remove_stopwords(data)
data =lemmatize_verbs(data)
data =stem_words(data)
print("After Text Preprocessing \n:", data)"




##################
plt.scatter(data2D[:,0], data2D[:,1], s=10, c=newsgroups_train.target)

plt.scatter(data2D[:,0], data2D[:,2], s=10, c=newsgroups_train.target)

plt.scatter(data2D[:,0], data2D[:,3], s=10, c=newsgroups_train.target)

plt.scatter(data2D[:,0], data2D[:,4], s=10, c=newsgroups_train.target)

plt.scatter(data2D[:,0], data2D[:,5], s=10, c=newsgroups_train.target)

plt.title("Training Data \n", size=20)
plt.show()

pca2 = PCA(n_components=6).fit(XTEST)
data2D2 = pca2.transform(XTEST)
plt.scatter(data2D2[:,0], data2D2[:,1], s=10, c=newsgroups_test.target)

plt.scatter(data2D2[:,0], data2D2[:,2], s=10, c=newsgroups_test.target)

plt.scatter(data2D2[:,0], data2D2[:,3], s=10, c=newsgroups_test.target)

plt.scatter(data2D2[:,0], data2D2[:,4], s=10, c=newsgroups_test.target)

plt.scatter(data2D2[:,0], data2D2[:,5], s=10, c=newsgroups_test.target)
plt.title("Test Data \n", size=20)
plt.show()
##########""
("cleaner", processed_feature),
# ('clf', svm.SVC(kernel='linear', C=1 ,decision_function_shape='ovr')),])
###########""""
if (predictions  == 'respect'):
    result[1] = 'Problème de Respect'
elif (predictions == 'sécurité'):
    result[1] = 'Problème de Sécurite'
elif (predictions  == 'metier'):
    result[1]= 'Problème de Métier'
elif (predictions  == 'technique'):
    result[1] = 'Problème Technique'
elif (predictions == 'gestion'):
    result[1]= 'Problème de Gestion'
elif (predictions == 'comportement'):
    result[1] = 'Problème de Comportement'   

############################################

data['char_length'] = y.apply(lambda x : len(x))
data['token_length'] =X.apply(lambda x : len(x.split(" ")))

fig, ax = plt.subplots(figsize=(16,8))
for sentiment in data['Categorie'].value_counts().sort_values()[-6:].index.tolist():
    #print(sentiment)
    ax.scatter(data[data['Categorie']==sentiment]['token_length'],
                data[data['Categorie']==sentiment]['char_length'], s=10)
    
labels = ['comportement', 'gestion', 'metier', 'respect' ,'sécurité','technique'   ]

plt.legend(labels)  
ax.set_title("Distribution of character length sentiment-wise [Top 5 sentiments]")
plt.show()