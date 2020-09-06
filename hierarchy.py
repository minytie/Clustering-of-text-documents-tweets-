from sklearn.decomposition import LatentDirichletAllocation,PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.stats import entropy
import matplotlib.pyplot as plt
from collections import Counter
import porter
import string


wordIndex = ""

def getStopSet(path = "stopwords.txt"):#construct the stopwords to set to minimize the time
	with open(path) as file:
		newStopWords = []
		stopwords = file.readlines()
		for element in stopwords:
			newStopWords.append(element.strip())
		stopwords = set(newStopWords)
		return stopwords

def readContent(path = "cnnhealth.txt"):
	with open(path,encoding = 'utf-8') as file:
		content = []
		lines = file.readlines()
		for line in lines:
			line = line.split('|')
			content.append(line[2])
		return content

def preProcessing(content):
	stopSet = getStopSet()
	p = porter.PorterStemmer()
	info = []
	for line in content:
		newLine = ""
		line = line.split(" ")
		for element in line:
			temp = element.split("://")
			temp1 = element.split("@")
			temp2 = element.split("#")
			temp3 = element.split("/")
			if len(temp)<2 and len(temp1)<2 and len(temp2)<2 and len(temp3)<2:
				element = element.strip()#clean the '\n' 
				element = element.lower()
				element = element.translate(str.maketrans('','', string.punctuation))
				element = p.stem(element)
				if element not in stopSet:
					newLine = newLine+element+" "
		info.append(newLine)
	return info

def toVector(content,n_components):
	vectorizer = TfidfVectorizer()
	vector = vectorizer.fit_transform(content)
	vector = vector.toarray()
	lda = LatentDirichletAllocation(n_components=n_components)
	result = lda.fit_transform(vector)
	return result

def cluster(vector):
	model = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')
	#model = AgglomerativeClustering(n_clusters=13, affinity='manhattan', linkage='average')
	model.fit(vector)
	return model.labels_

	
def pca(vector):
	pca_vec =  PCA(n_components=2).fit_transform(vector)
	return pca_vec
	
def plot(vector,labels,numLabels):
	for i in range(numLabels):
		plt.scatter(vector[labels==i, 0], vector[labels==i, 1])
	plt.show()
	

content = readContent()
content = preProcessing(content)
vector = toVector(content,5)
#dendrogram = sch.dendrogram(sch.linkage(vector, method='ward'))
result = cluster(vector)
#print(result)
score = 0
score = silhouette_score(vector,result)
lst = result.tolist()
a = Counter(lst)
print(a)
print(entropy(result, base = 2))
print(score)
diffSet = set(result)
print(len(diffSet))
vector = pca(vector)
#print(vector)
plot(vector,result,len(diffSet))


def get_top_features_cluster(content, prediction, n_feats):
	labels = np.unique(prediction)
	vectorizer = TfidfVectorizer()
	vector = vectorizer.fit_transform(content)
	tf_idf_array = vector.toarray()

	dfs = []
	for label in labels:
		id_temp = np.where(prediction==label) # indices for each cluster
		x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
		sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
		features = vectorizer.get_feature_names()
		best_features = [(features[i], x_means[i]) for i in sorted_means]
		df = pd.DataFrame(best_features, columns = ['features', 'score'])
		dfs.append(df)
	return dfs
def top_2_cluster(dfs):
	scores = []
	for i in dfs:
		score = i["score"].sum()
		scores.append(score)
	x = sorted(scores,reverse=True)[:2]
	best = scores.index(x[0])
	sec_best = scores.index(x[1])
	return best, sec_best
dfs = get_top_features_cluster(content,result, 20)
best, sec_best = top_2_cluster(dfs)
print(dfs[best][:20]['features'].tolist())
print(dfs[sec_best][:20]['features'].tolist())