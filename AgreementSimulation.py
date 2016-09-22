from __future__ import division #for floating point outputs from integer division
"""
Simulates 10 million search environments to test for agreements of information gain, probability gain, the probability of certainty heuristic, and the Take-The-Difference heuristic

Supplementary to:
Wu, C. M., Meder, B., Filimon, F., Nelson, J. D. (2016) Asking Better Questions: How presentation formats influence information search
@author: Charley M. Wu cwu@mpib-berlin.mpg.de
"""
import numpy as np
import itertools


"""Define Information Gain, Probability Gain, the probability of certainty heuristic, and Take-The-Difference heuristic"""

#inverts a binary variable
def invert(p):
	return 1 - p

#Shannon Entropy
def ent(p):
    """for binary values of p"""
    if p == 0 or p == 1:
        return 0
    else:
        invP = 1 - p
        ent = (p*np.log2(1/p)) + (invP*np.log2(1/invP))
        return ent

#Information Gain
def iGain(environmentArray): #environmentArray is 5 dimensions, comprising baseRate and 4 feature likelihoods
    baseRate = np.array([environmentArray[0], 1 - environmentArray[0]])
    feat1 = [environmentArray[1], environmentArray[2]]
    feat2 = [environmentArray[3], environmentArray[4]]
    marginals = [np.sum(feat1 * baseRate), np.sum(feat2*baseRate)]
    post1 = [feat1[0]*baseRate[0]/marginals[0], invert(feat1[0])*baseRate[0]/invert(marginals[0])]
    post2 = [feat2[0]*baseRate[0]/marginals[1],invert(feat2[0])*baseRate[0]/invert(marginals[1])]
    #if baserate is 0 or 1, iGain is 0
    if baseRate[0] == 0 or baseRate[0] == 1:
        eu1 = 0
        eu2 = 0
    else:
        priorEntropy = ent(baseRate[0])
        eu1 = priorEntropy - ((marginals[0] * ent(post1[0])) + ((1 - marginals[0]) * ent(post1[1])))
        eu2 = priorEntropy - ((marginals[1] * ent(post2[0])) + ((1 - marginals[1]) * ent(post2[1])))
    return [eu1, eu2]

#Probability Gain
def pGain(environmentArray, normalize = False): #environmentArray is 5 dimensions, comprising baseRate and 4 feature likelihoods
	baseRate = np.array([environmentArray[0], 1 - environmentArray[0]])
	#Feature Likelihoods
	feat1 = [environmentArray[1], environmentArray[2]]
	feat2 = [environmentArray[3], environmentArray[4]]
	#Marginals
	marginals = [np.sum(feat1 * baseRate), np.sum(feat2*baseRate)]
	#Posterior probabilities
	post1 = [feat1[0]*baseRate[0]/marginals[0], invert(feat1[0])*baseRate[0]/invert(marginals[0])]
	post2 = [feat2[0]*baseRate[0]/marginals[1],invert(feat2[0])*baseRate[0]/invert(marginals[1])]
	#if baserate is 0 or 1, pGain is 0
	if baseRate[0] == 0 or baseRate[0] == 1:
		eu1 = 0
		eu2 = 0
	else:
		priorCorrect = max(baseRate)
		eu1 = np.around(((marginals[0] * max(post1[0], invert(post1[0]))) + ((invert(marginals[0]) * max(post1[1], invert(post1[1])))) - priorCorrect), 14)
		eu2 = np.around(((marginals[1] * max(post2[0], invert(post2[0]))) + ((invert(marginals[1]) * max(post2[1], invert(post2[1])))) - priorCorrect), 14) 
	#normalize to a range of possible eu(*) between 0 and 1
	if normalize == True:
		eu1,eu2 = eu1 *2, eu2 * 2
	return [eu1, eu2] #return expected utilities for query1 and query2

#Probability of Certainty
def pOfC(environmentArray, normalize = False): #environmentArray is 5 dimensions, comprising baseRate and 4 feature likelihoods
	baseRate = np.array([environmentArray[0], 1 - environmentArray[0]])
	#Feature Likelihoods
	feat1 = [environmentArray[1], environmentArray[2]]
	feat2 = [environmentArray[3], environmentArray[4]]
	#Marginals
	marginals = [np.sum(feat1 * baseRate), np.sum(feat2*baseRate)]
	#Posterior probabilities
	post1 = [feat1[0]*baseRate[0]/marginals[0], invert(feat1[0])*baseRate[0]/invert(marginals[0])]
	post2 = [feat2[0]*baseRate[0]/marginals[1],invert(feat2[0])*baseRate[0]/invert(marginals[1])]
	#expected usefulness
	eu1 = max(np.multiply([0 if x<.99 else 1 for x in post1], [marginals[0], invert(marginals[0])])) #using 0.99 instead of 1 to allow for tolerance equivalent to rounding to a whole number
	eu2 = max(np.multiply([0 if x<.99 else 1 for x in post2], [marginals[1], invert(marginals[1])]))
	#return expected utilities for query1 and query2
	return [eu1, eu2]


#Take the Difference Heuristic
def TTD(environmentArray, normalize = False): #environmentArray is 5 dimensions, comprising baseRate and 4 feature likelihoods
	baseRate = np.array([environmentArray[0], 1 - environmentArray[0]])
	#Feature Likelihoods
	feat1 = [environmentArray[1], environmentArray[2]]
	feat2 = [environmentArray[3], environmentArray[4]]
	#Marginals
	marginals = [np.sum(feat1 * baseRate), np.sum(feat2*baseRate)]
	#Posterior probabilities
	post1 = [feat1[0]*baseRate[0]/marginals[0], invert(feat1[0])*baseRate[0]/invert(marginals[0])]
	post2 = [feat2[0]*baseRate[0]/marginals[1],invert(feat2[0])*baseRate[0]/invert(marginals[1])]
	#Natural Frequencies
	freq1 = np.hstack([np.multiply(feat1, baseRate), np.multiply([invert(f) for f in feat1],baseRate)]) 
	freq2 = np.hstack([np.multiply(feat2, baseRate), np.multiply([invert(f) for f in feat2],baseRate)]) 
	#expected utilities of each query
	eu1 = max( abs(freq1[0] - freq1[1]), abs(freq1[2] - freq1[3]))
	eu2 =max( abs(freq2[0] - freq2[1]), abs(freq2[2] - freq2[3]))
	#normalize to a range of possible eu(*) between 0 and 1
	if normalize == True:
		eu1,eu2 = eu1 *2, eu2 * 2
	return [eu1, eu2]



"""Define function to check agreement of predictions between two models"""
def agreement(Model1, Model2):
	#unpack expected utilities
	[Model1_eu1, Model1_eu2] = Model1
	[Model2_eu1, Model2_eu2] = Model2
	#calculate difference in expected utilities for each model
	diffModel1 = Model1_eu1 - Model1_eu2 #positive if query 1 is more useful, negative if query 2 is more useful, 0 if both queries are equally useful
	diffModel2 = Model2_eu1 - Model2_eu2
	#result is returned as a binary vector, corresponding to [agreement, disagreement, no prediction]
	result = [0,0,0]
	if diffModel1 == 0 or diffModel2 == 0:
		#if one of the model makes no prediction, due to equal expected utilities for both queries
		result[2] = 1
	else:
		#if both models predict the same query, then the product of the differences will be greater than zero
		if (diffModel1 * diffModel2) > 0:
			result[0] = 1
		else:
			#if there is a disagreement, return -1
			result[1] = 1
	return result #return a binary vector where the position of the 1 corresponds to [agreement, disagreement, no prediction]


"""Simulations comparing query predictions between probability gain and TTD"""

iterations = 10000000 #10 million iterations

if __name__ == "__main__":
	"""Main Simulation Code"""
	models = [pGain, iGain, pOfC, TTD] #list of models to compare
	modelNames = ["Probability Gain", "Information Gain", "Probability of Certainty Heuristic", "Take-The-Difference Heuristic"] #human readable model names
	#Create each pair-wise combination of models
	combs = list(itertools.combinations(range(len(models)),2)) 
	#empty array to store results
	results = np.zeros([len(combs),3]) #rows stores each unique pair-wise comparison between models, columns store the number of [agreements, disagreements, no predictions]
	#loop through number of replications
	for i in range(iterations):
		testEnv = np.random.rand(5) #generate random environments
		predictions = [] #list to store expected utilities computed by each model
		#iterate through each model
		for m in models:
			predictions.append(m(testEnv)) #add expected utilties to prediction list
		#iterate through each pairwise comparison
		for c in range(len(combs)):
			(m1,m2) = combs[c] #assign m1 and m2, which are integers used to index the list of models
			results[c,] += agreement(predictions[m1], predictions[m2]) #increment the results array at the row corresponding to the model comparison
	#Once simulation is complete, print human readable results
	for c in range(len(combs)):
		comparison = combs[c]
		print "****Comparing %s with %s****" % (modelNames[comparison[0]], modelNames[comparison[1]])
		print "Out of all environments:"
		print "%.2f%% agreement, %.2f%% disagreement, %.2f%% no prediction by one or both models" % tuple(results[c,] * 100 / iterations)
		print "Out of the environments where both models make a prediction:"
		print "%.2f%% have agreement and %.2f%% have disagreement" % tuple(results[c,0:2] * 100 / (iterations - results[c,2]))
		print "\n"


