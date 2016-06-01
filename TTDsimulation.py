"""
Supplementary to:
Wu, CM., Meder, B., Filimon, F., Nelson, JD. (2016) Asking Better Questions: How presentation formats influence information search
@author: Charley M. Wu cwu@mpib-berlin.mpg.de
"""

import numpy as np


"""Define Probability Gain and Take the Difference (TTD) Heuristic"""

#inverts a binary variable
def invert(p): 
	return 1 - p

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

"""Define function to check equivalency of predictions between two models"""
def equivalency(Model1_eu1, Model1_eu2, Model2_eu1, Model2_eu2):
	#calculate difference in expected utilities for each model
	diffModel1 = Model1_eu1 - Model1_eu2 #positive if query 1 is more useful, negative if query 2 is more useful, 0 if both queries are equally useful
	diffModel2 = Model2_eu1 - Model2_eu2
	if diffModel1 == 0 or diffModel2 == 0:
		#if one of the model makes no prediction, due to equal expected utilities for both queries, return true
		return True
	else:
		#if both models predict the same query, then the product of the differences will be greater than zero
		if (diffModel1 * diffModel2) > 0:
			return True
		else:
			return False


"""Simulations comparing query predictions between probability gain and TTD"""

def simulate(iterations):
	equivalent = True #assume equivalency between probability gain and TTD is true, and try to falsify
	for i in range(iterations):
		testEnv = np.random.rand(5) #generate random environments
		[euPG1,euPG2] = pGain(testEnv) #calculate expected utilities for probability gain
		[euTTD1,euTTD2] = TTD(testEnv) #calculate expected utilities for TTD
		equivalencyCheck = equivalency(euPG1, euPG2, euTTD1, euTTD2)
		if equivalencyCheck==True:
			continue
		else:
			return False
	return equivalent

"""Running the simulation"""

iterations = 10000000

results = simulate(iterations)

print "Is TTD equivalent to probability gain?"
print results
