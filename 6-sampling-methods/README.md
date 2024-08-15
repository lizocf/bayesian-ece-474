Instructions

The first part of the sampling methods project is to perform rejection sampling to draw samples from a Gaussian mixture model. 

 
Use the same mixture model from the EM project (1-D only), and I suggest using regular normal RV for the proposal distribution. 
Draw samples using this method and plot a histogram of the generated samples against your GMM.

 
Part two of the sampling methods mini-project is to re-do your linear regression project using MCMC to find an estimate for the weights.

 
Reuse your project 2 to generate the same training data. Just do this for 25 training samples

 
Use Equation 3.10 as the likelihood function, to be used with the training samples you generated. 
You may select any distribution you want for the prior on the weights, and recall that the posterior density on the weights w is 
proportional to the likelihood x prior. Note, you'll want to take the log of the posterior to keep this numerically stable

 
Use the Metropolis algorithm as defined in equation 11.33 to compute an estimate of the weights


Stretch goal, 5 points:  Take MCMC all the way and draw samples from the predictive distribution as well. See here for some hints: https://stats.stackexchange.com/questions/185311/why-does-sampling-from-the-posterior-predictive-distribution-px-new-mid-x-1