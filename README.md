# Evaluating the Practicality of Counterfactual Explanations

This repository holds key information about my *Masterthesis (18 EC)* for the Master's degree *Information Studies: Data Science Track* at the University of Amsterdam. 
Title: Evaluating the Practicality of Counterfactual Explanations
Supervisor: Ilse van der Linden

## Content 
**1. Survey:** Example and survey flow of the conducted survey, which can be accessed [here](https://uva.fra1.qualtrics.com/jfe/form/SV_8ccTefLDEIFxF8a) 
**2. Counterfactual Explanations:** Documentation on computation of the counterfactual explanations used for the study. The code computing counterfactual explanations following the CARE framework is taken from Peyman Rasouli's open source [code](https://github.com/peymanrasouli/CARE). To resemble explanations following [Wachter et al.](https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf) we remove the Sparsity objective in the code of CARE and only set the Validity to _True_ and Soundness, Coherency, and Actionability to _False_.
**3. Datasets:** For the study we use the Adult Income dataset and the Student Performance dataset. Both can be accessed through the UCI machine learning [repository](http://archive.ics.uci.edu/ml).
**4. Preprocessing:** The preprocessing steps follow [Zhu](https://rstudio-pubs-static.s3.amazonaws.com/235617_51e06fa6c43b47d1b6daca2523b2f9e4.html) and [Rasouli et al.](https://arxiv.org/abs/2108.08197) 
**5. Model:** The classification model, a multi-layer neural network, is the same model as used by [Rasouli et al.](https://arxiv.org/abs/2108.08197) 


## Abstract 
Machine learning models are increasingly used for decisions that directly affect people’s lives. These models are often a black box, meaning that the people affected cannot understand how or why the decision was made. However, according to the General Data Protection Regulation, they have the right to an explanation. Coun- terfactual explanations are a way to make machine learning models more transparent by showing how attributes need to be changed to get a different result. This type of explanation is considered easy to understand and human-friendly. To be used in real life, explana- tions must be practical, which means they must go beyond a purely theoretical framework. Research has focused on defining several objective functions to compute practical counterfactuals. However, it has not yet been tested whether people perceive the explanations as such in practice. To address this, we contribute by identifying properties that explanations must satisfy to be practical for human subjects. The properties are then used to evaluate the practicality of two counterfactual methods by conducting a user study. The study aims to compare the original approach to compute counterfactual explanations to a more recent framework. The results show that human subjects perceive the more recent method as more practical.
