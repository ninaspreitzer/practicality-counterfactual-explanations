# Evaluating the Practicality of Counterfactual Explanations

This repository holds key information about my *Masterthesis (18 EC)* for the Master's degree *Information Studies: Data Science Track* at the University of Amsterdam. 
* **Title:** Evaluating the Practicality of Counterfactual Explanations
* **Supervisor:** Ilse van der Linden, MSc
* **Institue:** [Civic AI Lab](https://www.civic-ai.nl/)
* **Submission Date:** 30-06-2022

## Abstract 
Machine learning models are increasingly used for decisions that directly affect people’s lives. These models are often a black box, meaning that the people affected cannot understand how or why the decision was made. However, according to the General Data Protection Regulation, they have the right to an explanation. 
<img width="846" alt="ML_decision" src="https://user-images.githubusercontent.com/57034840/190699746-06b60271-0321-46e5-91bd-1eaa4580330a.png">

Counterfactual explanations are a way to make machine learning models more transparent by showing how attributes need to be changed to get a different result. This type of explanation is considered easy to understand and human-friendly. To be used in real life, explanations must be practical, which means they must go beyond a purely theoretical framework. Research has focused on defining several objective functions to compute practical counterfactuals. However, it has not yet been tested whether people perceive the explanations as such in practice. To address this, we contribute by identifying properties that explanations must satisfy to be practical for human subjects. The properties are then used to evaluate the practicality of two counterfactual methods by conducting a user study. The study aims to compare the original approach to compute counterfactual explanations to a more recent framework. The results show that human subjects perceive the more recent method as more practical.
<img width="851" alt="Counterfactual_Explanation" src="https://user-images.githubusercontent.com/57034840/190699802-04623ac7-74a5-49e1-838e-91206655c040.png">

## Content 
#### 1 Survey
The full survey is available under [this link](https://uva.fra1.qualtrics.com/jfe/form/SV_8ccTefLDEIFxF8a). One example is saved [here]((Survey/Survey_Student_Example.pdf), as well as the survey flow, [here](Survey/Survey_Flow.pdf), which shows how the 30 different instances were randomized in the survey tool, named Qualtrics.

#### 2 Counterfactual Explanations
Documentation on computation of the [counterfactual explanations](Counterfactuals_Adult.ipynb). The code computing counterfactual explanations following the CARE framework is taken from Peyman Rasouli's open source [code](https://github.com/peymanrasouli/CARE). To resemble explanations following [Wachter et al.](https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf) we remove the Sparsity objective in the code of CARE and only set the Validity to _True_ and Soundness, Coherency, and Actionability to _False_.

#### 3 Datasets
For the study we use the Adult Income dataset and the Student Performance dataset. Both can be accessed through the UCI machine learning [repository](http://archive.ics.uci.edu/ml).

#### 4 Preprocessing
The [preprocessing steps](prepare_datasets.py) follow [Zhu](https://rstudio-pubs-static.s3.amazonaws.com/235617_51e06fa6c43b47d1b6daca2523b2f9e4.html) and [Rasouli et al.](https://arxiv.org/abs/2108.08197). 

#### 5 Model
The [classification model](create_model.py), a multi-layer neural network, is the same model as used by [Rasouli et al.](https://arxiv.org/abs/2108.08197).
