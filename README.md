# Learning Interpretable explanations by optimising Saliency Maps at training time 

Despite their success deep neural networks still lack interpretability and are regarded as black boxes. Tentative explanations are usually derived after training with post-hoc explainable artificial intelligence (xAI) methods, such as Input Gradients or Shapley values. However, there is no guarantee that models learn faithful attributions, a goal they were not trained for. In this study we illustrates an approach to enforce desirable properties on the saliency maps by adding specific explainability constraints in the loss function when training neural networks (NNs). We consider the following regularization terms for the gradients of the output wrt the input (perhaps the simplest way to obtain feature attributions with NNs): 

* Vanilla Gradient Regularization, that penalises large values for attributions (somehow inspired by standard Lasso).
* Consistency, that enforces similar attributions for instances in the same class. 
* Smoothness, that requires small differences in the attributions of nearby pixels.  
* Locality, that forces the NNs to focus on where the signal is present and neglect irrelevant features.
* Fidelity, that ensures that attributions with higher values actually correspond to more relevant input features for the model.
* Symmetry, that minimises the difference between the rotated saliency map and the saliency map of the rotated input image (other isometries could be easily incorporated). 

The post-hoc explanations obtained with these six different penalization terms are compared among them and with those obtained with a standard unconstrained model. In order to make a quantitative comparison, the following metrics and tests are performed: 

* Most Relevant First Out. 
* Faithfulness. 
* Complexity or entropy. 
* RemOve And Retrain.  
* Gaussian perturbations. 
* Square block perturbations. 

Experiments are performed on the MNIST and Fashion MNIST datasets by training shallow CNNs. The deep learning models are developed in PyTorch and Captum is used for generating post-hoc explanations. In particular, in order to run the code, the following libraries should be installed: 

* numpy
* 
