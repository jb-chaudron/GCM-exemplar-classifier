# GCM-exemplar-classifier

## 1 - Principe

In the theory of human categorization, a theory had been overhelming, the theory of exemplars.
Even though nowadays, many new models have appeared, the simple model using exemplars are still highly relevant.
Here is an implementation of the famous GCM (even though one might argue this is more an ALCOVE like implementation) where classes are predicted based on the similarity between an element to be classified and the exemplars used for the training.

The model output the probability of a class $R$ understood as 

$$
P(R|y) = \frac{\Beta_R \sum_{x \in R} s(x,y)}{\sum_r \Beta_r \sum_{x \in r} s(x,y)}
$$

Where $R$ and $r$ are classes in our dataset. $s_R$ is given by 

$$
s(x,y) = exp(-c d(x,y))
$$

Where $c$ is a parameter controlling the amount of fuzziness associated to an exemplar. The higher $c$ the smaller $d(x,y)$ should be, to output a higher $s(x,y)$ score.
$d(x,y)$ is given by

$$
d(x,y) = \sum_i \alpha_i | x_i - y_i|
$$

thus it is a weighted l1 norme.
