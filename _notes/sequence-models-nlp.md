---
layout: default
title: Sequence Models - NLP & Word Embeddings
---

# {{page.title}}

My course notes for the [Coursera class](https://www.coursera.org/learn/nlp-sequence-models).

## Word Embeddings
---
[Previously](sequence-models) we used a vocabulary and one-hot vector encoding to represent words.  
The problem with this is that it doesnt capture relationships between words at all.  

For example, an RRN might learn to predict  
**I want a glass of orange _juice_**  
but not generalize to  
**I want a glass of apple _juice_**

This is because the inner product between any two separate one-hot word vectors is 0.

An alternative is to expend the representation of each word so that it includes some features:  

|  | Man | Woman | King | Queen | Apple |Orange
|:---|:--- |:--- |:--- |:--- |:--- |:--- |
| Sex | -1 | 1 | -0.95 | 0.95 | 0 | 0 |
| Royal | 0 | 0 | 1 | 1 | 0.01 | 0 |
| Age | 0 | 0 | 0.7 | 0.69 | 0 | 0 |
| Food | 0 | 0 | 0 | 0 | 1 | 1 |
| ... | ... | ... | ... | ... | ... | ... |
| Verb | 0.5 | 0.1 | 0.3 | 0.3 | 0 | 0 |  

<br>
The word features will be learned and not as easy to understand, but they'll serve the same purpose as above.  

These groupings can be visualized. Its called _embedding_ because the representation can be embedded into some point in a 2D/3D space:

![2D Embedding Visualization](/assets/study-notes/sequence-models/nlp/1.png)  
**Fig 1**

---

Instead of using one-hot word representations, you use word embeddings. This makes it possible to make useful predictions
on words that you have embeddings for, but don't have in your training set.

![Named entity recognition](/assets/study-notes/sequence-models/nlp/2.png)  
**Fig 2**

For example, in figure 1, your training set contained the words `orange` and `farmer`, and this let you predict that `Sally Johnson`
is a name.

Your training example does not contain `dorian` and `cultivator` though, _but_ you do have the word embeddings, and
these let you know that `dorain` and `cultivator` are pretty close to `orange` and `farmer`. Therefore, you can still
 predict that `Robert Lin` is a name.
 
**The steps for this kind of transfer learning are:**  
 
 1. Learn (or download) word embeddings from a large text corpus. (1-100B words)
 2. Transfer embeddings to new task with smaller training set (~100k words)
 3. Continue to finetune the words embeddings with new data.
 
 ---
 
**Relation to face encoding**
   
In face encoding, the convolution neural network learns how to encode images of faces, similar to a hashing function.
New images can then be encoded and compared to the set of known images, to check if they are similar.
 
In leaning word embeddings, there is a fixed vocabulary of words (say $e_1, ... , e_{10,000}$) that learns a fixed embedding
for each of those words.
   
![Named entity recognition](/assets/study-notes/sequence-models/nlp/3.png)  
**Fig 3**

---

**Analogy Reasoning**  

|  | Man | Woman | King | Queen | Apple |Orange
|:---|:--- |:--- |:--- |:--- |:--- |:--- |
| Sex | -1 | 1 | -0.95 | 0.95 | 0 | 0 |
| Royal | 0 | 0 | 1 | 1 | 0.01 | 0 |
| Age | 0 | 0 | 0.7 | 0.69 | 0 | 0 |
| Food | 0 | 0 | 0 | 0 | 1 | 1 |

<br>
With word embeddings, an algorithm can be written that learns analogies, such as:

`Man -> Woman` as `King -> ?`

$e_{man} - e_{woman} \approx \begin{bmatrix}
-2 \\\ 0 \\\ 0 \\\ 0
\end{bmatrix}
$

$e_{king} - e_{queen} \approx \begin{bmatrix}
-2 \\\ 0 \\\ 0 \\\ 0
\end{bmatrix}
$

The difference between `man/woman` and `king/queen` is the sex, and the difference between both pairs is the same and
 this is a good analogy match.
 
So to solve these analogy match problems, find $e_?$ where $e_{man} - e_{woman} \approx e_{king} - e_{?}$.  

![Vector difirence](/assets/study-notes/sequence-models/nlp/4.png)  
**Fig 4**

_Fig 4: This visualization works with 2D/3D vectors, but isn't as clear with higer-dimensional embeddings._

Find word `w`: $argmax_w \space similarity(e_w, e_{king} - e_{man} + e_{woman})$  
(the `w` which maximizes the similarity function)  

**Similarity Function: Cosine Similarity**  
$sim(u, v) = \frac{u^{T}v}{\lVert u \rVert_2 \lVert v \rVert_2} = \cos(\theta)$  

![Cosine similarity](/assets/study-notes/sequence-models/nlp/cosine_sim.png)

_reminder:_ norm of `u` = $\lVert u \rVert_2 = \sqrt{\sum^n_{i=1} u^2_i }$


**Alternative Similarity Function: Squared Distance**
$sim(u, v) = \lVert u - v \rVert^2$  
Cosine similarity is used more often though.

---

**Embedding Matrix**  
$E = n_{embedding \space features} \times n_{vocabulary \space size}$ sized matrix.  
A word can be identified by an index. $O_w$ = some one-hot vector for that word.  
$E * O_w$ = $n_{embedding \space features} \times 1$ sized matrix = $e_w$ = embedding for word `w`.

![Embedding Matrix](/assets/study-notes/sequence-models/nlp/5.png)  
![Embedding Matrix Multiplication](/assets/study-notes/sequence-models/nlp/6.png)

Gradient descent is used to learn the parameters of matrix $E$. All the paramters of $E$ are initialized randomly.

_In practice, a column lookup function is used instead of matrix multiplication because it is more efficient._

---

### Learning Word Embeddings
 
You can learn word embeddings by learning a [language model](https://en.wikipedia.org/wiki/Language_model).  


Example - using a vocabulary size of 10,000 and 300 word embeddings (making $E$ a 300, 10,000 matrix):  

![Neural Language Model](/assets/study-notes/sequence-models/nlp/7.png)
**Fig 5**

_Fig 5: $o_n$ is the one-hot vector representation for that word.  
$e_n$ is the embedding representation vector of size 300 for word $n$._

The `softmax` returns a probability distribution for the next word in the sequence - in this case, "juice" should be the
most likely.

**The parameters of the learning algorithm will be**:  
$E$: The embedding matrix  
$W^{[1]}, b^{[1]}$: the dense layer  
$W^{[2]}, b^{[2]}$: the softmax layer
  
![Neural Language Model - Paramaters](/assets/study-notes/sequence-models/nlp/8.png)  
**Fig 6**
  
_Fig 6: the $e_n$ vectors can be stacked together, giving a vector of size 1800 in this case ($6 \times 300$)._

**You can also use an arbitrary window for selecting words.** Instead of using the entire previous phrase, you can just
use the last 4 words or something else like that.  

![Other Context / Target Pairs](/assets/study-notes/sequence-models/nlp/9.png)  
**Fig 7**

---

**Learning Model: Skip-Gram Model (Word2Vec)**  

Its called skip-gram because you choose some target $t$ that skips some words ahead of contect $c$.  

![Skip-Gram Model](/assets/study-notes/sequence-models/nlp/10.png)  
**Fig 8**

The problem with this model is that the summation in `softmax` is computationally intensive, since you have to sum over the
entire vocabulary:

$$\textrm{softmax:}\space p(t \vert c) = \frac{e^{\theta^T_t e_c}}{\sum^{10,000}_{j=1} e^{\theta^T_j e_c}}$$

---

**Learning Model: Negative Sampling**

Looking at the problem another way; pick some context/target pairs and do binary classification to see which are correct.

**I want a glass of _orange_ juice to go along with my cereal.**  

| Context | Word | Target?
|:---|:--- |:--- |
| orange | juice | 1 |
| orange | king | 0 |
| orange | book | 0 |
| orange | the | 0 |
| orange | of | 0 |

<br>
Then create a supervised learning problem to train on these pairs:    
![Negative Sampling](/assets/study-notes/sequence-models/nlp/11.png)  
**Fig 9** 

_Fig 9: $K$ should be 5-20 for smaller training sets and 2-5 for largest training sets._

The actual model looks like this:

![Negative Sampling Model](/assets/study-notes/sequence-models/nlp/12.png)  
**Fig 10**  

$P(y=1 \vert c, y) = \textrm{sigmoid}(\theta^T_t e_c)$

$o_{6257} \longrightarrow E \longrightarrow e_{6257} \longrightarrow \textrm{is "juice"?, is "king"?, etc...}$

There are 10,000 binary classification problems instead of a `softmax` with summation, but you only train on a very small
subset of those problems with each iteration.

**How do you sample the negative examples?**  
Option 1: sample based on frequency of each word  
Option 2: sample randomly: $\frac{1}{\vert v \vert}$  
Option 3: sample with a heuristic that combines the two (best):  

$$P(w_i) = \frac{f(w_i)^{\frac{3}{4}}}{\sum^{10,000}_{j=1} f(w_j)^{\frac{3}{4}} }$$

---

**Learning Model: GloVe Word Vectors**  

For every $c, t$, count how many times in the text corpus the target $t$ appears in context of $c$.  

$x_{ij}$ = Number of times $i$ ($t$) appears in context of $j$ ($c$).

$$ \textrm{minimize:} \space \sum_{i=1}^{10,000} \sum^{10,000}_{j=1} f(X_{ij}) (\theta^T_i e_j + b_i + b_j - \log X_{ij})^2$$

Where $f(X_ij)$ is a weighting term which equals 0 if $X_{ij}$ is 0, and call also be used to tweak the weight of certain words
(commonly "this", "is", "of", "a", etc...)

![GloVe Model](/assets/study-notes/sequence-models/nlp/13.png)  
**Fig 11** 

---

### Applications using Word Embeddings  

**Sentiment Classification**  

![Sentiment Classification - Simple Model](/assets/study-notes/sequence-models/nlp/14.png)  
**Fig 12**  

The problem with this though is that it isn't able to account for word order.  
**"Completely lacking in _good_ taste, _good_ service, and _good_ ambiance."** would probably be classified as a positive
review because of the frequency of _good_, even though it is a very negative review.  

A solution is to use a RNN model instead:

![Sentiment Classification - RNN Model](/assets/study-notes/sequence-models/nlp/15.png)  
**Fig 13**

---

**Debiasing Word Embeddings**  

**Man:Woman as King:Queen** [OK]  
**Man:Computer_Programmer as Woman:Homemaker** [Undesirable]  
**Man:Doctor as Woman:Nurse** [Undesirable]

Word embeddings will reflect the gender/ethnic/age/etc... biases from the text used to train the model, which is
undesirable in many applications.

![Bias Embeddings](/assets/study-notes/sequence-models/nlp/16.png)  
**Fig 13**

1. Identify Bias direction.  
$$\textrm{average} \left\{\begin{array}{l}e_{he} - e_{she} \\ e_{male} - e_{female} \\ \vdots \end{array}\right.$$

2. Neutralize: For every word that is not definitional (gender/age/etc.. not part of its dictionary definition),
project it to get rid of the bias.

    ![Neutralization](/assets/study-notes/sequence-models/nlp/18.png)  
    **Fig 14**  

    The neutralization step takes a vector such as $e_{receptionist}$ and zeros out the component in the direction of $g$,
    giving us $e_{receptionist\\_debiased}$.  
    
    $$e^{bias\_component} = \frac{e \cdot g}{||g||_2^2} * g$$
    
    $$e^{debiased} = e - e^{bias\_component}$$
    
    _$e^{bias\_component}$ is a projection of $e$ onto $g$._

3. Equalize pairs - make the word pairs equidistant from the non-bias axis.

![Bias Embeddings - After](/assets/study-notes/sequence-models/nlp/17.png)  
**Fig 15**