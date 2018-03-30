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

More notes to come..