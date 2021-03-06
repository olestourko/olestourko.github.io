---
layout: default
title: Sequence Models - Attention Mechanism
---

# {{page.title}}

My course notes for the [Coursera class](https://www.coursera.org/learn/nlp-sequence-models).

## Sequence-to-Sequence Structures
---
Useful for machine translation; the idea is to encode a string with one RNN, and then pass the encoding into another
sentence-generating RNN to get the translation.

**Jane visite l'Afrique en septembre $\rightarrow$ Jane is visiting African in September.**
$x^{<1>}, x^{<2>}, x^{<3>}, x^{<4>}, x^{<5>} \rightarrow y^{<1>}, y^{<2>}, y^{<3>}, y^{<4>}, y^{<5>}, y^{<6>}$

---
**Language Model vs. Machine Translation Model**  

Language Model:  
![Language Model](/assets/study-notes/sequence-models/attention-mechanism/1.png)  
**Fig 1**  

Predicts the probability of some sentence $P(y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \dots, y^{\langle Ty \rangle})$  

Machine Translation Model:  
![Machine Translation Model](/assets/study-notes/sequence-models/attention-mechanism/2.png)  
**Fig 2**  
_Green: the encoder network_  
_Purple: the decoder network_  

Also called a **conditional language model** because it models the probability of some output conditioned on some input
sentence: $P(y^{\langle 1 \rangle}, \dots, y^{\langle Ty \rangle} \vert x^{\langle 1 \rangle}, \dots, x^{\langle Ty})$  

---

The encoder/decoder model can come up with several translations, some better than others...  
- Jane is visiting Africa in September.  
- Jane is going to be visiting Africa in September.  
- In September, Jane will visit Africa.  
- Her African friend welcomed Jane in September.

The model should therefore the most likely translation; $\textrm{argmax} \space P(y^{\langle 1 \rangle}, \dots, y^{\langle Ty \rangle} \vert x)$
of $y^{\langle 1 \rangle}, \dots, y^{\langle Ty \rangle}$.

---
**Why Not Greedy Search?**

Greedy search picks the best 1st word $y^{\langle 1 \rangle}$, then the best second 2nd word $y^{\langle 2 \rangle}$, etc... 
The problem with this is that we need to look at the entire output sentence's probability.  

For example;  
- Jane is _visiting_ Africa in September.
- Jane is _going_ to be visiting Africa in September.

The 2nd sentence might be picked as best by greedy search if _going_ ($y^{\langle 3 \rangle}$) is considered more
probable than _visiting_, even though the 1st sentence is a better translation overall.

---
**Beam Search**

Greedy search picks the one most probable words and moves on, while Beam Search considers multiple alternatives. It has 
a parameter $B$, which controls the _beam width_. $B = 3$ means there are 3 most likely possibilities to choose from at
any time.

**Step 1**  
For the first step of evaluation, beam search finds the $B$ most likely possibilities by passing the sentence to translate
through an encoder/decoder network fragment (Figure 5) to find $P(y^{\langle 1 \rangle} \vert x )$, where $y^{\langle 1 \rangle}$
is a `softmax` output.

![Beam Search - Step 1](/assets/study-notes/sequence-models/attention-mechanism/3.png)  
**Fig 3**  

**Step 2**  
The second step of evaluation, beam search finds $P(y^{\langle 1 \rangle}, y^{\langle 2 \rangle} \vert x )$, building off
the $B$ best possibilities chosen in the previous step. The decoder network fragments now have another RNN cell, to which
the previous best word is hardcoded as the input. 

![Beam Search - Step 2](/assets/study-notes/sequence-models/attention-mechanism/4.png)
![Beam Search - Step 2](/assets/study-notes/sequence-models/attention-mechanism/5.png)  
 **Fig 4, 5**  
_Note that you'll have 3 copies of the network fragment._

**Step 3..end**  
The same flow is continued for all further steps, and eventual `<EOS>` is found for each beam.

![Beam Search - Step 3](/assets/study-notes/sequence-models/attention-mechanism/6.png)
**Fig 6**  

---
**Refinements to Beam Search: Length Normalization**

Beam search tries to maximize the probability

$$\textrm{argmax} \space \prod_{t=1}^{T_y} P(y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, y^{\langle t-1 \rangle})$$

which approaches a very small problem for larger translation sentences, because $P(y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \vert x) = 
P(y^{\langle 1 \rangle} | x) \times P(y^{\langle 2 \rangle} | x, y^{\langle 1 \rangle}) \dots$

Having a very small number is undesirable because it can cause floating point round-off issues.  

In practice, instead use  

$$\textrm{argmax} \space \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, y^{\langle t-1 \rangle})$$

to avoid these issues. Maximizing this sum of probabilities will still give Beam Search the same result.  
_Note: the log of a probability (n < 1) is always negative, so shouldn't this be a minimization problem instead?_  

You can also normalize the sum by the number of words in the translation, which will give even better results:  

$$\frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, y^{\langle t-1 \rangle})$$  

$\alpha$ is a hyperparameter you can optionally use to soften the normalization (usually set to 0.7). Its really just a 
heuristic and doesn't have a solid theoretical justification for it, but it still works well.  

**Choosing the Beam Width $B$**  

The larger $B$ is set, then you consider a lot of possibilities and get better results, but the computations also get slower.
Its a case of diminishing returns.

- Production Systems: $B$ is between 10 and 100  
- Research Systems: $B$ is between 1,000 and 3,000  

---
**Error Analysis in Beam Search**

Sentence to translate: **Jane visite l'Afrique en septembre.**  
Human translation: **Jane visits Africa in September.** ($y^*$)  
Machine translation: **Jane visited Africa last September.** ($\hat{y}$)  

The machine translation model has two components; the sequence-to-sequence model (encoder/decoder) and the beam search
algorithm. **Which component is most responsible for the translation error?**

Compute $P(y^* \vert x)$ and $P(\hat{y} \vert x)$ using the RNN model.

![RNN Model](/assets/study-notes/sequence-models/attention-mechanism/7.png)  
**Fig 7**  

**Case 1:** $P(y^* \vert x) \gt (\hat{y} \vert x)$  
Beam search chose $\hat{y}$, but $y^*$ attains higher $P(y \vert x)$.  
Conclusion: Beam search is at fault.  

**Case 2:** $P(y^* \vert x) \le (\hat{y} \vert x)$  
$y^\*$ is a better translation than $\hat{y}$, but RRN predicted $P(y^\* \vert x) \lt P(\hat{y} \vert x)$.  
Conclusion: RNN model is at fault.

_Note: if you are using length normalization, you should be evaluating the optimization objective instead of the probabilities._

---
**Bleu Score**  
A method of measuring how good a machine translation is with a single number.

**French:** Le chat est sur le tapis.  
**Reference 1:** The cat is on the mat.  
**Reference 2:** There is a cat on the mat.  
**MT Output:**  The cat the cat on the mat.
  
Blue Score works on pairs of words that are next to each other (bigrams).

$count$ = number of times the bigram appears in the MT Output.  
$count_{clip}$ = maximum number of times the biagram appears in either of the references.  

|**Bigram** | **Count** | **Clipped Count** |
|:---|:---|:---|
| the cat | 2 | 1 |
| cat the  | 1 | 0 |
| cat on | 1 | 1 |
| on the | 1 | 1 |
| the | 1 | 1 |
| the mat | 1 | 1 |

$$\textrm{Bleu Score on n-grams only} = P_n = \frac{\sum_{\textrm{n-gram} \in \hat{y}} count_{clip}(\textrm{n-gram}) }
{\sum_{\textrm{n-gram} \in \hat{y}} count(\textrm{n-gram})}$$

You would compute the Blue Scores for $P_1, P_2, P_3, P_4$ (n-grams 1 to 4) and combine them to get a final Blue Score:  

$$\textrm{BP} \times \exp \left(\frac{1}{4} \sum_{n=1}^{4} P_n \right)$$  

Where $\textrm{BP}$ is a brevity penalty which penalizes translations that are too short.  

$\textrm{BP} = \textrm{1 if MT_output_length > reference_length}$    
$\textrm{BP} = \exp(1 - \textrm{MT_output_length\reference_output_length}) \space \textrm{otherwise.}$

---
**Attention Model**  
When humans translate text; we don't read through an entire long sentence at once before beginning to translation. We look at
fragments instead, and the Attention Model is an RNN equivalent.

![Attention Model - Simplified](/assets/study-notes/sequence-models/attention-mechanism/8.png)  
**Fig 8**  

1. Bidirectional RNN (bottom)
2. Output of bottom RNNs are combined into context $c$, using a set of attention weights $\alpha^{\langle t, t' \rangle}$ ($t$ = word being computed,
$t'$ = RNN being grabbed from)
3. Combinations are fed into a unidirectional RNN (top), and translated words are output (remember, its a `softmax` acros the vocabulary.)

**Model Details**  

![Attention Model - Detailed](/assets/study-notes/sequence-models/attention-mechanism/9.png)    
**Fig 9**
  
$$context^{<t>} = \sum_{t' = 0}^{T_x} \alpha^{<t,t'>}a^{<t'>}$$  

$$\alpha^{\langle t, t' \rangle} = 
\frac{\textrm{exp}(e^{\langle t, t' \rangle})}
{\sum_{t'=1}^{T_x} \textrm{exp}(e^{\langle t, t' \rangle})}$$

Notice that $\alpha^{\langle t, t' \rangle}$ is just a version of `softmax`, and they will all equal to 1.  
The $e^{\langle t, t' \rangle}$ term can be found with a small neural network:

![e term](/assets/study-notes/sequence-models/attention-mechanism/10.png)  
![Attention Model - Detailed](/assets/study-notes/sequence-models/attention-mechanism/11.png)  
**Fig 10, 11**    

---

More notes to come...