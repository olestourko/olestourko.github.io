---
layout: default
title: Sequence Models
---

# {{page.title}}

My course notes for the [Coursera class](https://www.coursera.org/learn/nlp-sequence-models).

## Recurrent Neural Networks
---
### Notation

For the sequence, which happens to be a string (but can also be some other sequential data),
with labelled city names:

**_The city of Toronto_**
  
$x: x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, x^{\langle 3 \rangle}, x^{\langle 4 \rangle}$  
$y: [0, 0, 0, 1]$  

$Tx$ = length of sequence (4 for above)  
$Tx^{(i)}$ = sequence length for $i^{th}$ training example  

$x^{(i)\langle t \rangle}$ means the $t^{th}$ element of the $i^{th}$ training example. 

---

**Q:** But how do you represent words?  
**A:** Use a _dictionary_ of words:


| **Word** | **Position** |
|:---|:---|
| a | 1 |
| and  | 2 |
| city | 3 |
| Detroit | 4 |
| of | 5 |
| the | 6 |
| Toronto | 7 |

<br>
Each words then gets mapped to a one-hot representation based on its position in the dictionary.
For example, the word **_Toronto_** is represented as:

$\begin{bmatrix}
0 \\\
0 \\\
0 \\\
0 \\\
0 \\\
0 \\\
1 \\\
\end{bmatrix}$

Dictionaries are usually large in practice; at least 10,000 words.

---

### Recurrent Neural Network model

**Q:** Why not just use a standard neural network?  
**A:** Because feature vectors can be of different lengths (for example, strings of words). It is possible to
pad shorter feature vectors, but this is a poor solution. A standard neural network also won't share share features learned across diffirent positions of text.

In a RNN, the activation value of the previous example is fed into the current example: 

![Neural Network Layout](/assets/study-notes/sequence-models/0.png)

$a^{\langle 0 \rangle}$ is typically initialized to be all 0's.  

$a^{\langle t \rangle} = g(W_{aa} a^{\langle t-1 \rangle} \ W_{ax} x^{\langle t \rangle} + b_a)$ usually `tanh` or `relu`  
$\hat{y}^{\langle i \rangle} = g(W_{ya} a^{\langle t \rangle} + b_y)$ usually `softmax` or `sigmoid`.  

Where  

$a$ is the current activation (like a word in a string).    
$W_{aa}$ are the weights for input feature $a^{t-1}$.   
$W_{ax}$ are the weights for input feature $x^{t}$.  
$W_{ya}$ are the weights for $a^{\langle t \rangle}$  

The weight parameters are reused; they don't change between steps.

---

The equation for $a^{\langle t \rangle}$ can also be written as $g(W_a[a^{\langle t-1 \rangle}, x^{\langle t \rangle}])$.  

This means that you can stack the matrices $W_{aa}$ and $W_{ax}$ together to make $W_a$.  
For example, if $W_aa$ is 100x100 and $W_ax$ is 100x10,000, then $W_a$ will be 100x10,100. 

The $[a^{\langle t-1 \rangle}, x^{\langle t \rangle}]$ means you stack the vectors together:  
$[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] = \begin{bmatrix}
a^{\langle t-1 \rangle} \\\
x^{\langle t \rangle}
\end{bmatrix}
$

This alternative representation works because the matrix multiplication ends up doing the same thing as the original
(good explanation [here](https://www.coursera.org/learn/nlp-sequence-models/lecture/ftkzt/recurrent-neural-network-model) at 15:00)

---

### How does back-propagation work?

Forward-propagation looks like this, with the $a^{\langle t \rangle}$'s computed before the $\hat{y}^{\langle t \rangle}$'s:

![Forward-propagation](/assets/study-notes/sequence-models/1.png)  

To do back-propagation we do the reverse, and use the standard logitic regression loss function (aka cross-entropy loss):

$$\mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) \
= \
-y^{\langle t \rangle} log\space\hat{y}^{\langle t \rangle} - (1 - y^{\langle t \rangle}) \space log (1 - \hat{y}^{\langle t \rangle})$$

The total loss is:  

$$\mathcal{L}(y, \hat{y}) = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})$$

---

### Types of RNNs

The input length ($T_x$) isn't always the same as the output length ($T_y$), so other types of RNNs are available.

|:---|:---|
| ![One to one](/assets/study-notes/sequence-models/one-to-one.png)<br>One to one  | ![One to many](/assets/study-notes/sequence-models/one-to-many.png)<br>One to many|
| ![Many to one](/assets/study-notes/sequence-models/many-to-one.png)<br>Many to one | ![Many to many](/assets/study-notes/sequence-models/many-to-many.png)<br>Many to many |
| ![Many to many (encoder/decoder)](/assets/study-notes/sequence-models/many-to-many-encoder-decoder.png)<br>Many to many (encoder/decoder) |  |

---

### NLP: Language model and sequence generation

Consider a speech recognition - what is the correct sentence?

**_The apple and pair salad_**  
**_The apple and pear salad_**  

They both sound the same, but only the 2nd makes sense. A language model will give the first sentence a higher probability of being correct.
**A language model tells you the probabiltiy of any given sentence.** Specifically, what is the chance that the next sentence it reads will be the given one?  

Input: a sentence, written as a sequence $y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., y^{\langle T_y \rangle}$  
Output: $P(y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., y^{\langle T_y \rangle})$

Use a large body of text as the training set. Tokenize it by mapping each word to a one-hot vector or index, as described above in the "Notation" section.
Using something to denote the end of the sentence is useful too (`<EOS>`), but not absolutely necessary.

**_A sample sentence._** `<EOS>`  
$y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, y^{\langle 3 \rangle}, y^{\langle 4 \rangle}$

**Q:** What happens if a word is not in the dictionary?  
**A:** Have a token available for unknown words.

---

![RNN Model](/assets/study-notes/sequence-models/2.png)

- $a$ makes a `softmax` prediction about what $^{\langle t \rangle}$ will be - the probability for _every_ word in the dictionary.  
- $x^{\langle 2 \rangle} = y^{\langle 1 \rangle}$ : the previous correct word is fed in as the feature for the current step.

To train this RNN, use the cost function  

$$\mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) \
= \
-\sum_i y_i^{\langle t \rangle} \space log \space \hat{y}_i^{\langle t \rangle}$$ ([softmax loss function](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/))  
$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})$$

---

### Sampling novel sequences
An RNN can be used to generate new text. Instead of feeding $y^{\langle t \rangle}$ into $x^{\langle t+1 \rangle}$, you can feed
$\hat{y}^{\langle t \rangle}$ into $x^{\langle t+1 \rangle}$. $x^{\langle t+1 \rangle}$ is a softmax distribution (a probability for each word), so you can use `np.random.choice` to choose a word.

![Sampling](/assets/study-notes/sequence-models/3.png)  

---

### Vanishing gradients &amp; the  Gated Recurrent Unit

Vanishing gradients make long-range dependencies are difficult to learn. 

**_The cat, which already ate... was full_** vs. **_The cats, which already ate... were full_**

_Exploding_ gradients are easily fixed through
gradient clamping, but vanishing gradients are fixed by something called a Gated Recurrent Unit:

$c$ = memory unit  
$c^{\langle t \rangle} = a^{\langle t \rangle}$    
$\tilde{c}^{\langle t \rangle} = tanh(W_c[c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$ (the "candidate value")    
$\Gamma_u = sigmoid(W_u[c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u)$ (the "update gate")  

The purpose of the update gate is to determine when $c$ should be updated.

$c^{\langle t \rangle} = \Gamma_u * \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) * c^{\langle t-1 \rangle}$  
$c^{\langle t \rangle}$ and $\tilde{c}^{\langle t \rangle}$, $\Gamma_u$ can be vectors, so the $*$ operations are element-wise multiplication. 

This all works because $\Gamma_u$ usually stays very close to 0, until it needs to update.  
Full explanation [here](https://www.coursera.org/learn/nlp-sequence-models/lecture/agZiL/gated-recurrent-unit-gru). Note that this is a simplified
GRU, the Coursera video also describes a modified full GRU.

---

### Long Short Term Memory (LSTM)

A more powerful and more general version of the GRU.  

$\tilde{c}^{\langle t \rangle} = tanh(W_c[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$ (note the direct use of $a$ instead of $c$)   
$\Gamma_u = sigmoid(W_u[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u)$  
$\Gamma_f = sigmoid(W_f[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f)$ (the "forget" gate)  
$\Gamma_o = sigmoid(W_o[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f)$ (the "output" gate)  
$c^{\langle t \rangle} = \Gamma_u * \tilde{c}^{\langle t \rangle} + \Gamma_f * c^{\langle t-1 \rangle}$  
$a^{\langle t \rangle} = \Gamma_o * tanh \space c^{\langle t \rangle}$  

![LSTM Diagram](/assets/study-notes/sequence-models/4.png)