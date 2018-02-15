---
template: post
title: Study Notes - Sequence Models
---

# {{page.title}}

My course notes for the [Coursera class](https://www.coursera.org/learn/nlp-sequence-models/).

## Recursive Neural Networks
---
### Notation:

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
Dictionaries are usually large in practice; at least 10,000 words.

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

---

### Recurrent Neural Network Model