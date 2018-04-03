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

More notes to come...