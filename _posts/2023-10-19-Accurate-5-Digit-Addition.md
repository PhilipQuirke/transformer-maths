---
title: "Accurate 5-Digit Addition in Transformers"
date: 2023-10-19
---
# Introduction
The <a href="{{site.url}}/2023/10/14/Understanding-Addition.html">previous post</a> discussed how a toy (1-layer, 3-head) transformer model answers integer addition questions like "33357+82243=". 

<img src="{{site.url}}/assets/AdditionQuestionAnswer.svg" style="display: block; margin: auto;" />

It summarised the model's addition algorithm like this and stated it could solve > 99.5% of integer addition questions:

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

The model can't reliably answer 0.5% of questions. 
All these questions are similar to the question 06665+03335=10000 where a "Carry 1" cascades through multiple digit columns (aka Cascading Use Sum 9 or cascading US9). 

This post documents how to improve that model to do 5-digit addition 100% accurately.

The same CoLab notepad as before can be downloaded from <a href="{{site.url}}/assets/Accurate_Addition_in_Transformers.ipynb">here</a>, 
and used to train and test the model. You can alter the code to try out other approaches.

# Failed Attempts
While we are **not** trying to "program" the model there are various "general purpose" changes that might help the model learn to handle the cascasding US9 case better.

These attempts all failed:
- Increasing the frequency of US9 cases in the training data (CoLab Part 3 setting ‘more_ms9_cases’) so the model has more examples of this "hard" case to learn from.
- Increasing the number of attention heads from 3 to 4 or 5 to provide more computing power.
- Changing the question format from “12345+22222=” to “12345+22222equals” to give the model more calculation steps after the question is revealed before it needs to state the first answer digit.

# Multiple Layers
Increasing n_layers from 1 to 2 doubles the number of attention heads available to the model. 
Also, the literature says a multiple-layer model can also “compose” the heads together in new ways to implement more complex calculations. 

Now our model has n_layers = 2, n_head = 3, n_training_steps = 7K, n_digits = 5.
- With n_layers = 1, the model can handle all Simple US9 cases but only 25% of Cascading US9 cases. With n_layers = 2, the model can handle all Cascading US9 cases (CoLab Part 9). Excellent. How does it do this? 
Ablating Early Steps
- With n_layers = 1, the model does not use steps 0 to 10 and step 17. With n_layers = 2, the model does not use steps 0 to 7 and step 17. So it is doing useful calculations in 3 more steps! What do these steps do?

Ablating the useful steps (in CoLab Part 11) negatively impacts accuracy in a clear pattern: 
- Layer 8 impacts mostly A4. Mostly impacts SimpleUS9 task.
- Layer 9 impacts mostly A3. Mostly impacts SimpleUS9 and CascadeUS9 tasks.
- Layer 10 impacts mostly A2. Mostly impacts SimpleUS9 and CascadeUS9 tasks.
- Layer 11 impacts A5 only. All tasks
- Layer 12 impacts A4 only. All tasks
- Layer 13 impacts A3 only. All tasks
- Layer 14 impacts A2 only. All tasks
- Layer 15 impacts A1 only. All tasks
- Layer 16 impacts A0 only. All tasks

- 

So the 2-layer model uses the steps 11 to 16 in the same way as the 1-layer model did. But it now uses steps 8 to 10 to do US9-related calculations for A2 to A4. (Recall that A0 and A1 are simpler calculations and don’t use the US9 task).
Attention Patterns (5 digits, 2 layers)
Here is the attention pattern for the sample question “16044+14973=031017=”:


# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl (Kiko) Barez and Esben Kran. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions that a talented novice can make progress on, and so contribute to the field.
