---
title: "Accurate 5-Digit Addition in Transformers (incomplete)"
date: 2023-10-19
---
# Introduction
This post is an incomplete work-in-progress. 

The <a href="{{site.url}}/2023/10/14/Understanding-Addition.html">previous post</a> discussed how a toy (1-layer, 3-head) transformer model answers integer addition questions like "33357+82243=". 

<img src="{{site.url}}/assets/AdditionQuestionAnswer.svg" style="display: block; margin: auto;" />

It summarised the model's addition algorithm like this and stated it could solve > 99.5% of integer addition questions:

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

The model can't reliably answer 0.5% of questions. 
All these questions are similar to the question 06665+03335=10000 where a "Carry 1" cascades through multiple digit columns (aka Cascading Use Sum 9 or cascading US9). 

This post documents how to improve that model to do 5-digit addition 100% accurately.

The CoLab notepad for this blog can be downloaded fromm <a href="{{site.url}}/assets/Accurate_Addition_in_Transformers.ipynb">here</a>, 
and used to train and test the model. You can alter the code to try out other approaches.

# What didn’t work
We shouldn’t try to "program" or “teach” a model, but there are some standard "general purpose" changes that sometimes help make models more accurate.

For this model these approaches all failed to improve the model accuracy:
- Increasing the frequency of US9 cases in the training data so the model has more examples of this "hard" case to learn from (CoLab Part 3 setting ‘more_ms9_cases’).
- Increasing the number of attention heads from 3 to 4 or 5 to provide more computing power.
- Changing the question format from “12345+22222=” to “12345+22222equals” to give the model more calculation steps after the question is revealed before it needs to state the first answer digit.

# Two layers improved accuracy
What worked was increasing the number of model “layers” (n_layers) from 1 to 2. This doubles the number of attention heads in the model. 
Also, the literature says a multiple-layer model gains the ability to“compose” the attention heads together in new ways to implement more complex calculations.

With 1 layer, the model can handle all Simple US9 cases but only 25% of Cascading US9 cases. With 2 layers, 
the model can handle all Simple and Cascading US9 cases (CoLab Part 9) for 5 digit addition. Excellent! How does it do this? 

# Attention Pattern
Attention patterns are the easiest way to see the impact of changing n_layers. With 1 layer, the attention pattern showed one row of 3 attention heads. With 2 layers, there are now 6 attention heads over two rows. For example, for the question “16044+14973=” we get this attention pattern:

<img src="{{site.url}}/assets/AttentionPattern5Digit3Heads2Layer.png" style="display: block; margin: auto;" />

# Which Steps do useful calculations?
With 1 layer, (CoLab Part 11 shows) the model does not use steps 0 to 10 and step 17 to do useful calculations. So all calculations are done in the 6 steps 11 to 16.

With 2 layers, the model does not use steps 0 to 7 and step 17. So it is doing useful calculations in 10 steps! What calculations do these 10 steps perform?

# Double-staircases similarities
Both the 1 layer and 2 layer attention patterns contain a double-staircase, late in the steps, with the same length. Maybe they do the same job? 

Ablating the last 6 useful steps (in CoLab Part 11) negatively impacts accuracy, in exactly the same way as in the 1 layer model, supporting this intuition:
- Step 11 impacts A5 only. Impacts all (i.e. BA, UC1, US9) tasks.
- Step 12 impacts A4 only. Impacts all tasks.
- Step 13 impacts A3 only. Impacts all tasks.
- Step 14 impacts A2 only. Impacts all tasks.
- Step 15 impacts A1 only. Impacts all tasks.
- Step 16 impacts A0 only. Impacts all tasks.

One difference: With 2 layers, the staircase is 2 tokens wide. With 1 layer, the staircase was 3 tokens wide and handled BaseAdd (perfectly), UseCarry1 (perfectly) and UseSum9 (imperfectly). Our intuition is that with 2 layers, the staircase handles BaseAdd and UseCarry1 but some other mechanism handles UseSum9 (perfectly).

#Extra steps do UseSum9-related calculations
What calculations does the 2 layer model do in steps 8 to 10 (where the 1 layer model does nothing)? 

Ablating these steps (in CoLab Part 11) negatively impacts accuracy in a clear pattern:
- Step 8 impacts mostly A4. Mostly impacts Simple US9 task.
- Step 9 impacts mostly A3. Mostly impacts Simple and Cascading US9 tasks.
- Step 10 impacts mostly A2. Mostly impacts Simple and Cascading US9 tasks.
- So with 2 layers, the model uses steps 8 to 10 to do US9-related calculations for A2 to A4. 

Recall that A0 only needs BaseAdd, and A1 only needs BaseAdd and UseCarry1, so we won’t see any Use Sum 9 calculations for them.

# Pulling it all together
As the 2 layer model is 100% accurate, the algorithm for the model must be able to handle a cascading US9 question such as 66665+33335=100000. What algorithm can handle this? 

Draft / incorrect algorithm is:

<img src="{{site.url}}/assets/StaircaseA3L2_Detailed.svg" style="display: block; margin: auto;" />

TBC

# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl (Kiko) Barez and Esben Kran. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions that a talented novice can make progress on, and so contribute to the field.
