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
All these questions are similar to the question 06665+03335=10000 where a "Carry 1" in one column "cascades" through the next few higher columns that each "sum to 9" (e.g. 6 = 3). We call this Cascading Use Sum 9 (aka US9). 

This post documents how to improve that model to do 5-digit addition 100% accurately.

The CoLab notepad for this blog can be downloaded from <a href="{{site.url}}/assets/Accurate_Addition_in_Transformers.ipynb">here</a>, 
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
the model can handle all Simple and Cascading US9 cases (CoLab Part 9) for 5 digit addition. Excellent! 

To be accurate, the 2 layer algorithm must have the functionality of the 1 layer algorithm **and** 
functionality to handle the 06665+03335=10000 case by cascading the Carry 1 through multiple columns. How does it do this? 


# Attention Pattern
Attention patterns are the easiest way to see the change in model structure by changing n_layers. With 1 layer, the attention pattern showed one row of 3 attention heads. With 2 layers, there are now 6 attention heads over two rows. For example, for the question “16044+14973=” we get this attention pattern:

<img src="{{site.url}}/assets/AttentionPattern5Digit3Heads2Layer.png" style="display: block; margin: auto;" />

# Which steps do any useful calculations?
If we ablate all heads in each step and see if loss increases we can show which steps (if any) are **not** used by the algorithm. These steps can be excluded from further analysis.

CoLab Part 10 does this and shows:
- n_digits = 5, n_layers = 1 :
  - The addition algorithm does not use any data generated in steps 0 to 10 inclusive. The model also does not use the last (17th) step. Therefore, the addition is started and completed in 6 steps (11 to 16)
- n_digits = 5, n_layers = 2 :
  - The addition algorithm does not use any data generated in steps 0 to 7 inclusive. The model also does not use the last (17th) step. Therefore, the addition is started and completed in 9 steps (8 to 16). What calculations get done in the extra 3 steps?


# Which steps impact which digits and tasks?
Here we ablate all heads in each step and see if loss increases for specific **digits** and **tasks**. This shows which steps are associated with calculating which digits and tasks.

CoLab Part 11A does this and shows:
- n_digits = 5, n_layers = 1 :
  - Step 12 impacts A4 only. All tasks
  - Step 13 impacts A3 only. All tasks
  - Step 14 impacts A2 only. All tasks
  - Step 15 impacts A1 only. All tasks
  - Step 16 impacts A0 only. All tasks
- n_digits = 5, n_layers = 2 :
  - Step 8 impacts mostly A4. Mostly SimpleUS9.
  - Step 9 impacts mostly A3. Mostly SimpleUS9 and CascadeUS9.
  - Step 10 impacts mostly A2. Mostly SimpleUS9 and CascadeUS9.
  - Step 11 impacts A5 only. All tasks
  - Step 12 impacts A4 only. All tasks
  - Step 13 impacts A3 only. All tasks
  - Step 14 impacts A2 only. All tasks
  - Step 15 impacts A1 only. All tasks
  - Step 16 impacts A0 only. All tasks

Some notes:
- The extra 3 steps (8 to 11) appear to support the SimpleUS9 and CascadeUS9 calculations. Recall that A0 only needs BaseAdd, and A1 only needs BaseAdd and UseCarry1, so we won’t see any Use Sum 9 calculations for them.
- The last 5 steps (12 to 16) do approximately the same calculations in the 5 and 10 digit cases. 
- With 2 layers, the staircase is 2 tokens wide. With 1 layer, the staircase was 3 tokens wide and handled BaseAdd (perfectly), UseCarry1 (perfectly) and UseSum9 (imperfectly). Our intuition is that with 2 layers, the staircase handles BaseAdd and UseCarry1 but some other mechanism handles UseSum9 (perfectly).


# Which heads + steps impact which digits and tasks?
By studying attention patterns we can see which token each head attentions to in each step. But we are not sure if the model actually relies on the output of that neuron+step. Sometimes models train neurons to do calculations and then ignore their results.

CoLab Part 11B ablates **each** head in each step and see if loss increases for specific **digits** and **tasks**. This shows which steps are associated with calculating which digits and tasks. Any head+step not used in the calculations is marked with an X. 

<img src="{{site.url}}/assets/StaircaseA3L2_Summary.svg" style="display: block; margin: auto;" />


# Which heads + steps impact which BaseAdd?
Now we know all the heads+steps that are needed for the calculations, we can just study the remaining heads+steps.

We can studying simple (aka BaseAdd) questions like 12345+33333= that do not need UseCarry1 or UseSum9. CoLab Part 11B ablates **each** head in each step for a batch of BaseAdd questions, showing which heads+steps are involved in the BaseAdd calculations.









# Pulling it all together (TBD)
As the 2 layer model is 100% accurate, the algorithm for the model must be able to handle a cascading US9 question such as 66665+33335=100000. What algorithm can handle this? 

Draft / incorrect algorithm is:

<img src="{{site.url}}/assets/StaircaseA3L2_Detailed.svg" style="display: block; margin: auto;" />

TBC

# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl Barez and Esben Kran. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions that a talented novice can make progress on, and so contribute to the field.
