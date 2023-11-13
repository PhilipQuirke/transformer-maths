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

<img src="{{site.url}}/assets/StaircaseA3L2_Part1.svg" style="display: block; margin: auto;" />


# Hypothesis #1
Given the 2 layer attention pattern’s similarity to 1 layer pattern, and the above evidence, our first hypothesis was that the 2 layer algorithm:
- Has higher accuracy than the 1 layer algorithm.
- Is based on the same operations (BA, MC, MS) as the 1 layer.
- Uses the new early steps to (somehow) do the US9 calculations with higher accuracy than the 1 layer model.
- The long double staircase still finalises each answer digit’s calculation.
-The two attention nodes in the long double staircase steps do the BA and MC calculations and pull in US9 information calculated in the early steps.

If this is correct then the 2 layer algorithm successfully completes these calculations:
- A0 = D0.BA
- A1 = D1.BA + D0.MC
- A2 = D2.BA + (D1.MC or D1.MS & D0.MC)
- A3 = D3.BA + (D2.MC or D2.MS & D1.MC or D2.MS & D1.MS & D0.MC)
- A4 = D4.BA + (D3.MC or D3.MS & D2.MC or D3.MS & D2.MS & D1.MC or D3.MS & D2.MS & D1.MS & D0.MC)
- A5 = D4.MC or D4.MS & D3.MC or D4.MS & D3.MS & D2.MC or D4.MS & D3.MS & D2.MS & D1.MC or D4.MS & D3.MS & D2.MS & D1.MS & D0.MC

Answer digit A5 is the earliest-revalued answer. It is always 0 or 1. It must be calculated in steps 8 to 11, and revealed in step 11. Analysing the attention pattern, it turns out there are not enough active heads+layers in steps 8 to 11 to do all the parts of A5 calculation if the BA, MC and MS state data are calculated independently. So hypothesis #1 is incorrect.

# Hypothesis #2
To calculate A5 by step 11, the model must be packing more calculation into each head+layer in steps 8 to 11. A more compact representation of data would allow this. 

In hypothesis #2, we assume the model stores the sum of each digit pair as a single token in the range “0” to “18” (covering 0+0 to 9+9). We name this operator Dn.T1 - where T stands for “token addition” (and the 1 will be explained later):
- Dn.T1 = Dn + Dn’

The T1 operation does not understand mathematical addition. The model implements the T1 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “8” + “7” = “15”. There are 100 distinct mappings:
<img src="{{site.url}}/assets/Addition_T1Pairs.png" style="display: block; margin: auto;" />

We can retrieve the operator BA, MC & MS values from a Dn.T1 value as follows:
- Dn.BA = (Dn.T1 % 10) where % is the modulus operator 
- Dn.MC = (Dn.T1 // 10) where // is the integer division operator
- Dn.MS = (Dn.T1 == 9) where == is the equality operator

So T1 is a compact way to store intermediate results that still allows us to reason using the traditional BA, MC & MS operators. 

The Dn.T1 accuracy is imperfect because it is constrained to information from just one digit - hence the “1” in T1. We define another more accurate operator Dn.T2 that has “two-digit accuracy”. Dn.T2 is the pair sum for the nth digit plus the carry 1 (if any) from the n-1th digit T1:
- Dn.T2 = Dn.T1 + ( Dn-1.T1 // 10 )

Dn.T2 is more accurate than DnT1. The Dn.T2 value is always in the range “0” to “19” (covering 0+0+0 to 9+9+Carry1). This operation does not understand mathematical addition. The model implements the T2 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “12” + “1” = “13”. There are 38 distinct mappings: 
<img src="{{site.url}}/assets/Addition_T2mappings.png" style="display: block; margin: auto;" />

Dn.T2 can only be calculated after Dn.T1 and Dn-1.T1 have been calculated. 

We define operators Dn.T3, Dn.T4 & Dn.T5 each with higher accuracy::
- Dn.T3 = Dn.T1 + ( Dn-1.T2 // 10 )	Three-digit accuracy
- Dn.T4 = Dn.T1 + ( Dn-1.T3 // 10 )	Four-digit accuracy
- Dn.T5 = Dn.T1 + ( Dn-1.T4 // 10 )	Five-digit accuracy

The value D4.T5 is perfectly accurate as it integrates MC1 and cascading MS9 data all the way back to and including D0.T1. The values D1.T2, D2.T3, D3.T4 are also perfectly accurate. If we know these values we can calculate answer digits with perfect accuracy:
- D1.T2 % 10 gives A1
- D2.T3 % 10 gives A2
- D3.T4 % 10 gives A3
- D4.T5 % 10 gives A4
- D4.T5 // 10 gives A5

If the model is doing integer addition perfectly accurately, then it must be calculating D4.T5 by step 11 so an accurate A5 is revealed. Ablation tests tell us which steps+heads are doing useful calculations (but not what those steps actually do). Hypothesis #2 says the model uses the useful Step+Head to perform these operations so that D4.T5 is calculated in step 11:
- Step 8:
  - L0H1: D2: Calculate D2.T1 = D2 + D2’
  - L0H2: D3: Calculate D3:T1 = D3 + D3’
  - MLP: N/A: Not used. Could calculate inaccurate D3.T2 = D3.T1 + D2.T1 // 10
- Step 9
  - L0H1: D1: Calculate D1:T1 = D1 + D1’
  - MLP: N/A: Not used. Could calculate inaccurate D2.T2 = D2.T1 + D1.T1 // 10
- Step 10
  - L0H0: D0: Calculate D1.T2 = D1.T1 + (D0 + D0’) // 10. Perfectly accurate.
  - L0H1: D1: Not used. Duplicate of S9.L0H1?
  - L0H2: D1: Not used. Duplicate of S9.L0H1?
  - MLP: N/A: Calculate D2.T3 = D2.T1 + D1.T2 // 10. Perfectly accurate  
- Step 11:
  - L0H1: D3 : Calculate D3.T4 = D3.T1 + D2.T3. Perfectly accurate. 
  - L0H2: D4 : Calculate D4.T1
  - MLP: N/A: Calculate D4.T5 // 10 = (D4.T1 + D3.T4 // 10) // 10. Perfectly accurate A5.
- Step 12:
  - L0H0: D4 : Calculate D4.T5 = D4.T1 + D3.T4 // 10. Perfectly accurate.
  - L0H1: D3 : Not used. Could calculate accurate D3.BA = D3.T4 % 10
  - L0H2: D4 : Not used. Could calculate accurate D4.BA = D4.T5 % 10
  - MLP: N/A: Calculate D4.T5 % 10. Perfectly accurate A4
  - L1H0: =: Not used. Could calculate accurate D2.BA = D2.T3 % 10
  - L1H1: =:  Not used. Could calculate accurate D1.BA = D1.T2 % 10
  - MLP: N/A: Not used. 

Some notes :
- Possible issue: Ablation says two step 10 heads are useful, but they have are not used. 
  - Possible solution: These heads may be “duplicates” of S9.L0H1, splitting the workload
- Possible issue: To get a perfect A5, all the digits have been completed by step 11. Why does the model retain the redundant long staircase BA calculations?
  - Possible solution: The model is not optimising for compactness. Once it gives the right numeric answer consistently it stops optimising.
- Possible issue: Ablation says the layer 2 heads S12.L1H0 and S12.L1H1 are useful, but they have are not used. That is, no Layer 1 heads are used. This seems wrong.
  - Possible solution: The model is not optimising for algorithm compactness. The calculation of A4 may be spread over several heads  
- Possible issue: The calculation by S11.MLP of D4.T5 // 10 = (D4.T1 + D3.T4 // 10) // 10 seems complex. Can this calc be done by the MLP?
  - Possible solution: The D4.T1 and D3.T4 values are in the residual stream. Is this a trigram? TBA


# Pulling it all together (TBD)
As the 2 layer model is 100% accurate, the algorithm for the model must be able to handle a cascading US9 question such as 66665+33335=100000. What algorithm can handle this? 

Draft / incorrect algorithm is:

TBC

# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl Barez and Esben Kran. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions that a talented novice can make progress on, and so contribute to the field.
