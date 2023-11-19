---
title: "Accurate 5-Digit Addition in Transformers (incomplete)"
date: 2023-10-19
---
## Introduction
This post is an incomplete work-in-progress. 

The <a href="{{site.url}}/2023/10/14/Understanding-Addition.html">previous post</a> discussed how a toy (1-layer, 3-head) transformer model answers integer addition questions like "33357+82243=". 

<img src="{{site.url}}/assets/AdditionQuestionAnswer.svg" style="display: block; margin: auto;" />

It summarised the model's addition algorithm like this and stated it could solve > 99.5% of integer addition questions:

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

The model can't reliably answer 0.5% of questions. 
All these questions are similar to the question 06665+03335=10000 where a "Carry 1" in one column "cascades" through the next few higher columns that each "sum to 9" (e.g. 6 = 3). We call this Cascading Use Sum 9 (aka US9). 

This post investigates whether we can improve that model to do integer addition 100% accurately.

The CoLab notepad for this blog can be downloaded from <a href="{{site.url}}/assets/Accurate_Addition_in_Transformers.ipynb">here</a>, 
and used to train and test the model. You can alter the code to try out other approaches.


## What didn’t work
We shouldn’t try to "program" or “teach” a model, but there are some standard "general purpose" changes that sometimes help make models more accurate.

For this model these approaches all failed to improve the model accuracy:
- Increasing the frequency of US9 cases in the training data so the model has more examples of this "hard" case to learn from (CoLab Part 3 setting ‘more_ms9_cases’).
- Increasing the number of attention heads from 3 to 4 or 5 to provide more computing power.
- Changing the question format from “12345+22222=” to “12345+22222equals” to give the model more calculation steps after the question is revealed before it needs to state the first answer digit.
- Changing the n_layers to 2 and n_heads to 2, increasing the number of attention heads from 3 to 4. Even with 30K training steps it was still inaccurate on A5 for some questions.


## Two layers improved accuracy
What worked was increasing the number of model “layers” (n_layers) from 1 to 2, while retaining n_heads = 3. This doubles the number of attention heads in the model from 3 to 6. 
Also, the literature says a multiple-layer model gains the ability to “compose” the attention heads together in new ways to implement more complex calculations.

With 2 layers the model definitely does better (based on CoLab with 20K training epochs, batch_size= 64, n_heads = 3, lr = 0.00008, weight_decay = 0.1):

<img src="{{site.url}}/assets/Addition_AccuracyByLayersDigits.png" style="display: block; margin: auto;" />

To be accurate, the 2 layer algorithm must have the functionality of the 1 layer algorithm **and** 
functionality to handle the 06665+03335=10000 case by cascading the Carry 1 through multiple columns. How does it do this? 

Why isn't the 2 layer algorithm 100% accurate? 


## Attention Pattern
Attention patterns are the easiest way to see the change in model algorithm by changing n_layers. With 1 layer, the attention pattern showed one row of 3 attention heads. With 2 layers, there are now 6 attention heads over two rows. For example, the question “16044+14973=” gives this attention pattern:

<img src="{{site.url}}/assets/AttentionPattern5Digit3Heads2Layer.png" style="display: block; margin: auto;" />


## Which steps do any useful calculations?
If we ablate alls head in a step, and the loss does not increase, then that step is **not** used by the algorithm, and can be excluded from further analysis.

CoLab Part 10 does this analysis and shows:
- n_digits = 5, n_layers = 1 :
  - The addition algorithm does not use any data generated in steps 0 to 10 inclusive. The model also does not use the last (17th) step. Therefore, the addition is started and completed in 6 steps (11 to 16)
- n_digits = 5, n_layers = 2 :
  - The addition algorithm does not use any data generated in steps 0 to 7 inclusive. The model also does not use the last (17th) step. Therefore, the addition is started and completed in 9 steps (8 to 16). What calculations get done in the extra 3 steps?


## Which steps impact which digits?
If we ablate all heads in each useful step to see if loss increases for specific **digits**, we gain insights. With 2 layers, CoLab Part 11A shows:
- Step 8 impacts A4 & A5 
- Step 9 impacts A3 & A5  
- Step 10 impacts A2, A3, A4 & A5
- Step 11 impacts A5 only 
- Step 12 impacts A4 only 
- Step 13 impacts A3 only 
- Step 14 impacts A2 only 
- Step 15 impacts A1 only 
- Step 16 impacts A0 only 

## Which steps+heads impact which use cases?
If we repeat this experiment but only test each class of question one at a time, we gain insights. With 2 layers:
- For BA questions, CoLab Part 11C shows:
  - S0 to S11 and S17 are not relevant
  - L1 is not relevant
  - L0H1 is not relevant.
- For UC questions, CoLab Part 11D shows:
  - S0 to S11 and S17 are not relevant
  - L1 is not relevant.
- For Simple US9 questions, CoLab Part 11E shows:
  - S0 to S7 and S17 are not relevant
  - L1 is not relevant.
- For Cascade US9 questions, CoLab Part 11F shows:
-   S0 to S7 and S17 are not relevant.

## Which steps+MLP layers impact which use cases?
If we ablate an MLP layer in a step, and the loss does not increase, then that MLP layer is **not** used by the algorithm, and can be excluded from further analysis. With 2 layers, CoLab Part 10C shows:

- In steps 0 .. 7, the MLP layers are not used
- In steps 8 .. 10, only the layer 0 MLP is used
- In steps 11 .. 16, both MLP layers are used and they strongly align to A5 .. A0 in successive steps. 

## Which heads + steps focus on which tokens?
By inspecting attention patterns we can see which token each head attends to in each step. But we are not sure if the model actually relies on the output of that neuron+step. Sometimes models train neurons to do calculations and then ignore their results.

Combining the attention pattern with information from the above sections, we get the following diagram. Any steps+heads or steps+MLP that are not used in the calculations are marked with an X. 

<img src="{{site.url}}/assets/StaircaseA3L2H3_Part1.svg" style="display: block; margin: auto;" />

## Hypothesis
The above is our base evidence from which to hypothesise about how the 2-layer algorithm works.

Our intuition:
- The extra 3 steps (8 to 11) appear to support the SimpleUS9 and CascadeUS9 calculations. Recall that A0 only needs BaseAdd, and A1 only needs BaseAdd and UseCarry1, so Use Sum 9 calculations is not relevant for them.
- The last 5 steps (12 to 16) do approximately the same calculations in 1 and 2 layer cases. 
- With 2 layers, the staircase is 2 tokens wide. With 1 layer, the staircase was 3 tokens wide and handled BaseAdd (perfectly), UseCarry1 (perfectly) and UseSum9 (imperfectly). Our intuition is that with 2 layers, the staircase handles BaseAdd and UseCarry1 but a new algorithm in steps 8 to 10 handles UseSum9 (with better accuracy than the 1 layer).

  
# Hypothesis #1
Given the 2 layer attention pattern’s similarity to 1 layer pattern, and the above evidence, our first hypothesis was that the 2 layer algorithm:
- Is based on the same operations (BA, MC, MS) as the 1 layer.
- Uses the new early steps to (somehow) do the US9 calculations with higher accuracy than the 1 layer model.
- The long double staircase still finalises each answer digit’s calculation.
- The two attention nodes in the long double staircase steps do the BA and MC calculations and pull in US9 information calculated in the early steps.

If this is correct then the 2 layer algorithm successfully completes these calculations:
- A0 = D0.BA
- A1 = D1.BA + D0.MC
- A2 = D2.BA + (D1.MC or D1.MS & D0.MC)
- A3 = D3.BA + (D2.MC or D2.MS & D1.MC or D2.MS & D1.MS & D0.MC)
- A4 = D4.BA + (D3.MC or D3.MS & D2.MC or D3.MS & D2.MS & D1.MC or D3.MS & D2.MS & D1.MS & D0.MC)
- A5 = D4.MC or D4.MS & D3.MC or D4.MS & D3.MS & D2.MC or D4.MS & D3.MS & D2.MS & D1.MC or D4.MS & D3.MS & D2.MS & D1.MS & D0.MC

Looking at the above diagram and thinking about the A5 and A4 calculations, some thoughtss:
- Answer digit A5 is the earliest-revalued answer. It is always 0 or 1. As it is revealed in step 11, it must be calculated in steps 8 to 11. There are only 7 useful steps+heads (plus MLP layers) in these steps to do the above A5 calculation in.
- Answer digit A4 is revealed one step later, and only has another 5 useful steps+heads (plus MLP layer) to do the above A4 calculation in.
- Reusing the (hypothetical) A5 sub-calculations as part of the A4 sub-calculations looks complex.  
Our intuition is that there are not enough useful heads+steps in steps 8 to 11 to do the A5 and A4 calculations. So hypothesis #1 is incorrect.


# Hypothesis #2
Our second hypothesis was that the 2 layer algorithm:
- Has a **more compact** data representation (That is, it does not store BA, MC1 and US9 data as separate datums.)
- Can therefore pack more calculations into each head+layer in steps 8 to 11 (so it can calculate A5 in time).
- Has a data representation that allows re-use of some A5 sub-calculations in the A4-calculation (so it can calculate A4 in time). 

In hypothesis #2, we assume the model stores the sum of each digit pair as a single token in the range “0” to “18” (covering 0+0 to 9+9). We name this operator Dn.T1 - where T stands for “token addition” (and the 1 will be explained later):
- Dn.T1 = Dn + Dn’

The T1 operation does not understand mathematical addition. The model implements the T1 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “8” + “7” = “15”. There are 100 distinct mappings:
<img src="{{site.url}}/assets/Addition_T1Pairs.png" style="display: block; margin: auto;" />

T1 is a compact way to store data. If it needs to, the model can implement a bigram mapping to convert a T1 value into a BA, MC or MS value:
<img src="{{site.url}}/assets/Addition_T1Bigtrm.png" style="display: block; margin: auto;" />

Our notation shorthand for these "conversion" bigram mappings is:
- Dn.BA = (Dn.T1 % 10) where % is the modulus operator 
- Dn.MC = (Dn.T1 // 10) where // is the integer division operator
- Dn.MS = (Dn.T1 == 9) where == is the equality operator

Excluding D0.T1, the value Dn.T1 is not perfectly accurate because it is constrained to information from just one digit - hence the “1” in T1. We define another more accurate operator Dn.T2 that has “two-digit accuracy”. Dn.T2 is the pair sum for the nth digit plus the carry 1 (if any) from the n-1th digit T1:
- Dn.T2 = Dn.T1 + ( Dn-1.T1 // 10 )

Dn.T2 is more accurate than DnT1. The Dn.T2 value is always in the range “0” to “19” (covering 0+0+0 to 9+9+Carry1). The model can implement the T2 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “12” + “1” = “13”. There are 38 distinct mappings: 

<img src="{{site.url}}/assets/Addition_T2Mappings.png" style="display: block; margin: auto;" />

Dn.T2 can only be calculated after Dn.T1 and Dn-1.T1 have been calculated. 

We define operators Dn.T3, Dn.T4 & Dn.T5 each with higher accuracy:
- Dn.T3 = Dn.T1 + ( Dn-1.T2 // 10 )	Three-digit accuracy
- Dn.T4 = Dn.T1 + ( Dn-1.T3 // 10 )	Four-digit accuracy
- Dn.T5 = Dn.T1 + ( Dn-1.T4 // 10 )	Five-digit accuracy

The value D4.T5 is perfectly accurate as it integrates MC1 and cascading MS9 data all the way back to and including D0.T1. The values D1.T2, D2.T3, D3.T4 are also perfectly accurate. If we know these values we can calculate answer digits with perfect accuracy:
- D1.T2 % 10 gives A1 with zero loss
- D2.T3 % 10 gives A2 with zero loss
- D3.T4 % 10 gives A3 with zero loss
- D4.T5 % 10 gives A4 with zero loss
- D4.T5 // 10 gives A5 with zero loss

For the model to do integer addition perfectly accurately, it must be calculating D4.T5 by step 11 so an accurate A5 can be revealed. Ablation tests tell us which steps+heads are doing useful calculations (but not what those steps actually do). Hypothesis #2 says the model uses the useful Step+Head to perform these operations so that D4.T5 is calculated in step 11:
- Step 8:
  - L0H1: D2 focus: Calculate D2.T1 = D2 + D2’
  - L0H2: D3 focus: Calculate D3.T1 = D3 + D3’
  - L0MLP: A4 focus: Use is not understood. Could calculate inaccurate D3.T2 = D3.T1 + D2.T1 // 10
- Step 9
  - L0H1: D1 focus: Calculate D1:T1 = D1 + D1’
  - L0MLP: A3 focus: Use is not understood. Could calculate inaccurate D2.T2 = D2.T1 + D1.T1 // 10
- Step 10
  - L0H0: D0 focus: Calculate D1.T2 = D1.T1 + (D0 + D0’) // 10. Perfectly accurate.
  - L0H1: D1 focus: Use is not understood. Duplicate of S9.L0H1? 
  - L0H2: D1 focus: Use is not understood. Duplicate of S9.L0H1? 
  - L0MLP: ~A2 focus: Calculate D2.T3 = D2.T1 + D1.T2 // 10. Perfectly accurate  
- Step 11:
  - L0H1: D3 focus: Calculate D3.T4 = D3.T1 + D2.T3. Perfectly accurate. 
  - L0H2: D4 focus: Calculate D4.T1 = D4 + D4’
  - L0MLP: A5 focus: Calculate D3.MC = D3.T4 // 10  
  - L1MLP: A5 focus: Calculate D4.T5 // 10 = (D4.T1 + D3.MC) // 10. Perfectly accurate A5.
- Step 12:
  - L0H0: D4 focus: Calculate D4.T5 = D4.T1 + D3.T4 // 10. Perfectly accurate.
  - L0H1: D3 focus: Use is not understood. Could calculate accurate D3.BA = D3.T4 % 10 ? 
  - L0H2: D4 focus: Use is not understood. Could calculate accurate D4.BA = D4.T5 % 10 ? 
  - L0MLP: A4 focus: Calculate D4.T5 % 10. Perfectly accurate A4
  - L1H0: =: Use is not understood. Could calculate accurate D2.BA = D2.T3 % 10 ? TBA
  - L1H1: =: Use is not understood. Could calculate accurate D1.BA = D1.T2 % 10 ? TBA
  - L1MLP: A4 focus: Use is not understood.

Representing the above as a diagram:
<img src="{{site.url}}/assets/StaircaseA3L2H3_Part2.svg" style="display: block; margin: auto;" />

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
TBA


# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl Barez and Esben Kran. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions that a talented novice can make progress on, and so contribute to the field.
