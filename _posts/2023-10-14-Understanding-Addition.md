---
title: "Understanding Addition in Transformers"
date: 2023-10-14
---
# Introduction
If you type the question “When Mary and John went to the store, John gave a drink to” into Chat GPT, it will answer “Mary”.  
Chat GPT is a “Transformer” model that considers each word in the question and generates a predicted answer. 
You can also type in “12345+86764=” and it will give the right answer. How does it do this?

This blog explains how a toy (1-layer, 3-head) transformer model answers integer addition questions like:

<img src="{{site.url}}/assets/AdditionQuestionAnswer.svg" style="display: block; margin: auto;" />

This blog is written as an introduction to Mechanistic Interpretability and Transformer models for novices. 
It covers our investigation, testing and results of integer addition in transformers, building up section by section, and finally explaining this diagram:

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

A CoLab notepad is provided here. 
It contains all the code needed to train the model and use the trained model, create graphs, etc. You can alter the code to test out other approaches.

# Humans vs Model Learning
When we were learning to do addition, we likely memorized some facts (e.g. 1+1=2) but quickly learnt this was not scalable and then learnt the standard way to do addition. 

Transformer models are trained by us providing them with many example questions and scoring the correctness of their answer. Initially their answers are random, but over the training they discover (by themselves) ways to do addition accurately (aka with low loss).

We might expect the model to use the human approach to addition: adding first the lowest-value digit-pair (D0 + D0’), noting whether this sum generated a “Carry 1”, then adding D1 + D1’ + D0’s Carry 1 (if any), etc. This approach is the easiest for us, but it is very sequential (D0 then D1 etc) with a strong time-ordering.

As our model trains, it tries many possibly-useful approaches to many addition sub-tasks (e.g. D3 + D3’) in parallel. There is no overall coordination or time-ordering. The learning is more like evolution - using the answer scoring to prefer one approach over another. The below graph (from the CoLab Part 5) shows our model training over time - getting better at adding each of D0 to D4 digits - learning at a different speed for each digit. After training, the model has learnt a different way to do addition than humans.

<img src="{{site.url}}/assets/Addition5DigitTrainingLoss.png" style="display: block; margin: auto;" />

For 5 digit addition there are 10 billion possible questions, and the model gets accurate after being trained on 2 million questions, so the model isn’t using memorisation. What is it doing?

# Jigsaw 
We need to discard our preconceptions about the best way to do addition. Alternative approaches are feasible.

As a warm up exercise, consider how you do a jigsaw puzzle. You might use a combination of meta knowledge of the problem (placing edge pieces first), categorisation of resources (putting like-coloured pieces into piles), and an understanding of the expected outcome (looking at the picture on the box). 

But if instead you organized one person for each piece in the puzzle, who only knew their piece, and could only place it once another piece that it fit had already been placed, but couldn't talk to the other people, and did not know the expected overall picture, the strategy for solving the jigsaw changes dramatically.

When they start solving the jigsaw, the 4 people holding corner pieces place them. Then 8 people holding corner-adjacent edge pieces can place them. The process continues, until the last piece is placed near the middle of the jigsaw.

This approach parallels how transformer models learn. There is no pre-agreed overall strategy or communication or co-ordination between people (circuits) - just some “rules of the game" to obey. The people think independently and take actions in parallel. The tasks are implicitly time ordered by the game rules.
Investigation (5 digit addition with 1 layer model)

# Rules of Addition
Addition has some mathematical rules baked in that the model must learn and obey if it is to do addition accurately. We broke up these rules into sub-tasks, which could be learnt independently per digit - improving the accuracy of addition of just that digit. (All the sub-tasks need to be learnt for all the digits before the model can give the complete correct answer to a question.)

We defined 3 “simple” tasks:
- BaseAdd (aka BA) which calculates the sum of two digits, say D3 and D3’, modulo 10, ignoring any carry over from previous columns e.g. 9 + 6 = 5 
- MakeCarry1 (aka MC1), which is true if adding say D3 and D3’, results in a carry over of 1 to the next higher column. 
- MakeSum9 (aka MS9) which is true if say D3 and D3’ sum to exactly 9.

We defined 2 “compound” tasks that chain operations across digits:
- UseCarry1 (aka UC1), which takes the previous column's carry output and adds it to the sum of the current digit pair. That is it combines BA and MC1.
- UseSum9 (aka US9), which propagates (aka cascades) a carry over of 1 to the next column if the current column sums to 9 and the previous column generated a carry over. US9 is a complex task as it spans three digits. For some rare questions (e.g. 05555+04445=10000) US9 applies to up to four sequential digits, causing a chain effect, with the MC1 cascading through multiple digits. This cascade requires a time ordering of the US9 calculations from lower to higher digits.

For each training question, we can say whether to get the right answer the model needs to use:
- Just the BA task (e.g. 11111+22222= ), or 
- BA, MC1 and UC1 tasks (e.g. 00045+00055= ), or
- All the tasks including US9 (e.g. 05555+04445= )

Graphing the training loss for the BA and UC1 tasks side by side for say Digit 3 shows an interesting pattern. In Phase 1, both tasks have the same (high) loss. In Phase 2, both curves drop quickly but the BA curve drops faster than the UC1 curve. This “time lag" is because the BA task must be accurate before the UC1 task can be accurate. In Phase 3, both tasks’ loss curve decrease slowly over time. 

<img src="{{site.url}}/assets/AdditionDigit3BaUc1TrainingLoss.png" style="display: block; margin: auto;" />

This graph supports the idea that the model is learning these tasks independently. The CoLab notebook contains many graphs supporting this idea (Parts 5, 6 & 7).

# Time Ordering

Transformers process the question and predict the answer one token at a time, strictly from left to right. For 5 digit integer addition this gives a total of 18 tokens:

<img src="{{site.url}}/assets/AdditionQuestionAnswerSteps.svg" style="display: block; margin: auto;" />

