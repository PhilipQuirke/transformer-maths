---
title: "Understanding Addition in Transformers"
date: 2023-10-14
---
# Introduction
If you type the question “When Mary and John went to the store, John gave a drink to” into Chat GPT, it will answer “Mary”.
Chat GPT is a “Transformer” model that considers each word in the question and generates a predicted answer.
You can also type in “12345+86764=” and it will give the right answer. How does it do this?

This blog explains how a toy (1-layer, 3-head) transformer model answers integer addition questions like "33357+82243=". 
When written down as a sequence of tokens the question and answer look like this:

<img src="{{site.url}}/assets/AdditionQuestionAnswer.svg" style="display: block; margin: auto;" />

This blog is written as an introduction to Mechanistic Interpretability and Transformer models for novices. 
It covers our investigation, testing and results of integer addition in transformers, build understanding section by section, and finally explaining this diagram:

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

A CoLab notepad can be downloaded from <a href="{{site.url}}/assets/Understanding_Addition_in_Transformers.ipynb">here</a>. 
It contains all the code needed to train the model and use the trained model, create graphs, etc. You can alter the code to try out other approaches.

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

We used these tasks to analyse the model during training. While the model was training, we categorised each question based on which tasks the model must perform to get the correct answer. We categorised each training question as one of:
- Only the BA task is needed (e.g. 11111+22222= ), or 
- The BA, MC1 and UC1 tasks are needed (e.g. 00045+00055= ), or
- All the tasks including US9 are needed (e.g. 05555+04445= )

Graphing the training loss for different question categories side by side helps us see when the model learns tasks. For example, the training loss graph for say Digit 3 shows an interesting pattern. In Phase 1, both tasks have the same (high) loss. In Phase 2, both curves drop quickly but the BA curve drops faster than the UC1 curve. This “time lag" is because the BA task must be accurate before the UC1 task can be accurate. In Phase 3, both tasks’ loss curve decrease slowly over time. The 3 phases seem to correspond to “memorisation”, “algorithm discovery” and “clean-up”.

<img src="{{site.url}}/assets/AdditionDigit3BaUc1TrainingLoss.png" style="display: block; margin: auto;" />

This graph supports the idea that the model is learning these tasks independently. The CoLab notebook contains many graphs supporting this idea (Parts 5, 6 & 7).

# Time Ordering

This transformer model processes the question and predicts the answer one token at a time, strictly from left to right. For 5 digit integer addition this gives a total of 18 tokens:

<img src="{{site.url}}/assets/QuestionAnswerSteps.svg" style="display: block; margin: auto;" />

Before answering the question, the trained model focuses on (aka attends to) the first token, then the first two tokens, then the first three tokens, etc. At each of the 18 steps the model does some calculations. After the question is fully revealed (at step 11), the model starts predicting the answer tokens, revealing the highest-value (100,000s) A5 digit first, then the other answer digits, finishing with the lowest-value (units) A0 digit. So it must reveal answer digits in the reverse order from what a human doing addition would!

To get the A5 digit we need to know whether the question generated a Carry 1 in digit A4, which may depend on A3’s Carry 1, which may depend on A2’s Carry 1, etc. Which leads to the question, has the model calculated all the digits before it reveals A5? How do we find out?

# Attention Heads and Attention Patterns
A key internal part of the model is its “attention heads”. For technical reasons (not explained here) they are the only part of the transformer model that can move information between multiple tokens. BaseAdd combines values from two tokens (digits) so we expect the attention heads to be involved.

Our model has 3 attention heads. This sample “attention pattern” graph (from CoLab Part 13) shows which tokens each of the 3 attention heads are focused on in each of the 18 steps:

<img src="{{site.url}}/assets/AttentionPattern5D3H.svg" style="display: block; margin: auto;" />

The pattern is 18 by 18 squares. Time proceeds vertically downwards, with one additional token being revealed horizontally at each step, giving the overall triangle shape. After the question is fully revealed (at step 11), each head starts attending to pairs of question digits from left to right (i.e. high-value digits before lower-value digits) giving the “double staircase" shape. The three heads attend to a given digit pair in three different steps, giving a time ordering of heads. The fact that the three staircases do not overlap is part of the model’s algorithm.

# Ablating Early Steps
Another key investigative tool is called “ablating”. For example, we can choose any of the 18 steps and scramble the model’s data at that step, to see if the model can still correctly answer the question. If the model can still answer the question correctly, then that step was irrelevant.

For our model, we can scramble the steps 0 to 10, without impacting accuracy (CoLab Part 10). This means that the model is not doing useful calculations in these steps. In the above attention pattern, the top half vertically is all “noise” - it is not useful. The useful calculations start with the “double staircase”.

So the model starts calculating in step 11 and reveals the answer digit A5 in step 12! Seems unlikely the model has calculated all the digits in step 11 and 12. So how does it calculate A5 accurately when it may need a UC1 from A4 which may need a UC1 from A3 etc?

# Simple vs Cascading UseSum9
This lead us to create test data to investigate the model accuracy on UseSum9 cases like:
00450+00550. This is a “simple” (one step) US9 case. The model copes with this.
04450+05550. This is a “cascading” (two, three or four step) US9 case. The model does not cope with this.

Note that simple US9 cases are rare and cascading US9 cases are very rare. They turn up infrequently in (random) training data, and so don’t impact the summary loss figures much. Our 1-layer 3-attention-head transformer model has a rare case it can’t cope with.

# Ablating the Last Step
One more fact: For our model, we can also scramble step 17, without impacting accuracy. This means A0 must have been calculated in step 16 or before. 

It turns out that ablating step 16 scrambles (only) digit A0, ablating step 15 scrambles (only) digit A1, etc. Each answer digit has been calculated one step before it is revealed.

# Pulling it all together
Summarizing the above findings: In each of the steps 11 to 16, the model calculates one digit, revealing it the following step. In each step, the 3 attention heads attend to different pairs of digits. The model is accurate, with the exception of cascading US9 cases. So it must be doing BA, MC1, UC1, MS9 and simple US9 calculations. 

The algorithm summarized in this diagram is consistent with these facts:

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

In the diagram: **A:** The 5 digit question is revealed token by token. The highest-value digit is revealed first. **B:** From the "=" token, the model attention heads focus on successive pairs of digits, giving a 'staircase' attention pattern. **C:** The 3 heads are time-offset from each other by 1 token so in each step data from 3 tokens is available. **D:** To calculate A3, the 3 heads do independent simple mathematical calculations on D3, D2 & D1. The results are combined by the MLP.

In more detail, the “programming” of the algorithm for A3 is:

<img src="{{site.url}}/assets/StaircaseA3_Detailed.svg" style="display: block; margin: auto;" />

The neural network's algorithm, calculated in 22 co-ordinated heads and MLP layers, is equivalent to the following python code: 

```
def calculate_answer_digits( n_digits, q1, q2 ):
  answer = ""

  for i in range(n_digits):
      pos = n_digits - i - 1

      mc1_prev_prev = 0
      if pos >= 2 and (q1[pos-2] + q2[pos-2] >= 10):
         mc1_prev_prev = 1

      mc1_prev = 0
      if pos >= 1 and (q1[pos-1] + q2[pos-1] >= 10):
         mc1_prev = 1

      ms9_prev = 0
      if pos >= 1 and (q1[pos-1] + q2[pos-1] == 9):
         ms9_prev = 1

      prev_prev = 0
      if mc1_prev == 0 and (ms9_prev == 1 and mc1_prev_prev == 1):
        prev_prev = 1

      digit_answer = (q1[pos] + q2[pos] + mc1_prev + prev_prev) % 10

      answer = str(digit_answer) + answer

  return answer

q1 = [2,3,4,5,6]
q2 = [1,3,1,5,6]
print(calculate_answer_digits(5,q1,q2))
```

With n_layers = 1, the above findings stay the same for n_digits = 10 or 15. (CoLab Part 2).

# The MLP
The above diagram references the MLP and uses the term “trigrams”. For clarity of presentation, we didn’t discuss these beforehand. We discuss them here.

The MLP is another key part of a Transformer model. For technical reasons (not explained here), the MLP can be thought of as a “key-value pair" memory that can hold many bigrams and trigrams. We believe that the model’s MLP pulls together the two-state 1st head result, the tri-state 2nd head result and the ten-state 3rd head result value, treating them as a trigram with 60 (2 x 3 x 10) possible keys. For each digit, the MLP has memorized the mapping of these 60 keys to the 60 correct digit answers (0 to 9). 

We haven’t proven this experimentally. We know that our MLP is sufficiently large to store this many mappings with zero interference between mappings.

# Answer Digits A0 and A1
The A0 and A1 digit calculations are simpler than the other digits:
A1 never uses the US9 task. 
A0 never uses the US9 or UC1 tasks.

These digits have their own digit-specific algorithms that are simplifications of the above algorithm. This simpler algorithm explains the fast training speed of these digits.

# Conclusions
The algorithm is very different from the human approach to addition. The algorithm is elegant & compact. It has high parallelism. 

Given the question’s high density, the need to answer promptly, and the small size of the model, the algorithm’s accuracy ( > 99.5% ) is impressive. (The next post shows how to increase the algorithm’s accuracy further.)

# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl Barez and Esben Kran in mentoring me through the process of developing this blog and the associated paper. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions (such as this blog’s question) that a talented novice can make progress on, and so contribute to the field.

This post is based on a paper written by Philip Quirke and Fazl Barez which was submitted in Sept 2023 to the ICLR 2024 conference. The paper is available at http://arxiv.org/abs/2310.13121
