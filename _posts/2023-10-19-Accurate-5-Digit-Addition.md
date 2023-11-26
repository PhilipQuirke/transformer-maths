---
title: "Accurate 5-Digit Addition in Transformers (incomplete)"
date: 2023-10-19
---

## Introduction
When a builder starts to constuct a home, they don't start my making nails. Instead they use standard nails that are a standard shape and have a very low failure rate.
When they say build a wall, they understand how the strength of the nails contribute to the strength of the resulting wall. In some sense the nails are a "known good" component. 

In Machine Learning can we create small "known good" models that:
- Have a very low loss (preferrably zero)
- Have a well understood algorithm
- Are compact
- Can be re-used in larger models to help create larger "known good" models.

In this blog, we show an integer addition model with a loss of 0.000000002 that can do 1 million addition questions without error. We explain its algorithm in detail. We claim it is a "known good" component model.

In the future, we aim to create a "known good" integer multiplication model. Doing multiplication (at least for humans) includes some addition sub-tasks. If, before training, we initialise parts of our larger multiplication model, with a copy of the known good integer addition model, will the multiplication model re-use the addition model? Will this make it easier to understand the multiplication model algorithm?   

## Background

The <a href="{{site.url}}/2023/10/14/Understanding-Addition.html">previous post</a> discussed how a toy (1-layer, 3-head) transformer model answers integer addition questions like "33357+82243=". 

<img src="{{site.url}}/assets/AdditionQuestionAnswer.svg" style="display: block; margin: auto;" />

The model's algorithm is well understood (refer diagram) but it has a loss of ~0.5% so it is not "known good".

<img src="{{site.url}}/assets/StaircaseA3_Summary.svg" style="display: block; margin: auto;" />

This post investigates whether we can improve that model to do integer addition 100% accurately, and still understand its algorithm.

The CoLab notepad for this blog can be downloaded from <a href="{{site.url}}/assets/Accurate_Addition_in_Transformers.ipynb">here</a>, 
and used to train and test the model. You can alter the code to try out other approaches.


## Increasing accuracy
How can we increase the accuracy (i.e. decrease the loss) of the model?

# What didn’t work
We shouldn’t try to "program" or “teach” a model, but there are some standard "general purpose" changes that sometimes help make models more accurate.

For this model these approaches all failed to improve the model accuracy:
- Increasing the frequency of MakeSum9 cases in the training data so the model has more examples of this "hard" case to learn from (CoLab Part 3 setting ‘more_ms9_cases’).
- Increasing the number of attention heads from 3 to 4 or 5 to provide more computing power.
- Changing the question format from “12345+22222=” to “12345+22222equals” to give the model more calculation steps after the question is revealed before it needs to state the first answer digit.
- Changing the n_layers to 2 and n_heads to 2, increasing the number of attention heads from 3 to 4. Even with 30K training steps it was still inaccurate on A5 for some questions.

# What did work
What worked was increasing the number of model “layers” (n_layers) from 1 to 2, while retaining n_heads = 3. This doubles the number of attention heads in the model from 3 to 6. 
Also, the literature says a multiple-layer model gains the ability to “compose” the attention heads together in new ways to implement more complex algorithms.

These are the results for the 1 layer model used in <a href="{{site.url}}/2023/10/14/Understanding-Addition.html">Understanding Addition</a>:

<table>
    <thead>
        <tr>
            <th>Attribute vs model</th>
            <th>5D, 3H, 1L, 20K </th>
            <th>10D, 3H, 1L, 20K </th>
            <th>15D, 3H, 1L, 20K </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td># Digits</td>
            <td>5</td>
            <td>10</td>
            <td>15</td>
        </tr>
        <tr>
            <td># Attention Heads</td>
            <td>3</td>
            <td>3</td>
            <td>3</td>
        </tr>
        <tr>
            <td># Layers</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
        </tr>        
        <tr>
            <td># Training Batches</td>
            <td>20K</td>
            <td>20K</td>
            <td>20K</td>
        </tr>
        <tr>
            <td>Loss</td>
            <td>0.008314</td>
            <td>0.040984</td>
            <td>0.031746</td>
        </tr>     
        <tr>
            <td># Heads used</td>
            <td>15</td>
            <td>29</td>
            <td>46</td>
        </tr>
        <tr>
            <td># MLPs used</td>
            <td>6</td>
            <td>11</td>
            <td>16</td>
        </tr>
        <tr>
            <td>Seq correct answers</td>
            <td>~20</td>
            <td>~5</td>
            <td>~8</td>
        </tr>           
    </tbody>
</table>

"Seq correct answers" is roughly the number of addition question that model will get correct before it gets one wrong. (Based on CoLab Part14 experimental results.)  

With one extra layer the model gains accuracy (using CoLab with batch_size = 64, n_heads = 3, lr = 0.00008, weight_decay = 0.1):

<table>
    <thead>
        <tr>
            <th>Attribute vs model</th>
            <th>5D, 3H, 2L, 20K </th>
            <th>5D, 3H, 2L, 30K </th>
            <th>10D, 3H, 2L, 20K </th>
            <th>10D, 3H, 2L, 30K </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td># Digits</td>
            <td>5</td>
            <td>5</td>
            <td>10</td>
            <td>10</td>
        </tr>
        <tr>
            <td># Attention Heads</td>
            <td>3</td>
            <td>3</td>
            <td>3</td>
            <td>3</td>
        </tr>
        <tr>
            <td># Layers</td>
            <td>2</td>
            <td>2</td>
            <td>2</td>
            <td>2</td>
        </tr>          
        <tr>
            <td># Training Batches</td>
            <td>20K</td>
            <td>30K (10K more)</td>
            <td>20K</td>
            <td>30K (10K more)</td>
        </tr>
        <tr>
            <td>Loss</td>
            <td>0.0000027</td>
            <td>0.000000002</td>
            <td>0.0000943</td>
            <td>0.0001162</td>
        </tr>     
        <tr>
            <td># Heads used</td>
            <td>21</td>
            <td>19 (2 less)</td>
            <td>57</td>
            <td>51 (6 less)</td>
        </tr>
        <tr>
            <td># MLPs used </td>
            <td>15</td>
            <td>15</td>
            <td>27</td>
            <td>28 (+1)</td>
        </tr>
        <tr>
            <td>Seq correct answers</td>
            <td> > 4000 </td>
            <td> > 1M (~1M more!) </td>
            <td> > 1600 </td>
            <td> > 2750 (+1150)</td>
        </tr>           
    </tbody>
</table>

To be accurate, the 2 layer algorithm must learn the functionality of the 1 layer algorithm **and** 
learn additional functionality to handle the 06665+03335=10000 case by cascading the Carry 1 through multiple columns. 
How does it implement this? 

The 2 layer algorithm with 30K training batches has a final training loss of 0.000000002. Is the algorithm 100% accurate? 
This is hard to prove either way. CoLab Part14 provides evidence of accuracy via abrute force approach - doing 1,000,000 additions with no errors. 
If we understood the model algorithm, this might offer evidence for/against 100% accuracy. 


## What model parts are doing useful calculations?
A good first step in understanding a model is to look at **what** parts of the model are actually doing something useful when the model generates an answer to a question. 
We can do this analysis, before we understand the **how** the model is generating the answer.  

Most models don't use all their steps, attention heads and MLP layers to generate an answers. 
This diagram shows what parts the model is using in integer addition (for n_layers = 2, n_heads = 3, n_digits = 5): 

<img src="{{site.url}}/assets/StaircaseA5L2H3_Part1.svg" style="display: block; margin: auto;" />

The diagram is constructed for information gathering in a few steps. We'll describe the steps shortly, but some initial notes:
- A cell containing an X is not used to generate an answer 
- A attention head cell containing say D3,D3' is paying attention to those question digits.
- An MLP cell containing say A4 has a significant impact on the accuracy of that answer digit.
- The diagram was manually drawn in draw.io using the automatically generated information from the sections below.

# Attention patterns
Attention patterns show us what the model's attention heads are paying attention to. With 2 layers, there are now 6 attention heads over two rows. For example, the question “382954+92495=130790” gives this attention pattern:

<img src="{{site.url}}/assets/AttentionPattern5Digit3Heads2Layer.png" style="display: block; margin: auto;" />

Viewing a number of examples can give you a feel for patterns in the model's attention heads over successive steps.

# Which steps+heads do any useful calculations?
Sometimes the model does not use entire prediction steps. If we ablate alls head in a step, and the loss does not increase, then that step is **not** used by the algorithm, and can be excluded from further analysis. We find:

- With n_digits = 5, n_layers = 1, the addition algorithm does not use any data generated in steps 0 to 10 inclusive. The model also does not use the last (17th) step. Therefore, the addition is started and completed in 6 steps (11 to 16)
- With n_digits = 5, n_layers = 2, the addition algorithm does not use any data generated in steps 0 to 7 inclusive. The model also does not use the last (17th) step. Therefore, the addition is started and completed in 9 steps (8 to 16). What are the extra 3 steps used for?
- In **both** the above cases, the pattern failures in steps 11 to 16 each fail in exactly one digit, and the failing digit progresses steadily from A5 to A0 step by step.

CoLab Part 10B does this analysis and produces the below output. Note that an entry like "%SimpleUS9=31" means that 31% of SimpleUS9 questions failed - not 31% of all questions failed.

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>Fails</th>
            <th>% Fails by Case</th>
            <th># Fails by Patterns</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>1</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>2</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>3</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>4</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>5</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>6</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>7</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>8</td>
            <td>7%</td>
            <td>%SimpleUS9=31, %CascadeUS9=10, </td>
            <td>yNyyyy=12, NNyyyy=4, </td>
        </tr>
        <tr>
            <td>9</td>
            <td>10%</td>
            <td>%CascadeUS9=29, %SimpleUS9=28, </td>
            <td>yyNyyy=11, yNNyyy=9, NNNyyy=3, </td>
        </tr>
        <tr>
            <td>10</td>
            <td>15%</td>
            <td>%CascadeUS9=61, %SimpleUS9=28, </td>
            <td>yyyNyy=11, NNNNyy=11, yyNNyy=8, yNNNyy=4, yNyNyy=1, NyNNyy=1, </td>
        </tr>
        <tr>
            <td>11</td>
            <td>37%</td>
            <td>%MC1=52, %CascadeUS9=56, %SimpleUS9=31, </td>
            <td>Nyyyyy=89, </td>
        </tr>
        <tr>
            <td>12</td>
            <td>72%</td>
            <td>%MC1=87, %BA=68, %SimpleUS9=62, %CascadeUS9=46, </td>
            <td>yNyyyy=171, </td>
        </tr>
        <tr>
            <td>13</td>
            <td>68%</td>
            <td>%MC1=88, %BA=84, %SimpleUS9=44, %CascadeUS9=20, </td>
            <td>yyNyyy=163, </td>
        </tr>
        <tr>
            <td>14</td>
            <td>74%</td>
            <td>%MC1=85, %BA=80, %CascadeUS9=56, %SimpleUS9=56, </td>
            <td>yyyNyy=178, </td>
        </tr>
        <tr>
            <td>15</td>
            <td>66%</td>
            <td>%MC1=86, %BA=84, %SimpleUS9=38, %CascadeUS9=17, </td>
            <td>yyyyNy=158, </td>
        </tr>
        <tr>
            <td>16</td>
            <td>74%</td>
            <td>%MC1=83, %BA=88, %SimpleUS9=79, %CascadeUS9=27, </td>
            <td>yyyyyN=177, </td>
        </tr>
        <tr>
            <td>17</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>


# Which steps+MLP layers impact which use cases?
If we ablate an MLP layer in a step, and the loss does not increase, then that MLP layer is **not** used by the algorithm, and can be excluded from further analysis. With 2 layers, we find:

CoLab Part 10C does this analysis and for 2 layers finds that the addition algorithm does not use the MLPs in steps 0 to 7 inclusive, and also does not use the last (17th) step. 


# Which steps+heads/MLPs impact which answer digits summarised?
CoLan Part 10B works out which steps are used in the models calculations 
For each useful step, Parts 10C and 10D ablate one attention head and one MLP at a time and record the impact on loss..
Part 10E combines this information together into the below summary. A non-zero number means that when the cell is ablated, 
the model produces this percentage of bad answers (and so the cell is necessary for accurate answers.) 

<img src="{{site.url}}/assets/AccuratePart10E.png" style="display: block; margin: auto;" />

CoLab 10E also shows, for each useful head and MLP layer by useful step, "digit pattern(s)" of the incorrect answers. 
If the cell has a trailing "+" then this cell has another (less frequent) incorrect digit pattern. 
If a cell has one digit pattern then this gives us an insight into which digit the cell most impacts: 

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>8</th>
            <th>9</th>
            <th>10</th>
            <th>11</th>
            <th>12</th>
            <th>13</th>
            <th>14</th>
            <th>15</th>
            <th>16</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>L0H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=107</td>
            <td>yyNyyy=99</td>
            <td>yyyNyy=80</td>
            <td>yyyyNy=75</td>
            <td>yyyyyN=90</td>
        </tr>
        <tr>
            <td>L0H1</td>
            <td>yNyyyy=12, +, +</td>
            <td>yNyyyy=2, +, +, +</td>
            <td>NNNNyy=12, +, +, +</td>
            <td>Nyyyyy=7</td>
            <td>yNyyyy=51</td>
            <td>yyNyyy=26</td>
            <td>yyyNyy=9</td>
            <td>yyyyNy=73</td>
            <td></td>
        </tr>
        <tr>
            <td>L0H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=70</td>
            <td>yNyyyy=53</td>
            <td>yyNyyy=61</td>
            <td>yyyNyy=73</td>
            <td>yyyyNy=57</td>
            <td>yyyyyN=34</td>
        </tr>
        <tr>
            <td>MLP </td>
            <td>yNyyyy=6, +</td>
            <td>yNyyyy=1</td>
            <td>yyyNyy=20, +, +, +, +, +, +, +, +, +</td>
            <td>Nyyyyy=43</td>
            <td>yNyyyy=164</td>
            <td>yyNyyy=153</td>
            <td>yyyNyy=146</td>
            <td>yyyyNy=150</td>
            <td>yyyyyN=173</td>
        </tr>
        <tr>
            <td>L1H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=15</td>
            <td>yNyyyy=121</td>
            <td>yyNyyy=142</td>
            <td>yyyNyy=133</td>
            <td>yyyyNy=100</td>
            <td>yyyyyN=95</td>
        </tr>
    </tbody>
</table>

# Which steps+heads impact BA questions?

If we ablate each head in each step but only test BA questions, we gain insights. With 2 layers, these failures occur:

<img src="{{site.url}}/assets/AccuratePart11A.png" style="display: block; margin: auto;" />

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>8</th>
            <th>9</th>
            <th>10</th>
            <th>11</th>
            <th>12</th>
            <th>13</th>
            <th>14</th>
            <th>15</th>
            <th>16</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>L0H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=7</td>
            <td>yyNyyy=4</td>
            <td>yyyNyy=5</td>
            <td>yyyyNy=9</td>
            <td>yyyyyN=8</td>
        </tr>
        <tr>
            <td>L0H1</td>
            <td></td>
            <td>Nyyyyy=1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L0H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yyNyyy=1</td>
            <td>yyyNyy=2</td>
            <td>yyyyNy=2</td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td>yyyNyy=1, +</td>
            <td></td>
            <td>yNyyyy=15</td>
            <td>yyNyyy=16</td>
            <td>yyyNyy=14</td>
            <td>yyyyNy=12</td>
            <td>yyyyyN=12</td>
        </tr>
        <tr>
            <td>L1H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=9</td>
            <td>yyNyyy=12</td>
            <td>yyyNyy=10</td>
            <td>yyyyNy=5</td>
            <td>yyyyyN=6</td>
        </tr>
    </tbody>
</table>

# Which steps+heads impact MC1 questions?
If we ablate each head in each step but only test MC1 questions, we gain insights. With 2 layers, these failures occur:

<img src="{{site.url}}/assets/AccuratePart11B.png" style="display: block; margin: auto;" />

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>8</th>
            <th>9</th>
            <th>10</th>
            <th>11</th>
            <th>12</th>
            <th>13</th>
            <th>14</th>
            <th>15</th>
            <th>16</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>L0H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=16</td>
            <td>yyNyyy=14</td>
            <td>yyyNyy=7</td>
            <td>yyyyNy=12</td>
            <td>yyyyyN=13</td>
        </tr>
        <tr>
            <td>L0H1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=5</td>
            <td>yyNyyy=1</td>
            <td>yyyNyy=1</td>
            <td>yyyyNy=8</td>
            <td></td>
        </tr>
        <tr>
            <td>L0H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=8</td>
            <td>yNyyyy=6</td>
            <td>yyNyyy=11</td>
            <td>yyyNyy=3</td>
            <td>yyyyNy=1</td>
            <td>yyyyyN=2</td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=1</td>
            <td>yNyyyy=10</td>
            <td>yyNyyy=12</td>
            <td>yyyNyy=4</td>
            <td>yyyyNy=9</td>
            <td>yyyyyN=6</td>
        </tr>
        <tr>
            <td>L1H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=6</td>
            <td>yyNyyy=3</td>
            <td>yyyNyy=1</td>
            <td>yyyyNy=1</td>
            <td></td>
        </tr>
    </tbody>
</table>

# Which steps+heads impact SimpleUS9 questions?
If we ablate each head in each step but only test SimpleUS9 questions, we gain insights. With 2 layers, these failures occur:

<img src="{{site.url}}/assets/AccuratePart11C.png" style="display: block; margin: auto;" />

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>8</th>
            <th>9</th>
            <th>10</th>
            <th>11</th>
            <th>12</th>
            <th>13</th>
            <th>14</th>
            <th>15</th>
            <th>16</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>L0H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=14</td>
            <td>yyNyyy=11</td>
            <td>yyyNyy=6</td>
            <td>yyyyNy=13</td>
            <td>yyyyyN=10</td>
        </tr>
        <tr>
            <td>L0H1</td>
            <td>yNyyyy=6</td>
            <td></td>
            <td>yyyNyy=7</td>
            <td>Nyyyyy=1</td>
            <td>yNyyyy=12</td>
            <td>yyNyyy=7</td>
            <td>yyyNyy=2</td>
            <td>yyyyNy=7</td>
            <td></td>
        </tr>
        <tr>
            <td>L0H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=6</td>
            <td></td>
            <td></td>
            <td>yyyNyy=8</td>
            <td>yyyyNy=3</td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td>yNyyyy=2</td>
            <td></td>
            <td>yyyNyy=7</td>
            <td></td>
            <td>yNyyyy=9</td>
            <td>yyNyyy=5</td>
            <td>yyyNyy=5</td>
            <td></td>
            <td>yyyyyN=9</td>
        </tr>
        <tr>
            <td>L1H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=7</td>
            <td>yyNyyy=9</td>
            <td>yyyNyy=12</td>
            <td>yyyyNy=9</td>
            <td>yyyyyN=4</td>
        </tr>
    </tbody>
</table>

# Which steps+heads impact CascadingUS9 questions?
If we ablate each head in each step but only test CascadingUS9 questions, we gain insights. With 2 layers, these failures occur:

<img src="{{site.url}}/assets/AccuratePart11D.png" style="display: block; margin: auto;" />

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>8</th>
            <th>9</th>
            <th>10</th>
            <th>11</th>
            <th>12</th>
            <th>13</th>
            <th>14</th>
            <th>15</th>
            <th>16</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>L0H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>yNyyyy=20</td>
            <td>yyNyyy=18</td>
            <td>yyyNyy=9</td>
            <td>yyyyNy=9</td>
            <td>yyyyyN=19</td>
        </tr>
        <tr>
            <td>L0H1</td>
            <td>Nyyyyy=2</td>
            <td></td>
            <td>NNNNyy=10, +, +</td>
            <td></td>
            <td>yNyyyy=7</td>
            <td></td>
            <td></td>
            <td>yyyyNy=20</td>
            <td></td>
        </tr>
        <tr>
            <td>L0H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=9</td>
            <td></td>
            <td></td>
            <td>yyyNyy=9</td>
            <td>yyyyNy=6</td>
            <td>yyyyyN=10</td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td>yyNNyy=6, +, +, +, +, +, +</td>
            <td>Nyyyyy=8</td>
            <td>yNyyyy=12</td>
            <td>yyNyyy=7</td>
            <td>yyyNyy=6</td>
            <td>yyyyNy=3</td>
            <td>yyyyyN=4</td>
        </tr>
        <tr>
            <td>L1H0</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>L1H2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>MLP </td>
            <td></td>
            <td></td>
            <td></td>
            <td>Nyyyyy=11</td>
            <td>yNyyyy=10</td>
            <td>yyNyyy=29</td>
            <td>yyyNyy=27</td>
            <td>yyyyNy=21</td>
            <td>yyyyyN=1</td>
        </tr>
    </tbody>
</table>

## Inituition
The above is our base evidence to help us work out how the 2-layer algorithm works. Our intuition is:

- The last 5 steps (12 to 16) do approximately the same calculations in 1 and 2 layer cases.
- The extra 3 steps (8 to 11) in the 2 layer case, implement more accurate SimpleUS9 and CascadeUS9 calculations. 
- In the 2 layer case, the double staircase is 2 tokens wide. In the 1 layer case, the double staircase was 3 tokens wide and handled BaseAdd (perfectly), UseCarry1 (perfectly) and UseSum9 (imperfectly). Our intuition is that with 2 layers, the staircase handles BaseAdd and UseCarry1 but a new algorithm in steps 8 to 10 handles UseSum9 (with better accuracy than the 1 layer).

The answer digit A5 seems like a good place to focus our efforts:
- It is the earliest-revealed answer
- Despite always being 0 or 1, it is the hardest digit to calculation accurately as it may depend on a cascade all the way from D0+D0' e.g. 55555+44445=100000
- It is revealed in step 11. The diagram says it must be calculated in steps 8 to 11, using just 5 useful steps+heads and 3 useful step+MLPs!
- The 5 useful steps+heads pay attention to D2+D2', D1+D1', D0+D0', D3+D3', D4+D4'. Each pair of question digits is attended to once. This is obviously not random.


## Hypothesis
using the above information, we now seek to understand the detail of how the 2-layer algorithm is implemented.

# Hypothesis 1
Given the 2 layer attention pattern’s similarity to 1 layer pattern, and the above evidence, our first hypothesis was that the 2 layer algorithm:
- Is based on the same BaseAdd (BA), MakeCarry1 (MC) and UseSum9 (US) operations as the 1 layer.
- Uses the new early steps to (somehow) do the US9 calculations with higher accuracy than the 1 layer model.
- The long double staircase still finalises each answer digit’s calculation.
- The two attention nodes in the long double staircase steps do the BA and MC calculations and pull in US information calculated in the early steps.

If this is correct then the 2 layer algorithm successfully completes these calculations:
- A0 = D0.BA
- A1 = D1.BA + D0.MC
- A2 = D2.BA + (D1.MC or (D1.MS & D0.MC))
- A3 = D3.BA + (D2.MC or (D2.MS & D1.MC) or (D2.MS & D1.MS & D0.MC))
- A4 = D4.BA + (D3.MC or (D3.MS & D2.MC) or (D3.MS & D2.MS & D1.MC) or (D3.MS & D2.MS & D1.MS & D0.MC))
- A5 = D4.MC or (D4.MS & D3.MC) or (D4.MS & D3.MS & D2.MC) or (D4.MS & D3.MS & D2.MS & D1.MC) or (D4.MS & D3.MS & D2.MS & D1.MS & D0.MC)

Our intuition is that there are not enough useful heads+steps and heads+MLPs in steps 8 to 11 to complete the A5 calculation this way. So we abandoned this hypothesis.


# Hypothesis 2
Our second hypothesis was that the 2 layer algorithm:

- Has a **more compact** data representation. That is, it does not store BA, MC and US data as separate datums.
- Can pack more calculations into each head+layer and head+MLP in steps 8 to 11, allowing it to calculate A5 by step 11.
- Has a data representation that allows re-use of some A5 sub-calculations in the A4-calculation, allowing it to calculate A4 by step 12. 

Our claim is that the model stores the sum of each digit pair as a single token in the range “0” to “18” (covering 0+0 to 9+9). We name this operator Dn.T1, where T stands for “token addition”, and the 1 will be explained later:

- Dn.T1 = Dn + Dn’

The T1 operation does not understand mathematical addition. The model implements the T1 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “8” + “7” = “15”. There are 100 distinct mappings:
<img src="{{site.url}}/assets/Addition_T1Pairs.png" style="display: block; margin: auto;" />

T1 is a compact way to store data. If it needs to, the model can implement a bigram mapping to convert a T1 value into a BA, MC or MS value:
<img src="{{site.url}}/assets/Addition_T1Bigrams.png" style="display: block; margin: auto;" />

Our notation shorthand for these "conversion" bigram mappings is:

- Dn.BA = (Dn.T1 % 10) where % is the modulus operator 
- Dn.MC = (Dn.T1 // 10) where // is the integer division operator
- Dn.MS = (Dn.T1 == 9) where == is the equality operator

The D0.T1 value is perfectly accurate. But the other Dn.T1 values are **not** perfectly accurate because each is constrained to information from just one digit - hence the “1” in T1. We define another more accurate operator Dn.T2 that has “two-digit accuracy”. Dn.T2 is the pair sum for the nth digit plus the carry 1 (if any) from the n-1th digit T1:

- Dn.T2 = Dn.T1 + ( Dn-1.T1 // 10 )

Dn.T2 is more accurate than Dn.T1. The Dn.T2 value is always in the range “0” to “19” (covering 0+0+0 to 9+9+MakeCarry1). The model can implement the T2 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “12” + “1” = “13”. There are 38 distinct mappings: 

<img src="{{site.url}}/assets/Addition_T2Mappings.png" style="display: block; margin: auto;" />

We define operators Dn.T3, Dn.T4 & Dn.T5 each with higher accuracy:

- Dn.T3 = Dn.T1 + ( Dn-1.T2 // 10 )	Three-digit accuracy
- Dn.T4 = Dn.T1 + ( Dn-1.T3 // 10 )	Four-digit accuracy
- Dn.T5 = Dn.T1 + ( Dn-1.T4 // 10 )	Five-digit accuracy

The value D4.T5 is perfectly accurate as it integrates MC1 and cascading MS9 data all the way back to and including D0.T1. The values D1.T2, D2.T3, D3.T4 are also all perfectly accurate. If the model knows these values it can calculate answer digits with perfect accuracy:

- A1 = D1.T2 % 10 with zero loss
- A2 = D2.T3 % 10 with zero loss
- A3 = D3.T4 % 10 with zero loss
- A4 = D4.T5 % 10 with zero loss
- A5 = D4.T5 // 10 with zero loss

For the model to do integer addition perfectly accurately, it must calculate D4.T5 by step 11 so an accurate A5 can be revealed.
Understanding how the model calculates A5 will help us how understand the model's algorithm works. 

Applying this mathematical framework within the constraints of the above "What model parts are doing useful calculations" diagram, we hypothesise this is how the model calculates A5 by step 11:

- Step 8:
  - L0.H1: D2 attention: Calculate D2.T1 = D2 + D2’
  - L0.MLP: A4 impact: Not used
- Step 9
  - L0.H1: D1 attention: Calculate D1.T1 = D1 + D1’
  - L0.MLP: A4 impact: Not used
- Step 10
  - L0.H1: D0 attention: Calculate D1.T2 = D1.T1 + (D0 + D0’) // 10. Perfectly accurate.
  - L0.MLP: A2 .. A5 impact: Calculate D2.T3 = D2.T1 + D1.T2 // 10. Perfectly accurate. 
- Step 11:
  - L0.H1: D3 attention: Calculate D3.T4 = D3 + D3’ + D2.T3. Perfectly accurate. 
  - L0.H2: D4 attention: Calculate D4.T1 = D4 + D4’
  - L0.MLP: A5 impact: Calculate D3.MC = D3.T4 // 10  
  - L1.MLP: A5 impact: Calculate D4.T5 // 10 = (D4.T1 + D3.MC) // 10. Perfectly accurate A5.
 
The calculation by S11.MLP of D4.T5 // 10 = (D4.T1 + D3.MC) // 10 seems complex. Can this calc be done by the MLP? 
The D4.T1 and D3.MC values are in the residual stream. This is a bigram which the MLP can do.

There are MLP layers in S8 & s9 that are not used. It is theoretically unnecessary, but the model does depend on it. 
Ignoring this gap in our understanding for now, we further hypothesise this is how the model calculates A4 by step 12:

- Step 12:
  - L0.H0: D4 attention: Calculate D4.T5 = D4.T1 + D3.T4 // 10. Perfectly accurate.
  - L0.H1: D3 attention: Not used
  - L0.H2: D4 attention: Not used
  - L0.MLP: A4 impact: Calculate D4.T5 % 10. Perfectly accurate A4
  - L1.MLP: A4 impact: Not used

In step 12, there are another 2 heads and 1 MLP layer that are not used. 
They are theoretically unnecessary, but the model does depend on them. 

Obviously our hypothesis is not 100% right, but we have shown a hypothetical way to calculate A5 and A4 in time.
Ignoring "not used" cells for now, modifying the first diagram, we can show our hypothesis diagramatically:

<img src="{{site.url}}/assets/StaircaseA5L2H3_Part2.svg" style="display: block; margin: auto;" />

In this hypothesis, all the answer digits have been perfectly calculated by step 11. 
So why does the model retain the redundant long staircase BA calculations in step 11 to 16? It could just read the results from the work done in steps 8 to 11. Two options:
- The model is not optimising for compactness. The long staircase is discovered early and it works for simple questions. Once the overall algorithm gives low loss consistently it stops optimising.
- The model is not doing all the calcuations for all digits by step 11. Why should it? Maybe Hypothesis 2 requires too much co-ordination between cells and the model actually achieves less by step 11.

Time to start experimenting to get more information!


# Experimentation 2

## Part 13A: Claim: D3.T1 is calculated at S11.L0.H1 and S12.L0.H1
With n_digits = 5, n_layers = 2 and n_heads = 3:

We claim D3.T1 is calculated at S11.L0.H1 to help calculate A5 and A4. If correct then when ablating S11.L0.H1 should increase the A4 and A5 loss when D3+D3' >= 10. Part 13A finds that A5 loss increases as expected, but A4 loss does not change! So S11.L0.H1 is used to calculate A5 but not used to calculate A4.

So we claim D3.T1 is also calculated at S12.L0.H1 to help calculate A4. (This would eliminate one of the hypothesis's "not used" heads.) If correct then ablating S12.L0.H1, should increase the A4 loss when D3+D3' >= 10. We find that A4 loss increases but only in 25% of questions. Why 25%? This head is useful for A4 but it is not D3.T1. What is it?

This test shows the digit focus of the cell, that the cell is useful, that the cell has impact depending on whether D4+D4' >= 10 or not, but it does not show that the cell is implementing D4.T1. It could be implementing D4.MC or something else.


# Hypothesis 3
If Hypothesis 1 was too cold, and Hypothesis 2 was too hot, then maybe Hypothesis 3 will be just right:

The model's data model is more compact than Hypothesis 1 but less than Hypothesis 2. In early steps the model does just enough to correctly produce A4 and A5.

Our claim is that in steps 8 to 11 the model stores the sum of each digit pair as a tri-state variable. 
We name this operator Dn.C1, where C stands for “case”, and the 1 stands for 1 digit acurracy

- Dn.C1 = TriCase( Dn, Dn' ) where TriCase is a function implementing 3 distinct mappings:
    - Dn.C1 is 8 when Dn + Dn' sums to 0 to 8  
    - Dn.C1 is 9 when Dn + Dn' sums to exactly 9 
    - Dn.C1 is 10 when Dn + Dn' sums to 10 to 18 

The model implements the C1 operator as a bigram mapping from 2 input tokens to 1 result token e.g. “4” + “7” = “10”. There are 100 distinct mappings. 
For ease of reading we use "8", "9" and "10" to represent the 3 states, but the model will use some different but equivalent representation of the 3 states. 

If it needs to, the model can implement a bigram mapping to convert a C1 value into a MC or MS (but not a BA) value. Our notation shorthand for these "conversion" bigram mappings is:

- Dn.MC = (Dn.C1 == 10)
- Dn.MS = (Dn.C1 == 9)

We define another more accurate operator Dn.C2 that has “two-digit accuracy”. We defined Dn.C2 as follows: 
- Dn.C2 = TriAdd( Dn.C1, Dn-1.C1 )

Where TriAdd is a function implementing 6 distinct mappings:
<table>
    <thead>
        <tr>
            <th>Dn.C2</th>
            <th>Dn.C1</th>
            <th>Dn-1.C1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>8</td>
            <td>8</td>
            <td>8</td>
        </tr>
        <tr>
            <td>8</td>
            <td>8</td>
            <td>9</td>
        </tr>
        <tr>
            <td>8</td>
            <td>8</td>
            <td>10</td>
        </tr>
        <tr>
            <td>9</td>
            <td>8</td>
            <td>10</td>
        </tr>
        <tr>
            <td>10</td>
            <td>9</td>
            <td>10</td>
        </tr>
        <tr>
            <td>10</td>
            <td>10</td>
            <td>10</td>
        </tr>
    </tbody>
</table>

Dn.C2 is more accurate than Dn.C1. The model can implement the C2 operator as a bigram mapping from 2 input tokens to 1 result token.

We define operators Dn.C3, Dn.C4 & Dn.C5 each with higher accuracy:

- Dn.C3 = TriAdd( Dn.C1, Dn-1.C2 )	Three-digit accuracy
- Dn.C4 = TriAdd( Dn.C1, Dn-1.C3 )	Four-digit accuracy
- Dn.C5 = TriAdd( Dn.C1, Dn-1.C4 )	Five-digit accuracy

The values D4.C5, D3.C4, D2.C3 and D1.C2 are perfectly accurate as they integrate MC and cascading MS data all the way back to and including D0.C1. 
The model can uses these value to calculate perfectly accurate answer digits:

- A4 = ( D4.BA + D3.C4 ) % 10 
- A5 = ( D4.C5 == 10 )

Applying this mathematical framework within the constraints of the above "What model parts are doing useful calculations" diagram, we hypothesise this is how the model calculates A5 by step 11:

- Step 8:
  - L0.H1: D2 attention: Calculate D2.C1 = TriCase(D2, D2’)
  - L0.MLP: A4 & A5 impact: Not used
- Step 9
  - L0.H1: D1 attention: Calculate D1.C1 = TriCase(D1, D1')
  - L0.MLP: A4 impact: Not used
- Step 10
  - L0.H1: D0 attention: Calculate D0.C1 = TriCase(D0, D0’) 
  - L0.MLP: A2 .. A5 impact: Calculate D2.C3 = TriAdd(D2.C1, TriAdd(D1.C1, D0.C1)). Trigram mapping. Perfectly accurate. 
- Step 11:
  - L0.H1: D3 attention: Calculate D3.C1 = TriCase(D3, D3’)
  - L0.H2: D4 attention: Calculate D4.C1 = TriCase(D4, D4’)
  - L0.MLP: A5 impact: Calculate D4.C5 = TriAdd(D4.C1, TriAdd(D3.C1, D2.C3)).
  - L1.MLP: A5 impact: Calculate A5 = ( D4.C5 == 10 ). Perfectly accurate.

There are MLP layers in S8 and S9 that are not understood. It is theoretically unnecessary, but the model does depend on it. 
Ignoring this gap in our understanding for now, we further hypothesise this is how the model calculates A4 by step 12:

- Step 12:
  - L0.H0: D4 attention: Calculate D4.BA = (D4 + D4') % 10. 
  - L0.H1: D3 attention: Calculate D3.C1 = TriCase(D3, D3’). Duplicate of S11.L0.H1 
  - L0.H2: D4 attention: Not used. Could be duplicate of L0.H0
  - L0.MLP: A4 impact: Calculate D3.C4 = TriAdd(D3.C1, D2.C3). Relies on D2.C3 from S10.L0.MLP. Perfectly accurate
  - L1.MLP: A4 impact: Calculate A4 = (D4.BA + D3.C4) % 10. Perfectly accurate.

Hypothesis 3 has 2 unexplained MLP cells and maybe 1 unused Head.

<img src="{{site.url}}/assets/StaircaseA5L2H3_PartC.svg" style="display: block; margin: auto;" />

Time to start experimenting to get more information!

# Experimentation 3

Hypothesis has these testable claims:

- S12.L0.H1 is a duplicate of S11.L0.H1. So A4 accuracy does not depends on S11.L0.H1. This is proved true by experiment.

- S12.L0.MLP relies on S10.L0.MLP. So A4 accuracy depends on S10.L0.MLP and so also on S8.L0.H1, S9.L0.H1, S10.L0.H1. This is proved true by experiment.

- Is S12.L0.H2 a duplicate of S12.L0.H0? Is impact of ablating S12.L0.H0 and S12.L0.H2 the same? Part 10D shows:

<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>Layer</th>
            <th>Head</th>
            <th>% Fails</th>
            <th>% Fails by Case</th>
            <th># Fails by Patterns </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>12</td>
            <td>0</td>
            <td>0</td>
            <td>45</td>
            <td>%MC1=40, %BA=46, %CascadeUS9=56, %SimpleUS9=44,</td>
            <td>yNyyyy=107,</td>            
        </tr>
        <tr>
            <td>12</td>
            <td>0</td>
            <td>2</td>
            <td>22</td>
            <td>%MC1=46, %SimpleUS9=13, %CascadeUS9=2, </td>
            <td>yNyyyy=53, </td>            
        </tr>
    </tbody>
</table>

They are not the same. H2 has no role in BA calcs while H0 has. 



Some notes :
- Possible issue: Are there other ways to formulate the mathematical framework or different ways to map the framework to the calculation cells?
  - Possible solution: Do experiments to test the hypothesis    


# Pulling it all together (TBD)
TBA


# Acknowledgements
I gratefully acknowledge the support of the Apart Lab specifically Fazl Barez and Esben Kran. I am also thankful for Neel Nanda etc compiling a list of simple open Mechanistic Interpretability questions that a talented novice can make progress on, and so contribute to the field.
