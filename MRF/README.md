In this exercise, I calculated the approximate probability of a binary MRF by using Gibbs Sampling and Mean-Field Approximation.

First I created x by Gibbs sampling (configuration number was set to 1000 iteration).
Then y samples were generated from x such that y ~ N(x, 1).
Given the noisy information y, we want to estimate x.
I generated the following x and y:


        [[1 1 1 1 1]                [[ 1.12394296  1.93604989  1.29733012  2.28706531  1.89175775]                     
        [1 1 1 1 1]                 [ 0.21060333  2.01258485  1.76135494  0.28466702 -0.57350169]                              
    x = [1 1 1 1 1]        y =      [ 1.13629946  0.96366533  0.4769108   2.74035392 -1.0986966]                                               
        [1 1 1 1 1]                 [ 1.14205407  0.72839782  1.07481377 -1.16915695  -1.66621771]
        [1 1 1 0 1]]                [ 0.71462959  1.71224286  1.2584767   0.84272283  0.07834539]]                         


First I calculated the true probabilty p(x=1|y) by summing over all 2<sup>24</sup> and got the next result:
<prep>

                [[0.89281651 0.97596663 0.96937225 0.97384065 0.89536511]  
                [0.88703589 0.98905988 0.98613419 0.90553754 0.62336015]  
    p(x=1|y) =  [0.93961155 0.97372573 0.9613243  0.95529946 0.54223134]                                                    
                [0.93294333 0.96002931 0.94226847 0.72828578 0.77183075]  
                [0.86159537 0.95412458 0.92387066 0.78999799 0.65206973]]  

</prep>

By appling Gibbs sampling (with 1000 estimates) I got the following results:


                     [[0.89281651 0.97596663 0.96937225 0.97384065 0.89536511]
                      [0.88703589 0.98905988 0.98613419 0.90553754 0.62336015]
    Gibbs_p(x=1|y) =  [0.93961155 0.97372573 0.9613243  0.95529946 0.54223134]                  
                      [0.93294333 0.96002931 0.94226847 0.72828578 0.77183075]                              
                      [0.86159537 0.95412458 0.92387066 0.78999799 0.65206973]]                             

Next, I examined the influence of the number of estimates of Gibbs sampling(the number of samples to mean over) and got the following plot:

<p align="center">
  <img src="C:\Users\HP\OneDrive - Bar-Ilan University\אוניברסיטה\תואר שני\סמסטר ג\(83841) למידת מכונה סטטיסטית\תרגילים\Solutions\MRF\Error of Gibbs vs Num of estimation.png" alt="true parameters">
</p>


Afterwards, I applied the Mean-Field approximation(with 10 iteration) to compute the same probabilty and got:


                    [[0.91979999 0.98549993 0.97669181 0.98964104 0.94686719]
                    [0.92234505 0.99498176 0.99388181 0.96270448 0.75474079]
    MF_p(x=1|y) =   [0.96723871 0.98680475 0.98014244 0.99492021 0.68925255]
                    [0.96529946 0.98311249 0.9854973  0.88647003 0.94800687]
                    [0.89146357 0.98027518 0.97228088 0.93375239 0.79279629]]

On the same way I examined the effect of the number of MF's iteration on both the Mean-Square-Error(MSE) and Kullback Leibler divergence and got the following plot:

<ins>MSE</ins> - 
<p align="center">
  <img src="C:\Users\HP\OneDrive - Bar-Ilan University\אוניברסיטה\תואר שני\סמסטר ג\(83841) למידת מכונה סטטיסטית\תרגילים\Solutions\MRF\Error of Mean Field vs Num of iteration.png" alt="true parameters">
</p>

<ins>KL</ins> - 
<p align="center">
  <img src="C:\Users\HP\OneDrive - Bar-Ilan University\אוניברסיטה\תואר שני\סמסטר ג\(83841) למידת מכונה סטטיסטית\תרגילים\Solutions\MRF\Error of KL vs Num of iteration.png" alt="true parameters">
</p>

