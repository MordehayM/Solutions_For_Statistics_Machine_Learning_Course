In order to analyze the performance of the EM algorithm, I have examined the Log-Likelihood of the data ad the function of the iteration number.
First, I initialized the parameter with the true parameters and then applied the EM algoithm. We can see the result in the following figure:
<p align="center">
  <img src="C:\Users\HP\OneDrive - Bar-Ilan University\אוניברסיטה\תואר שני\סמסטר ג\(83841) למידת מכונה סטטיסטית\תרגילים\Solutions\EM algorithm for MoG\Initialized parameters are the true parameters.png" alt="true parameters">
</p>

On the same way, I initialized the parameters with random numbers and got the next fig:

<p align="center">
  <img src="C:\Users\HP\OneDrive - Bar-Ilan University\אוניברסיטה\תואר שני\סמסטר ג\(83841) למידת מכונה סטטיסטית\תרגילים\Solutions\EM algorithm for MoG\Initialized parameters are random.png" alt="true parameters">
</p>

As we can see, for randomly initialized parameters, EM algorithm converges after a larger number of iteration than the parameters initialized with true values.

Another fact is about the increasing of the samples. As the number of samples increases, the fit to the real MoG model increases and thus the more accurate the paramters are.
Moreover, for each random initializing the Likelihood score is different.