In this exercise, I implemented the Kalman filter.
For the following setting of the problem:  
state_vector = z<sub>t</sub> = (position<sub>t</sub>, velocity<sub>t</sub>),   
z<sub>t</sub> = Az<sub>t</sub> + v<sub>t</sub>,  
observation = x<sub>t</sub> = position<sub>t</sub> = Cz<sub>t</sub> + w<sub>t</sub>  
with v<sub>t</sub>~N(0, Q), w<sub>t</sub>~N(0, R),  
I got the following result:
<p align="center">
  <img src="position vs time.png" alt="true parameters">
</p>
As we can see The MSE is reduced by 3.17 times when the Kalman filter is applied as opposed to just using the observation