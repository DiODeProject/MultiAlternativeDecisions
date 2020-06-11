# Optimal-policy-for-value-based-decision-making

These codes modify the analysis presented in the paper by Tajima, Drugowitsch & Pouget (2016) [1], to examine the implication of maximising geometric-discounted future rewards on the optimal policy, rather than maximising arithmetic mean reward rate.

**N.B.** reward rate across trials is not maximised, but set to 0; this has the effect of considering only single trial dynamics, since the cost of waiting for the next trial becomes zero. 


CITATION:

[1] Satohiro Tajima*, Jan Drugowitsch*, and Alexandre Pouget.
Optimal policy for value-based decision-making. 
*Nature Communications*, **7**:12400, (2016). 
**Equally contributed.*


USAGE:
`valueDecisionBoundaryRR.m()` generates figures used in Figs. 3 or 6 in the paper.  You can switch the utility function and reward calculation assumed in the model by removing commented-out lines in the code, and toggling the value of the `geometric` variable, respectively 
