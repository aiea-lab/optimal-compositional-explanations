# Changes after ICML

**Note**: After submission, we applied several optimizations to make the code run faster. These edits do not change the meaning or scale of the results. However, small variations may appear in visited-node counts and runtime. Specifically, we made the following improvements:

- We unified the implementation of all beam variants so that they now share almost the same structure (except for state-space ordering).

- We updated the formula implementation:
	- We replaced the old formula hash function (available <a href="https://github.com/KRLGroup/Clustered-Compositional-Explanations/blob/main/src/formula.py">here</a>) with a new one that is slightly faster, although it covers fewer cases than the previous implementation.
	- We removed the `OrderedFormula` class (described <a href="https://github.com/KRLGroup/Clustered-Compositional-Explanations/blob/main/src/formula.py">here</a>) and now handle both queueing and ordering directly within the formula class. Previously, as in CCE, `OrderedFormula` was used both to sort the state space and as the heap key during optimal search.

- We optimized propagation:
	- The code now treats every newly discovered ancestor during propagation as a visited node (previously, only nodes expanded during search were considered visited; ancestors could therefore be revisited multiple times, and IoU generation for them during propagation was not counted as state visitation).
	- During propagation, we now generate quantities and propagate information only for states that have not already been visited (previously, this was done every time).

- For beam variants, we no longer expand nodes in the current beam that were already expanded in previous beam levels.

- We no longer differentiate between maximum and minimum unique quantities since they can be computed exactly (previously, we differentiated between maximum and minimum extras and intersections). This modification saves computation and runtime for the optimal and beam-optimal algorithms at the cost of reduced code cleanliness (i.e., we broke symmetry between quantities).

We tested this implementation and did not observe significant differences in the results compared to the previous implementation. If you observe any significant differences, please let us know by opening an issue. We maintain the legacy code and would be happy to investigate any discrepancies.