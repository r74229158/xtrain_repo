## Welcome to the Theoretical Aspects of XtrAIn

This folder contains notebooks performing different experiments that were used to empirically and statistically test theoretical aspects of the method. These tests were presented in the supplementary material of the paper. 

They are not part of the main pipeline of XtrAIn but might be useful for researchers for testing, evaluation or further development.

### Summary 

Here is a summary of the `.ipynb` files:
- `00_artificial_step`: It calculates the attribution scores resulting from the artificial step,
- `01_logit_evolution`: It calculates the evolution of logits and derives a mean value for change of logits corresponding to target and non-target neurons,
- `02_breaking_linearity`: Compares XtrAIn for path-dependent and path-independent environments (by skipping steps of the path-dependent env),
- `03_additive_inv_property`: Breaks XtrAIn's computed attribution score in <u>theoretical</u> and <u>practical</u> terms and performs a comparison of three (XtrAIn, `df`, `df'`) based on Deletion AUC.