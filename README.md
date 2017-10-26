# Temporal-Consistent-Frame-Selection-For-Complex-Event-Detection-in-Videos
Software package for Temporal Consistent Evidence (i.e., frame) Selection, which is designed to select the most informative  video frames from weakly labeled videos crawled from YouTube. 

This folder contains:

************************************************************************************

1. SLEP_package_4.1

- This is a third-party software package available at http://yelab.net/software/SLEP/, 
  which is a sparse learning software package. In solving the optimization of our proposed 
  temporal consistent evidence selection procedure, we apply one optimization function called nnLeastR()
  in this package in our own optimization function solve_ADMM.m. 

2. Main.m

- This is a toy example illustrating how to run the entire program to solve the frame selection problem.
  Some synthetic video features are generated as the input feature for the learning process. 
  
3. EvidenceSel.m 

- This function is used to select the most informative frames in videos based on the proposed temporal 
  consistent evidence selection method. 

4. SimEmb.m

- This function is used to embed each video onto the frame dictionary based on similarity calculation. 
  It is called by EvidenceSel.m. The embedded representation of video is used as input for 
  automatic feature selection (i.e., frame selection).   

5. L2_distance.m 

- This is a third-party software contributed by Roland Bunschoten. This function is used to calculate 
  L2 distance between feature vectors of video frames. It is called by SimEmb.m. 

6. Solve_ADMM.m 

- This function is used to solve the optimization of the proposed frame selection method. It iteratively updates 
  three set of parameters (w, a and z) till convergence. It is called by EvidenceSel.m.  

7. Lasso_w.m  

- This function is used to solve L1 norm minimization, which is called by Solve_ADMM.m.
