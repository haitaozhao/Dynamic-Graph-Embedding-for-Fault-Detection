# Dynamic Graph Embedding for Fault Detection

----------
- Code for the paper "Dynamic Graph Embedding for Fault Detection"
- The demo codes can be found in the directory "Matlab_code". They are developed to do the fault detection on the data of Fault 1. the file "myConstructW.m" is developed to obtain the similarities in Equation (6). In the file, we give the annotations according to the Eqautions in the paper. "myfunction_tensorLPP_markov_paper.m" is the main program, which can run directly. "TensorLGE.m" and "TensorLPP.m" are the codes needed for the main program. Both "TensorLGE.m" and "TensorLPP.m" are designed by Deng cai, who is the second author of the paper "Tensor Subspace Analysis", published in Neural Information Processing Systems 18 (NIPS 2005). The file "kde.m" is the code of kernel density estimation which is used to determine the control limit of T2 and SPE statistics.
- "File_published_by_matlab_in_PDF.pdf" is the running results and the codes published with MATLAB® R2015b. "Files_and_results_published_by_matlab.zip" contains the html version published with MATLAB® R2015b.
- "Twin_Peaks.mat" is the 3000 data points used for Figure 1. 