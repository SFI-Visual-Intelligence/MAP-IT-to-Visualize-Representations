function [matrixNN] = compute_matrixNN(indexP,k);
% The function "compute_matrixNN" creates a binary matrix where the i'th
% row corresponds to point number i. A "1" is put in the l'th element
% of row i if point l is one of the k nearest neighbors of point i.
% Otherwise "0". This is an operation done once.
%
% [matrixNN] = compute_matrixNN(indexP,k);
%
% Input: "indexP" stores the indicies of the sorted transition matrix P
% in descending order. "k" is the number of nearest neightbors considered.
% 
% Output: Binary matrix "matrixNN".
%
% (C) Robert Jenssen, 2024
% UiT The Arctic University of Norway

matrixNN = zeros(n);

for i = 1 : n;

    matrixNN(i,indexP(i,1:k)) = 1;

end;