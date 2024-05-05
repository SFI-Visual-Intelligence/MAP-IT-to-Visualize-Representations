function [Pm_owns_xi,PnotNN] = compute_grads_prepare(matrixNN,P);
% The function "compute_grads_prepare" enables to split up computation
% of gradients for the MAP IT cost function in two parts. One which
% concerns gradient contributions from neighborhoods of each point i 
% and one which comes from the non-neighbors of the point i.
%
% [Pm_owns_xi,PnotNN] = compute_grads_prepare(matrixNN,P);
%
% Input: A transition probability matrix, either "P" or "Q". A binary
% matrix that indicates nearest neighbors of each point. 
%
% Output: "Pm_owns" refer to marginal probability for point i computed 
% in its neighborhood. "PnotNN" collects pairwise transition probabilities
% for non-neighbors. 
%
% The reason for computing "Pm_owns" and "PnotNN" is the different way
% neighbohbors for the point i and non-neighbors for that point contribute
% to the gradient with respect to point i.
%
% (C) Robert Jenssen, 2024
% UiT The Arctic University of Norway
n = size(P,1);

% Each row is the marginal probabilities computed in a neighborhood
% around a point in the input space
tildeP = matrixNN * P;

% Marginal prob for each point in its own neighborhood
Pm_own = diag(tildeP);

% Identifies Pm's in neighborhood of each xi (on each row)
Pm_owns_xi = repmat(Pm_own',[n 1]) .* matrixNN;

% Needed later - points that are not neighbors of xi
PnotNN = P .* (1-matrixNN);

