function y_grads = compute_ygrads(P,Pm_owns_xi,PnotNN,Q,matrixNN,ydata);

% This is already done for P:
[Qm_owns_xi,QnotNN] = compute_grads_prepare(matrixNN,Q);

% In the MAP IT cost function there are two constants wich basically
% act as adaptive step sizes on attractive and repulsive forces, 
% respectively. Choose here to simplify these a bit since results 
% seem robust to these choices and they are easy to compute as below:
c1 = trace(P*Q');
c2 = trace(Q*Q');

% Needed to compute gradients for neighbors 
LNN = -(max(Pm_owns_xi ./ c1, realmin) - max(Qm_owns_xi ./ c2, realmin)) .* (Q.*Q);

% Needed to compute gradients for not neighbors
LnotNN = -(max(PnotNN ./ c1, realmin) - max(QnotNN ./ c2, realmin)) .* (Q.*Q);

% Compute the two different types of gradient contributions
y_gradsNN = (diag(sum(LNN, 2)) - LNN) * ydata;
y_gradsnotNN = (diag(sum(LnotNN, 2)) - LnotNN) * ydata;

% Sum up and output
y_grads = y_gradsNN + y_gradsnotNN;