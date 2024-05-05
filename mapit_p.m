function Ydata = mapit_p(P, labels, no_dims, k)
% The function "mapit_p" implements the basic MAP IT algorithm to
% visualize representations as described in the paper:
%
% "MAP IT to Visualize Representations"
% Robert Jenssen
% ICLR 2024
%
%   Ydata = mapit_p(P, labels, no_dims)
%
% MAP IT is inspired by t-SNE and this is reflected in parts of the code.
% Please visit Laurens van der Maaten's web page for code and more 
% information on t-SNE: https://lvdmaaten.github.io/tsne/
% 
% In the implementation below it is assumed that transition probabilities
% are computed similarly to (symmetric) t-SNE, using the same general 
% approach when it comes to perplexity etc, but this is not necessary for 
% the MAP IT theory in general.
% 
% Input: A matrix "P" of input space (source) transition probabilities. 
% "labels" is only used for plotting and not for the optimization. 
% The variable "no_dims" will normally be 2. The variable "k" affects 
% results as it defines the neighborhood structure which is important
% for the computation of gradients. Normally, a values in the range 
% 5-15 would probably be "reasonable" choices.
% 
% Output: "Ydata" is the obtained representation in the low-dimensional
% (usually two-dimensional) target space. This is the representation
% which is visualized in the end.
% 
% (C) Robert Jenssen, 2024
% UiT The Arctic University of Norway

    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    
    % First check whether we already have an initial solution
    if numel(no_dims) > 1
        initial_solution = true;
        ydata = no_dims;
        no_dims = size(ydata, 2);
    else
        initial_solution = false;
    end
    
    % Initialize some variables
    n = size(P, 1);                                     % number of instances
    momentum = 0.5;                                     % initial momentum
    final_momentum = 0.8;                               % value to which momentum is changed
    mom_switch_iter = 250;                              % iteration at which momentum is changed
    stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
    max_iter = 1000;                                    % maximum number of iterations
    epsilon = 50;                                       % initial learning rate

    % Make sure P-vals are set properly
    % This is to be comparable to van Maaten () assuming transition 
    % probabilities are computed similar to t-SNE
    P(1:n + 1:end) = 0;                                 % set diagonal to zero
    P = 0.5 * (P + P');                                 % symmetrize P-value

    % MAP IT is not using the trick of early exaggeration (t-SNE) but
    % keepping the option open
    if ~initial_solution
        %P = P * 4;                                      % lie about the P-vals to find better local minima
        P = P * 1;
    end
    
    [Ps,indexP] = sort(P,2,'descend');   % Find nearest neighbors
 
    % Matrix where a "one" in a row indicates a neighbor
    [matrixNN] = compute_matrixNN(indexP,k,n);

    % Need these for computing gradients over neighbors and 
    % non-neighbors, respectively. Want to do this only once
    % over the "P" transition probabilities
    [Pm_owns_xi,PnotNN] = compute_grads_prepare(matrixNN,P);

    % Initialize the solution
    if ~initial_solution
        ydata = .0001 * randn(n, no_dims);
    end
    y_incs  = zeros(size(ydata));

    % Run the iterations
    for iter=1:max_iter
        
        % Compute joint probability that point i and j are neighbors
        sum_ydata = sum(ydata .^ 2, 2);
        Q = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
        Q(1:n+1:end) = 0;
        
        % Computing gradients split over nearest neighbor contributions
        % and contributions from non-neighbors
        y_grads = compute_ygrads(P,Pm_owns_xi,PnotNN,Q,matrixNN,ydata);
        
        % Taking into acount momentum for the updates
        y_incs = momentum * y_incs + epsilon * y_grads; 

        % Do the updates
        ydata = ydata + y_incs;
        
        % Update the momentum if necessary
        if iter == mom_switch_iter
            momentum = final_momentum;
        end
        if iter == stop_lying_iter && ~initial_solution
            % Not really using this, keeping the option open
            %P = P ./4
        end
        
        % This is a proxy to the full cost function which is easy
        % to compute in order to monitor progress and used here 
        % for convenience when comparing runs (not influencing
        % optimization)
        cost(iter) = -log(sum(sum(P*Q))) + log(sum(sum(Q*Q)));

        % Print out progress
        if ~rem(iter, 10)
            cost(iter) = -log(sum(sum(P*Q))) + log(sum(sum(Q*Q)));
            disp(['Iteration ' num2str(iter) ': error is ' num2str(cost(iter))]);     
        end
        
        % Display scatter plot (maximally first three dimensions)
        if ~rem(iter, 10) && ~isempty(labels)
            if no_dims == 2
                % Assuming two dimensions normally for the target space
                scatter(ydata(:,1), ydata(:,2), 40, labels, 'filled');
            else
                % Not assuming more than three target space dimensions, usually
                % only two
                scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 40, labels, 'filled');
            end
            axis tight
            axis off
            drawnow
        end
    end
    Ydata.data = ydata;
    Ydata.cost = cost;
    Ydata.k = k;
    Ydata.iter = max_iter;
    Ydata.epsilon = epsilon;
    