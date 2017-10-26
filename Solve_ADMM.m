function [w,obj] = Solve_ADMM(X, S, lambda, C, options)

% Input: 

%  - X : (N by L matrix) where N is the # of videos, L is the dimensionality of the embedded 
%         representation of videos, i.e., the total number of frames in the training videos.   
                              
%  - S : (L by L matrix) where L is the total # of frames in the training videos. It is  
%         the Laplacian matrix of the temporal consistency relationship matrix.                     
     
% -  lambda :  input parameter for the L1 term. The value can be set according to 
%              the performance of cross validation. 

% -  C :  input parameter for the temporal consistency regularization term. The value 
%         can be determined based on the performance of cross validation. 

% -  options :  Parameters for controlling the termination of ADMM iterative procedure. It 
%               contains three fields:
%               * mu: weighting parameter of the quadratic penalty in the augmented Lagrangian. 
%               * nbitermax: maximum number of iterations for ADMM. 
%               * stopobjratio: the ratio of objective function value between successive iterations. 
%               Please refer to the submitted PDF report for details.  

% Output:
%   - w : 1 by L evidence selective parameter vector in which each entry indicates 
%         the importance of the corresponding frame in a video. 

%   - obj : the values of the objective function.  


%% Initialization
staterand=rand('seed');
staterandn=randn('seed');
[num dim]= size(X);
one = ones(num,1);
mu=options.mu;
if size(lambda,1)~= dim
    lambda=lambda(1)*ones(dim,1);
end

% Parameter Initialization
z=randn(num,1)*0.1; % Lagrangian multipliers
w=randn(dim,1)*1;         % primal variables

% initialize a and compute objective value
S = C*S;
a=1-X*w;
aplus=(a>0).*a;
delta = 0.2; % empirical setting, this value can be changed according to cross validation performance 
penal= delta*sum(aplus); % aplus takes into account the hinge function 
obj(1) = sum(abs(lambda.*w)) + w'*S*w + penal;
fprintf('Iteration = %d, l1(w) = %d, wSw = %d, penal = %d, obj = %d.\n',1,sum(abs(lambda.*w)), ...
    w'*S*w,delta*penal,obj(1));

XtX = X'*X+2*S/mu;

%% ADMM Iterations
for iter = 1:options.nbitermax
   
    % Update w
    a1 = one - a - z;
    w = LASSO_w(w, X'*a1 , XtX+2*S/mu, XtX, lambda, mu);
    
    % Update a
    v= one - z - X*w;
    aplus=(a>0).*a;
    aminus=(a<0).*abs(a);
    vauxp=[];
    for k = 1:500
        aminusold=aminus;
        aminus=(-v+aplus).*( (-v+aplus)>0);
        b=aminus+v;
        aplusold2=aplus;
        [aplus,~] = nnLeastR(eye(length(aplus)),b,delta);
        if max(abs(aplusold2-aplus))<max(abs(aplus))*1e-4 && max(abs(aminusold-aminus))<max(abs(aminus))*1e-4
            break
        end
    end
    a = aplus-aminus;
    penal= delta*sum(aplus); % aplus takes into account the hinger function
    
    obj(iter+1) = sum(lambda.*abs(w)) + penal + w'*S*w;
    if abs(obj(iter+1)-obj(iter))<(options.stopobjratio*abs(obj(iter+1)))
        break
    end;
     fprintf('Iteration = %d, l1(w) = %d, wSw = %d, penal = %d, obj = %d.\n',iter+1,sum(abs(lambda.*w)), ...
    w'*S*w,penal,obj(iter+1));

    % update Lagrangian multiplier z
    z= z + (X*w - one + a);
    if sum(abs(z))>1e10 || sum(abs(a))>1e10
        break % divergence of the algorithm
    end
    
end

staterand=rand('seed');
staterandn=randn('seed');

