function [w] = LASSO_w(w_old, secondmember, X, XtX, lambda, mu)

%% This function is designed based on the derivation for optimizing w in the 
%  iterative ADMM optimization procedure, in which the second input parameter secondmember 
%  equals X(1-a-z). To understand the parameters and optimization procedure in this function, 
%  please refer to the detailed mathematical derivation.   

w = w_old;
y=secondmember;

objFold=0;
objF=inf;
for i=1:100;
    backtrack=1;
    pas=0.0001;
    if abs(objFold - objF)< objFold*1e-4
        break
    end;
    objFold=objF;
    
    while backtrack
        wa= w + pas*(X'*secondmember-XtX*w);
        
        wa= sign(wa).*max(0,abs(wa) - pas*lambda/mu);
        objF= 0.5*norm(y-X*wa).^2 + 1/mu*sum(abs(lambda.*wa));
        obj(i) = objF;
        objQ= 0.5*norm(y-X*w).^2 + 1/mu*sum(abs(lambda.*w)) + ...
            (wa-w)'*(X'*y-XtX*w) + 1/2/pas*norm(wa-w).^2;
        if objF > objQ || objF > objFold
            pas=pas*0.5;
        else
            backtrack = 0;
            w=wa;
         
        end;
    end;
    %fprintf('Iteration = %d, obj = %d.\n',i,obj(i));
end;