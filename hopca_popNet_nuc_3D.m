%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function to compute HOPCA for a population of networks
%takes a semi-symmetric tensor X - p x p x m ( x n) - 3D or 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%X - the p x p x n tensor  - only 3D
%K - # of unique U factors to extract - should be small
%options - a struct with possible values:
%options.maxit - max iteration
%options.thr - threshold to establish convergence
%options.startv - a p x K matrix of intialization values fo v
%options.startu and .startw are analogous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%Z - a p x p x K matrix of PC networks
%W - an n x K matrix of subject weights for each of the of the PC networks
%V - a p x K factor for the network mode
%D - a K length scaling vector for the factors
%U - an m x K factor for the measure mode
%Xhat - residual tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[Z,W,Vall,Dall,Uall,Xhat,obj] = hopca_popNet_nuc_3D(X,K,lambda,options)


Uall = []; Vall = []; W = []; Dall = []; 
Z = tensor;
Xhat = X;
ns = size(X);
maxit = 1e3; thr = 1e-1;
if isfield(options,'maxit')
    maxit = options.maxit;
end
if isfield(options,'thr')
    thr = options.thr;
end

p = ns(1); n = ns(3);
for k=1:K
    if k==1
        Pw = eye(n);
    else
        Pw = eye(n) - W*W';
    end    
    tmp = randn(p);
    Ztmp = tmp'*tmp;
    [V,D] = eig(Ztmp);
    what = rand(n,1);
    w = what/norm(what);
    tmp = full(ktensor(diag(D),V,V,repmat(Pw*w,1,p)));
    
    obj = norm(Xhat-tmp)^2/(2*n);%+lambda*sum(diag(D));
    
    indo = 1; iter = 1;
    
    while indo>thr & maxit>iter
        oldo = obj(end);
        [V,D] = eig(double(ttv(Xhat,Pw*w,3))); %+90*eye(68,68));
        [tmp,ord] = sort(diag(D),'descend');
        D = D(ord,ord); V = V(:,ord);
        %D = diag(max(diag(D) - lambda,0));
        %modified by zw
        
        %ind = find(diag(D)>0);
        
        ind = 1:3;
        
        if(length(ind)<2)
           keyboard; 
        end
        
        Zten = tensor(V*D*V',[p p 1]);
        what = Pw*double(ttt(Xhat,Zten,[1 2],[1 2])./sum(diag(D).^2));
        w = what/norm(what);
        tmp = full(ktensor(diag(D(ind,ind)),V(:,ind),V(:,ind),repmat(Pw*w,1,length(ind))));
        obj = [obj norm(Xhat-tmp)^2/(2*n)];%+lambda*sum(diag(D))];
        indo = abs((obj(end) - oldo)/obj(1));
        iter = iter + 1;
    end
    Z(:,:,k) = Zten; W = [W w];
    Vall = [Vall V(:,ind)]; Dall = [Dall diag(D(ind,ind))']; 
    Uall = [Uall repmat(w,1,length(ind))];
    Xhat = Xhat - tmp;
end

