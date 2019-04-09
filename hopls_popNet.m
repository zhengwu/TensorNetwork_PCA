%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function to compute HOPLS for a population of networks
%takes a semi-symmetric tensor X - p x p x m x n (4D) or p x p x n (3D)
%and the response Y is n x q
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%X - the tensor - either 3D - p x p x n OR 4D - p x p x m x n
%Y - the n x q response matrix
%K - # of factors to extract
%options - a struct with possible values:
%options.pca - an indicator for whether PLS-PCA should be performed
%(e.g. treat covariance of data as identity) - note that this is
%the only option for now 
%options.maxit - max iteration
%options.thr - threshold to establish convergence
%options.startv - a p x K matrix of intialization values for v
%options.startu and .startw are analogous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%Zhat - pls compnents - a n x K matrix
%V - a p x K factor for the network mode (pls loadings on network)
%D - a K length scaling vector for the factors
%U - an m x K factor for the measure mode
%W - an n x K factor for the subject mode (null if X is 3D)
%Xhat - residual tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To Do - figure out when complex eigenvectors arise and write a
%catch for this - seems to only occur when K is larger than the
%truth
%%%%%%%%%%%%%%%%%%%%%%%%%


function[Zhat,V,D,U,W,Xhat] = hopls_popNet(X,Y,K,options)

narginchk(3,4);

U = []; V = []; W = []; D = []; Xhat = []; Zhat = [];
maxit = 1e3; thr = 1e-6;
if isfield(options,'maxit')
    maxit = options.maxit;
end
if isfield(options,'thr')
    thr = options.thr;
end
sx = size(X);
if length(sx)==3
    if size(Y,2)==1
        [V,tmp] = eigs(double(ttv(X,Y,3)),K);
        for i=1:K
            Zhat(:,i) = double(ttv(X,V(:,i),1))'*V(:,i);
        end        
    else
        options.proj = 1; 
        [V,D,U,W,Xhat] = hopca_popNet(squeeze(ttm(X,Y',3)),K,options);
        for i=1:K
            Zhat(:,i) = double(ttv(X,V(:,i),1))'*V(:,i);
        end        
    end
else
    if size(Y,2)==1    
        Xhat = ttv(X,Y,4);
        ns = size(Xhat);
        p = ns(1); m = ns(3);
        for k=1:K
            if isfield(options,'startv')
                v = options.startv(:,k);
            else
                v = randn(p,1); v = v/norm(v);
            end
            if isfield(options,'startu')
                u = options.startu(:,k);
            else
                u = randn(m,1); u = u/norm(u);
            end
            if k==1
                Pv = eye(p);
                Pu = eye(m);
            else
                Pv = eye(p) - V*V';
                Pu = eye(m) - U*U';
            end            
            obj = (Pv*v)'*double(ttv(Xhat,Pv*v,2))*(Pu*u);
            ind = 1; iter = 1; 
            while ind>thr & maxit>iter
                oldo = obj(end);
                uhat = Pu*double(ttv(Xhat,v,1))'*v;
                u = uhat/norm(uhat);
                [v,tmp] = eigs(Pv*double(ttv(Xhat,u,3))*Pv,1);
                obj = [obj (Pv*v)'*double(ttv(Xhat,Pv*v,2))*(Pu*u)];
                ind = abs((obj(end) - oldo)/obj(1));
                iter = iter + 1;
            end                
            d = v'*double(ttv(Xhat,u,3))*v;
            Zhat = [Zhat double(ttv(ttv(X,v,1),v,1))'*u];
            Xhat = Xhat - full(ktensor(d,v,v,u));
            V = [V v]; U = [U u]; D = [D d];   
        end        
    else
        Xhat = ttm(X,Y',4);
        ns = size(Xhat);
        p = ns(1); m = ns(3); n = ns(4);
        for k=1:K
            if isfield(options,'startv')
                v = options.startv(:,k);
            else
                v = randn(p,1); v = v/norm(v);
            end
            if isfield(options,'startu')
                u = options.startu(:,k);
            else
                u = randn(m,1); u = u/norm(u);
            end
            if isfield(options,'startw')
                w = options.startw(:,k);
            else
                w = randn(n,1); w = w/norm(w);
            end
            if k==1
                Pv = eye(p);
                Pu = eye(m);
            else
                Pv = eye(p) - V*V';
                Pu = eye(m) - U*U';
            end            
            obj = (Pu*u)'*double(ttv(ttv(Xhat,Pv*v,1),Pv*v,1))*w;
            ind = 1; iter = 1;
            while ind>thr & maxit>iter
                oldo = obj(end);
                uhat = Pu*double(ttv(ttv(Xhat,v,1),v,1))*w;
                u = uhat/norm(uhat);
                what = double(ttv(ttv(Xhat,v,1),v,1))'*u;
                w = what/norm(what);
                [v,tmp] = eigs(Pv*double(ttv(ttv(Xhat,u,3),w,3))*Pv,1);
                obj = [obj (Pu*u)'*double(ttv(ttv(Xhat,Pv*v,1),Pv*v,1))*w];
                ind = abs((obj(end) - oldo)/obj(1));
                iter = iter + 1;
            end    
            d = u'*double(ttv(ttv(Xhat,v,1),v,1))*w;
            Zhat = [Zhat double(ttv(ttv(X,v,1),v,1))'*u];
            Xhat = Xhat - full(ktensor(d,v,v,u,w));
            V = [V v]; U = [U u]; W = [W w]; D = [D d];
        end
    end            
end
