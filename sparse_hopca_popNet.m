%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function to compute sparse and/or non-negative HOPCA for a
%population of networks 
%takes a semi-symmetric tensor X - p x p x m ( x n) - 3D or 4D
%note - sparsity and / or non-negativity only on U's and W's -
%might aid in interpretability in certain siutations - no sparsity
%on V's
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%X - the p x p x m (x n) tensor 
%K - # of factors to extract
%options - a struct with possible values:
%options.proj - an indicator for whether V should be orthognal (1 =
%yes; 0 = no)
%options.maxit - max iteration
%options.thr - threshold to establish convergence
%options.startv - a p x K matrix of intialization values fo v
%options.startu and .startw are analogous
%options.lamu - a non-negative scalar penalty parameter for the L1
%norm of u - .lamw analogous
%options.posu - an indicator of whether a non-negativity constraint
%on u should be used (1 = yes; 0 = no) .posw analogous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%V - a p x K factor for the network mode
%D - a K length scaling vector for the factors
%U - an m x K factor for the measure mode
%W - an n x K factor for the subject mode (null if X is 3D)
%Xhat - residual tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To Do - 1) figure out when complex eigenvectors arise and write a
%catch for this - seems to only occur when K is larger than the
%truth
%2) do this for a sequence of lambda's and lambda's selected via
%nester BIC criterion (see sparse_cp_bic.m for how I set this up
%for AISTATS paper; I think this can be done here as well)
%%%%%%%%%%%%%%%%%%%%%%%%%

function[V,D,U,W,Xhat] = sparse_hopca_popNet(X,K,options)

narginchk(2, 3);

U = []; V = []; W = []; D = [];
Xhat = X;
ns = size(X);
proj = 1;
maxit = 1e3; thr = 1e-6;
lamu = 0; lamw = 0;
posu = 0; posw = 0;
if isfield(options,'proj')
    proj = options.proj;
end
if isfield(options,'maxit')
    maxit = options.maxit;
end
if isfield(options,'thr')
    thr = options.thr;
end
if isfield(options,'lamu')
    lamu = options.lamu;
end
if isfield(options,'lamw')
    lamw = options.lamw;
end
if isfield(options,'posu')
    posu = options.posu;
end
if isfield(options,'posw')
    posw = options.posw;
end

if length(ns)==3
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
        if proj==0
            obj = v'*double(ttv(Xhat,v,2))*u-lamu*norm(u,1);
            ind = 1; iter = 1;
            while ind>thr & maxit>iter
                oldo = obj(end);
                uhat = soft_thr(double(ttv(Xhat,v,1))'*v,lamu,posu);
                if norm(uhat)==0
                    u = zeros(m,1);
                else
                    u = uhat/norm(uhat);
                end                
                [v,tmp] = eigs(double(ttv(Xhat,u,3)),1);
                obj = [obj v'*double(ttv(Xhat,v,2))*u-lamu*norm(u,1)];
                ind = abs((obj(end) - oldo)/obj(1));
                iter = iter + 1;
            end    
        else
            if k==1
                P = eye(p);
            else
                P = eye(p) - V*V';
            end            
            obj = (P*v)'*double(ttv(Xhat,P*v,2))*u -lamu*norm(u,1);
            ind = 1; iter = 1; 
            while ind>thr & maxit>iter
                oldo = obj(end);
                uhat = soft_thr(double(ttv(Xhat,v,1))'*v,lamu,posu);
                if norm(uhat)==0
                    u = zeros(m,1);
                else
                    u = uhat/norm(uhat);
                end                
                [v,tmp] = eigs(P*double(ttv(Xhat,u,3))*P,1);
                obj = [obj (P*v)'*double(ttv(Xhat,P*v,2))*u-lamu*norm(u,1)];
                ind = abs((obj(end) - oldo)/obj(1));
                iter = iter + 1;
            end                
        end
        d = v'*double(ttv(Xhat,u,3))*v;
        Xhat = Xhat - full(ktensor(d,v,v,u));
        V = [V v]; U = [U u]; D = [D d];
    end            
else
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
        if proj==0
            obj = u'*double(ttv(ttv(Xhat,v,1),v,1))*w-lamu*norm(u,1)-lamw*norm(w,1);
            ind = 1; iter = 1;
            while ind>thr & maxit>iter
                oldo = obj(end);
                uhat = soft_thr(double(ttv(ttv(Xhat,v,1),v,1))*w,lamu,posu);
                if norm(uhat)==0
                    u = zeros(m,1);
                else
                    u = uhat/norm(uhat);
                end                
                what = soft_thr(double(ttv(ttv(Xhat,v,1),v,1))'*u,lamw,posw);
                if norm(what)==0
                    w = zeros(n,1);
                else
                    w = what/norm(what);
                end                
                [v,tmp] = eigs(double(ttv(ttv(Xhat,u,3),w,3)),1);
                obj = [obj u'*double(ttv(ttv(Xhat,v,1),v,1))*w-lamu*norm(u,1)-lamw*norm(w,1)];
                ind = abs((obj(end) - oldo)/obj(1));
                iter = iter + 1;
            end    
        else
            if k==1
                P = eye(p);
            else
                P = eye(p) - V*V';
            end            
            obj = u'*double(ttv(ttv(Xhat,P*v,1),P*v,1))*w-lamu*norm(u,1)-lamw*norm(w,1);
            ind = 1; iter = 1;
            while ind>thr & maxit>iter
                oldo = obj(end);
                uhat = soft_thr(double(ttv(ttv(Xhat,v,1),v,1))*w,lamu,posu);
                if norm(uhat)==0
                    u = zeros(m,1);
                else
                    u = uhat/norm(uhat);
                end                
                what = soft_thr(double(ttv(ttv(Xhat,v,1),v,1))'*u,lamw,posw);
                if norm(what)==0
                    w = zeros(n,1);
                else
                    w = what/norm(what);
                end                
                [v,tmp] = eigs(P*double(ttv(ttv(Xhat,u,3),w,3))*P,1);
                obj = [obj u'*double(ttv(ttv(Xhat,P*v,1),P*v,1))*w-lamu*norm(u,1)-lamw*norm(w,1)];
                ind = abs((obj(end) - oldo)/obj(1));
                iter = iter + 1;
            end    
        end
        d = u'*double(ttv(ttv(Xhat,v,1),v,1))*w;
        Xhat = Xhat - full(ktensor(d,v,v,u,w));
        V = [V v]; U = [U u]; W = [W w]; D = [D d];
    end            
end
