%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function to compute HOPCA for a population of networks
%takes a semi-symmetric tensor X - p x p x m ( x n) - 3D or 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change - V's, U's and W's are orthogonal
%To Do: very dependent on starting values at each k for u, v, w
%within the for loop for each k, we should start at 5 random
%initalizations and take the one resulting in the highest objective
%value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%X - the p x p x m (x n) tensor 
%K - # of factors to extract
%options - a struct with possible values:
%options.maxit - max iteration
%options.thr - threshold to establish convergence
%options.startv - a p x K matrix of intialization values fo v
%options.startu and .startw are analogous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%V - a p x K factor for the network mode
%D - a K length scaling vector for the factors
%U - an m x K factor for the measure mode
%W - an n x K factor for the subject mode (null if X is 3D)
%Xhat - residual tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[V,D,U,W,Xhat,obj] = hopca_popNet_new(X,K,options)

narginchk(2, 3);

U = []; V = []; W = []; D = [];
Xhat = X;
ns = size(X);
proj = 1;
maxit = 1e3; thr = 1e-6;
if isfield(options,'proj')
    proj = options.proj;
end
if isfield(options,'maxit')
    maxit = options.maxit;
end
if isfield(options,'thr')
    thr = options.thr;
end

%local iteration
local_search_ITER = 5;

if length(ns)==3
    p = ns(1); m = ns(3);
    for k=1:K
        
        % for each factor
        localsiter = 0;
        
        while( ~isfield(options,'startv') & local_search_ITER>localsiter)
            
            localsiter = localsiter +1;
            
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
                %Pv = eye(p);
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

            tmp_Xhat = Xhat - full(ktensor(d,v,v,u));
            
            Obj_localsearchiter(localsiter) = norm(tmp_Xhat);
            
            record_d(localsiter) = d;
            record_v{localsiter} = v;
            record_u{localsiter} = u;
        end
        [obj_min,obj_idx]=min(Obj_localsearchiter);
        v = record_v{obj_idx};
        u = record_u{obj_idx};
        d = record_d(obj_idx);
        
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
        if k==1
            Pv = eye(p);
            Pu = eye(m);
            Pw = eye(n);
        else
            Pv = eye(p) - V*V';
            Pu = eye(m) - U*U';
            Pw = eye(n) - W*W';
        end            
        obj = (Pu*u)'*double(ttv(ttv(Xhat,Pv*v,1),Pv*v,1))*(Pw*w);
        ind = 1; iter = 1;
        while ind>thr & maxit>iter
            oldo = obj(end);
            uhat = Pu*double(ttv(ttv(Xhat,v,1),v,1))*w;
            u = uhat/norm(uhat);
            what = Pw*double(ttv(ttv(Xhat,v,1),v,1))'*u;
            w = what/norm(what);
            [v,tmp] = eigs(Pv*double(ttv(ttv(Xhat,u,3),w,3))*Pv,1);
            obj = [obj (Pu*u)'*double(ttv(ttv(Xhat,Pv*v,1),Pv*v,1))*(Pw*w)];
            ind = abs((obj(end) - oldo)/obj(1));
            iter = iter + 1;
        end    
        d = u'*double(ttv(ttv(Xhat,v,1),v,1))*w;
        Xhat = Xhat - full(ktensor(d,v,v,u,w));
        V = [V v]; U = [U u]; W = [W w]; D = [D d];
    end            
end
