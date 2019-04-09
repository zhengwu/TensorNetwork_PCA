%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function to compute HOPCA for a population of networks
%takes a semi-symmetric tensor X - p x p x m ( x n) - 3D or 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%V - a p x K factor for the network mode
%D - a K length scaling vector for the factors
%U - an m x K factor for the measure mode
%W - an n x K factor for the subject mode (null if X is 3D)
%Xhat - residual tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Calabrese
%To Do - figure out when complex eigenvectors arise and write a
%catch for this - seems to only occur when K is larger than the
%truth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[V,D,U,W,Xhat,Obj] = hopca_popNet(X,K,options)

narginchk(2, 3);

U = []; V = []; W = []; D = [];
Xhat = X;
ns = size(X);
proj = 1;
maxit = 1e3; thr = 1e-6;
Local_Search_Iter = 5;
localsiter = 0;
removediagonal = 0;

if isfield(options,'proj')
    proj = options.proj;
end
if isfield(options,'maxit')
    maxit = options.maxit;
end
if isfield(options,'thr')
    thr = options.thr;
end

if length(ns)==3
    
    %3-way tensor
    p = ns(1); m = ns(3);
    for k=1:K
        
        %initilize the k-th factor
        if isfield(options,'startv')
            Local_Search_Iter = 1;
        end
        
        %initial the local iter index;
        localsiter = 0;
        
        while(Local_Search_Iter>localsiter)
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
        
            if proj==0
                %if v_k does not necessary to be orthogonal to each other
                obj = v'*double(ttv(Xhat,v,2))*u;
                ind = 1; iter = 1;
                
                while ind>thr && maxit>iter
                    oldo = obj(end);
                    uhat = double(ttv(Xhat,v,1))'*v;
                    u = uhat/norm(uhat);
                    [v,tmp] = eigs(double(ttv(Xhat,u,3)),1);
                    obj = [obj v'*double(ttv(Xhat,v,2))*u];
                    ind = abs((obj(end) - oldo)/obj(1));
                    iter = iter + 1;
                end

            else
                %if v_k is orthogonal to each other
                if k==1
                    P = eye(p);
                else
                    P = eye(p) - V*V';
                end
                
                obj = (P*v)'*double(ttv(Xhat,P*v,2))*u;
                ind = 1; iter = 1;
                while ind>thr && maxit>iter
                    oldo = obj(end);
                    uhat = double(ttv(Xhat,v,1))'*v;
                    u = uhat/norm(uhat);
                    [v,tmp] = eigs(P*double(ttv(Xhat,u,3))*P,1);
                    obj = [obj (P*v)'*double(ttv(Xhat,P*v,2))*u];
                    ind = abs((obj(end) - oldo)/obj(1));
                    iter = iter + 1;
                end
                
            end
            
            d = v'*double(ttv(Xhat,u,3))*v;
            tmp_Xhat = Xhat - full(ktensor(d,v,v,u));
            
%             %explained variation
%             PCs.V = v;
%             PCs.U = u;
%             Obj_localsearchiter(localsiter) = var_explained(X,1,PCs);

            %Obj_localsearchiter(localsiter) = norm(tmp_Xhat);
            Obj_localsearchiter(localsiter) = v'*double(ttv(Xhat,v,2))*u;

            record_d(localsiter) = d;
            record_v{localsiter} = v;
            record_u{localsiter} = u;
        end
        [obj_min,obj_idx]=max(Obj_localsearchiter);
        v = record_v{obj_idx};
        u = record_u{obj_idx};
        d = record_d(obj_idx);
        
        Xhat = Xhat - full(ktensor(d,v,v,u));
        V = [V v]; U = [U u]; D = [D d];
        
        if(removediagonal==1)
            for ii = size(Xhat,3)
                for jj=1:size(Xhat,1)
                   Xhat(jj,jj,ii) = 0; 
                end
            end
        end
        
    end
else
    
    %4-way tensor
    p = ns(1); m = ns(3); n = ns(4);
    for k=1:K
        
        %initilize the k-th factor
        if isfield(options,'startv')
            Local_Search_Iter = 1;
        end
        
        %initial the local iter index;
        localsiter = 0;
        
        while(Local_Search_Iter>localsiter)
            localsiter = localsiter +1;
            
            if isfield(options,'startv')
                v = options.startv(:,k);
            else
                v = randn(p,1); v = v/norm(v);
            end
            
            if isfield(options,'startu') && k<(size(options.startu,2)+0.5)
                u = options.startu(:,k);
            else
                u = randn(m,1); u = u/norm(u);
            end
            
            if isfield(options,'startw') && k<(size(options.startw,2)+0.5)
                w = options.startw(:,k);
            else
                w = randn(n,1); w = w/norm(w);
            end
            
            if proj==0
                %if v_k does not necessary to be orthogonal to each other
                
                obj = u'*double(ttv(ttv(Xhat,v,1),v,1))*w;
                ind = 1; iter = 1;
                
                while ind>thr && maxit>iter
                    oldo = obj(end);
                    uhat = double(ttv(ttv(Xhat,v,1),v,1))*w;
                    u = uhat/norm(uhat);
                    
                    what = double(ttv(ttv(Xhat,v,1),v,1))'*u;
                    w = what/norm(what);
                    
                    [v,tmp] = eigs(double(ttv(ttv(Xhat,u,3),w,3)),1);
                    
                    obj = [obj u'*double(ttv(ttv(Xhat,v,1),v,1))*w];
                    ind = abs((obj(end) - oldo)/obj(1));
                    iter = iter + 1;
                end
            else
                if k==1
                    P = eye(p);
                else
                    P = eye(p) - V*V';
                end
                
                obj = u'*double(ttv(ttv(Xhat,P*v,1),P*v,1))*w;
                ind = 1; iter = 1;
                while ind>thr && maxit>iter
                    oldo = obj(end);
                    uhat = double(ttv(ttv(Xhat,v,1),v,1))*w;
                    u = uhat/norm(uhat);
                    what = double(ttv(ttv(Xhat,v,1),v,1))'*u;
                    w = what/norm(what);
                    [v,tmp] = eigs(P*double(ttv(ttv(Xhat,u,3),w,3))*P,1);
                    obj = [obj u'*double(ttv(ttv(Xhat,P*v,1),P*v,1))*w];
                    ind = abs((obj(end) - oldo)/obj(1));
                    iter = iter + 1;
                end
            end
            d = u'*double(ttv(ttv(Xhat,v,1),v,1))*w;
            tmp_Xhat = Xhat - full(ktensor(d,v,v,u,w));
            
            Obj_localsearchiter(localsiter) = u'*double(ttv(ttv(Xhat,v,1),v,1))*w;
            
            record_d(localsiter) = d;
            record_v{localsiter} = v;
            record_u{localsiter} = u;
            record_w{localsiter} = w;
        end
        
         [obj_min,obj_idx]=max(Obj_localsearchiter);
        v = record_v{obj_idx};
        u = record_u{obj_idx};
        w = record_w{obj_idx};
        d = record_d(obj_idx);
                    
        Xhat = Xhat - full(ktensor(d,v,v,u,w));
        V = [V v]; U = [U u]; W = [W w]; D = [D d];
    end            
end
Obj = norm(Xhat);
