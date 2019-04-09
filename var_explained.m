%%%%%%%%%%%%%%
%variation explained by k-th PCs

function [percent] = var_explained(X,K,PCs)
%input: X - the tensor data
%       K - variation explained by k-th PCs
%       PCs - PC for V, U, and W
%       D - scalar for first K PCs

dimX = size(X);
array_dim = 0;

if(length(dimX) == 3)
    array_dim = 3;
elseif(length(dimX) == 4)
    array_dim = 4;
else
    array_dim = -1; 
end

% if we are dealing with 3 way tensor
if(array_dim==3)
    %get the first k PCs
    V = PCs.V;
    U = PCs.U;
    
    if(K>size(V,1))
        PV = eye(size(V,1),size(V,1));
    else
        PV = V(:,1:K)*inv(V(:,1:K)'*V(:,1:K))*V(:,1:K)';
    end;
    
    if(K>size(U,1))
        PU = eye(size(U,1),size(U,1));
    else
        PU = U(:,1:K)*inv(U(:,1:K)'*U(:,1:K))*U(:,1:K)';
    end
    
    ProjX = ttm(X,{PV,PV,PU});
    
    percent = norm(ProjX)^2/(norm(X)^2);
end

% if we are dealing with 4 way tensor
if(array_dim==4)
    %get the first k PCs
    tV = PCs.V;
    tU = PCs.U;
    tW = PCs.W;
    
    if(K>size(tV,1))
        PV = eye(size(tV,1),size(tV,1));
    else
        PV = tV(:,1:K)*inv(tV(:,1:K)'*tV(:,1:K))*tV(:,1:K)';
    end;
    
    if(K>size(tU,1))
        PU = eye(size(tU,1),size(tU,1));
    else
        PU = tU(:,1:K)*inv(tU(:,1:K)'*tU(:,1:K))*tU(:,1:K)';
    end
    
    if(K>size(tW,1))
        PW =  eye(size(tW,1),size(tW,1));
    else
        PW = tW(:,1:K)*inv(tW(:,1:K)'*tW(:,1:K))*tW(:,1:K)';
    end
    
    ProjX = ttm(X,{PV,PV,PU,PW});
    %ProjX = ttm(X,PV,1);
    
    percent = norm(ProjX)^2/(norm(X)^2);
end;
