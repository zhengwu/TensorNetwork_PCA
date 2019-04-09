%%%%%%%%%%%%%%%%%%%%%%%
%function to compute HOOI for a population of networks
%takes a semi-symmetric tensor X - p x p x n ( x m) - 3D or 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%X - the p x p x n (x m) tensor 
%K - # of factors to extract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%V - a p x K factor for the network mode
%D - a K length scaling vector for the factors
%U - an n x K factor for the subject mode
%W - an m x K factor for the metrics mode (null if X is 3D)
%%%%%%%%%%%%%%%%%%%%%

function[V,DiagD,Dd,U,W] = hooi_popNet(X,K)

ns = size(X); 
D = 0; W = [];
[V,tmp1,tmp2] = svds(double(tenmat(X,1)),K);
[U,tmp1,tmp2] = svds(double(tenmat(X,3)),K);
maxit = 1e3; thr = 1e-6;

if length(ns)==3
    ind = 1; iter = 1;
    while ind>thr & iter<maxit
        oldV = V;
        [V,tmp1,tmp2] = svds(double(tenmat(ttm(ttm(X,V',2),U',3),1)),K);
        [U,tmp1,tmp2] = svds(double(tenmat(ttm(ttm(X,V',1),V',2),3)),K);
        ind = norm(abs(V) - abs(oldV))/norm(abs(oldV)); iter = iter + 1;
    end    
    Dd = ttm(ttm(ttm(X,V',1),V',2),U',3);
    for i=1:K
        D(i) = abs(Dd(i,i,i));
    end
    DiagD = D;
else
    [W,tmp1,tmp2] = svds(double(tenmat(X,4)),K);
    ind = 1; iter = 1;
    while ind>thr && iter<maxit
        oldV = V;
        [V,tmp1,tmp2] = svds(double(tenmat(ttm(ttm(ttm(X,V',2),U',3),W',4),1)),K);
        [U,tmp1,tmp2] = svds(double(tenmat(ttm(ttm(ttm(X,V',1),V',2),W',4),3)),K);
        [W,tmp1,tmp2] = svds(double(tenmat(ttm(ttm(ttm(X,V',1),V',2),U',3),4)),K);
        ind = norm(abs(V) - abs(oldV))/norm(abs(oldV)); iter = iter + 1;
    end            
    Dd = ttm(ttm(ttm(ttm(X,V',1),V',2),U',3),W',4);
    for i=1:min(K,size(Dd,4))
        D(i) = abs(Dd(i,i,i,i));
    end
    
    DiagD = D;
end
