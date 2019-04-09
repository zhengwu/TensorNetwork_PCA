%%%%%%%%%%%%%%%%%%%%%
%4/08/19
%toy simulations to test code

%requires Matlab Tensor Toolbox
addpath('tensor_toolbox') 
addpath('tensor_toolbox/met')

%3D case
p = 10;
m = 5;
K = 2;
Dt = [100 50]';

%generating tensor X with dim p*p*m
xx = randn(p,500);
[Vt,tmp] = eigs(xx*xx',K);
xx = randn(m,500);
[Ut,tmp] = eigs(xx*xx',K);
X = full(ktensor(Dt,Vt,Vt,Ut)) + randn(p,p,m);

%v_k is orthogonal to each other
options = struct('proj',1);
[V,D,U,W,Xh] = hopca_popNet(X,2,options);

%v_k is not orthogonal to each other
options = struct('proj',0);
[V,D,U,W,Xh] = hopca_popNet(X,2,options);



%4D case 
p = 10;
m = 5;
n = 8;
K = 2;
Dt = [100 50]';

xx = randn(p,500);
[Vt,tmp] = eigs(xx*xx',K);
xx = randn(m,500);
[Ut,tmp] = eigs(xx*xx',K);
xx = randn(n,500);
[Wt,tmp] = eigs(xx*xx',K);
X = full(ktensor(Dt,Vt,Vt,Ut,Wt)) + randn(p,p,m,n);

options = struct('proj',0);
[V,D,U,W,Xh] = hopca_popNet(X,2,options);

options = struct('proj',1);
[V,D,U,W,Xh] = hopca_popNet(X,2,options);

options = struct('proj',1);
[V,D,U,W,Xh] = hopca_popNet(X,2,options);
options.lamu = 2.5;
options.lamw = 2.5;
options.startv = V;
options.startw = W;
options.startu = U;
[V,D,U,W,Xh] = sparse_hopca_popNet(X,2,options);



%use HOPCA to analyze and interpret population of networks
options = struct('proj',1);
[V,D,U,W,Xh] = hopca_popNet(X,2,options);

[tmp,ord] = sort(D,'descend');

%best network summary of the population (that explains most variance)
Sig = double(ttv(ttv(X,U(:,ord(1)),3),W(:,ord(1)),3));

%viz for the measures - like PC scatterplots
scatter(U(:,ord(1)),U(:,ord(2)))

%viz for the subjects - like PC scatterplots
scatter(W(:,ord(1)),W(:,ord(2)))

