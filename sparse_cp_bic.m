%%%%%%%%%%%%%%%%
%function to compute sparse CP

%lamu, lamv, lamw must be all of the same length
%biclam is an indicator of whether to test u,v, or w


function[U,V,W,D,Xhat,bicmat,optlams] = sparse_cp_bic(x,K,lamu,lamv,lamw,biclam,orthog,startu,startv,startw,posu,posv,posw,maxit)
U = []; V = []; W = []; D = [];
Xhat = x;
ns = size(x);
n = ns(1); p = ns(2); q = ns(3);
r = length(lamu); setr = 1:r;
bicmat = zeros(r,K); optlams = [];
Us = zeros(n,r); Vs = zeros(p,r); Ws = zeros(q,r); ds = zeros(r,1);
for k=1:K
    for j=1:r
        if norm(startu)==0
            u = randn(n,1);
            v = randn(p,1);
            w = randn(q,1);
        else
            u = startu(:,k);
            v = startv(:,k);
            w = startw(:,k);
        end
        ind = 1; iter = 0; thr = 1e-6;
        while ind>thr & maxit>iter
            oldu = u; oldv = v; oldw = w;
            if lamu(j)==0 & k>1 & orthog==1
                if norm(v)>0 & norm(w)>0
                    Xtild = ttm(Xhat,eye(n) - U(:,1:(k-1))*U(:,1:(k-1))',1);
                    uhat = double(ttv(Xtild,v,2))*w;
                    u = uhat/norm(uhat,2);
                else
                    u = zeros(ns(1),1);
                end            
            else            
                uhat = soft_thr(double(ttv(Xhat,v,2))*w,lamu(j),posu);
                if norm(uhat)==0
                    u = zeros(ns(1),1);
                else
                    u = uhat/norm(uhat,2);
                end
            end
            if lamv(j)==0 & k>1 & orthog==1
                if norm(u)>0 & norm(w)>0                
                    Xtild = ttm(Xhat,eye(p) - V(:,1:(k-1))*V(:,1:(k-1))',2);
                    vhat = double(ttv(Xtild,u,1))*w;
                    v = vhat/norm(vhat,2);
                else
                    v = zeros(ns(2),1);
                end
            else            
                vhat = soft_thr(double(ttv(Xhat,u,1))*w,lamv(j),posv);
                if norm(vhat)==0
                    v = zeros(ns(2),1);
                else
                    v = vhat/norm(vhat,2);
                end
            end
            if lamw(j)==0 & k>1 & orthog==1
                if norm(u)>0 & norm(v)>0
                    Xtild = ttm(Xhat,eye(q) - W(:,1:(k-1))*W(:,1:(k-1))',3);
                    what = double(ttv(Xtild,u,1))'*v;
                    w = what/norm(what,2);
                else
                    w = zeros(ns(3),1);
                end
            else            
                what = soft_thr(double(ttv(Xhat,u,1))'*v,lamw(j),posw);
                if norm(what)==0
                    w = zeros(ns(3),1);
                else
                    w = what/norm(what,2);
                end
            end        
            ind = norm(oldu - u)/norm(oldu) + norm(oldv - v)/norm(oldv) + norm(oldw - w)/norm(oldw);
            iter = iter + 1;
        end
        d = v'*double(ttv(Xhat,u,1))*w;
        switch biclam
          case 'u'
            df = sum(u~=0);
          case 'v'
            df = sum(v~=0);
          case 'w'
            df = sum(w~=0);
        end
        xr = sum((double(reshape(Xhat - full(ktensor(d,u,v,w)),[n*p*q 1])).^2));
        bicmat(j,k) = log( xr / (n*p*q) ) + (log(n*p*q)/(n*p*q)).*df;
        Us(:,j) = u; Vs(:,j) = v; Ws(:,j) = w; ds(j) = d;
    end
    ind = bicmat(:,k)==min(bicmat(:,k));
    if sum(ind)>1
        ind = min(setr(ind));
    end
    switch biclam
      case 'u'
        optlams = [optlams; lamu(ind)];
      case 'v'
        optlams = [optlams; lamv(ind)];
      case 'w'
        optlams = [optlams; lamw(ind)];
    end
    Xhat = Xhat - full(ktensor(ds(ind),Us(:,ind),Vs(:,ind),Ws(:,ind)));
    U = [U Us(:,ind)]; V = [V Vs(:,ind)]; W = [W Ws(:,ind)]; D = [D ds(ind)];    
end



