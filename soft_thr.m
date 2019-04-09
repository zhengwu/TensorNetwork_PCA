%%%%%%%%%%%%%%
%soft thresholding function

function[u] = soft_thr(a,lam,pos);
if pos==0
    u = sign(a).*max(abs(a) - lam,0);
else 
    u = max(a - lam,0);
end
