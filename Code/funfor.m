function y = funfor( x,mode,n,m,alpha2,Mask,MH,W,nsim)
if mode==1
    x  = MH'*x(:);
    x  = reshape(x,n,m);
    y1 = sqrt(alpha2)*(Mask.*x);
    y2 = x*W;
    y  = [y1(:);y2(:)];
else
    x1 = reshape(x(1:n*m),n,m);
    x2 = reshape(x(n*m+1:end),n,nsim);
    y1 = sqrt(alpha2)*(Mask.*x1);
    y2 = x2*W';
    y  = y1+y2;
    y  = MH*y(:);
end
end

