function [f1,f2] = NLfunForward_joint(x,g,params)
e = params.numr*params.nr;
L = x(1:e);
R = x(e+1:end);
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    f1 = [params.afun(L*R');sqrt(params.alpha)*vec(L*R')];
    f2 = 0;
else 
    g1 = g(1:params.numr*params.numc);
    g2 = g(params.numr*params.numc+1:end);
    fp = params.afunT(g1)+reshape(sqrt(params.alpha)*g2,params.numr,params.numc);
    f1 = [vec(fp*R); vec(fp'*L)];
    f2 = vec(fp);
end
end