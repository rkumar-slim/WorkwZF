function [ f,g ] = misfit(x,Q,Dfull,model)
D   = F(x, Q, model);
res = D - Dfull;
g   = DF(x,Q,res,-1,model);
f   = 0.5*norm(res(:))^2;
end

