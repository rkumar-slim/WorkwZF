function [f,g] = misfit_regularization(m,Q,Dobs,model,alpha3)

r          = F(m,Q,model) - Dobs;
% objective value
f          = .5*norm(r(:))^2 + alpha3*0.5*norm(m(:))^2;
g          =  DF(m,Q,r,-1,model) + alpha3*m(:);
%figure(1);imagesc(reshape(m,model.n));drawnow;
end
