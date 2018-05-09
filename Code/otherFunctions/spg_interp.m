function Dout = spg_interp( D,Dobs,Dapprox,model,Mask)

Dout = gather(zeros(size(D)));
for i =1:size(D,3)
    
    D1                  = gather(D(:,:,i));
    Dsub1               = gather(Dobs(:,:,i));
    Dapp                = gather(Dapprox(:,:,i))*model.W;
    alpha2              = 100;
    alpha1              = 10;
    MH                  = opMH(size(D1,1),size(D1,2));
    n                   = size(D1,1);
    m                   = size(D1,2);
    if model.W==1
        nsim = m;
    else
        nsim = size(model.W,2);
    end
    Afun                = @(x,mode)funfor(x,mode,n,m,alpha2,Mask,MH,model.W,nsim);
    b                   = [sqrt(alpha2)*Dsub1(:);Dapp(:)];
    options             = struct();
    options.verbosity   = 1;
    options.iterations  = 100;
    options.project     = @(x,weight,tau) NormNuc_project(n,2*m-1,x,tau/alpha1);
    options.primal_norm = @(x,weight)     NormNuc_primal(n,2*m-1,x);
    options.dual_norm   = @(x,weight)     NormNuc_dual(n,2*m-1,x);
    sigma               = norm(Dsub1(:))/1e5;
    [X,r,g,data]        = spgl1(Afun,b,[],sigma,[],options);
    X                   = reshape(MH'*X(:),n,m);
    Dout(:,:,i)         = X;
end
Dout = distributed(Dout);
end

