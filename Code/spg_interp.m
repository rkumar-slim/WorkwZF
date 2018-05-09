function [Dout,SNR] = spg_interp( Dobs,Dapprox,model,Mask)

Dout = gather(zeros(size(Dobs,1),size(model.W,2),length(model.freq)));
for i =1:size(Dobs,3)
    Dsub1               = gather(Mask.*Dobs(:,:,i));
    Dapp                = gather(Dapprox(:,:,i));
    alpha2              = 300;
    alpha1              = 30; %10;
    MH                  = opMH(model.nrec,model.nsrc);
    n                   = model.nrec;
    m                   = model.nsrc;
    if model.W==1
        nsim = m;
    else
        nsim = size(model.W,2);
    end
    Afun                = @(x,mode)funfor(x,mode,n,m,alpha2,Mask,MH,model.W,nsim);
    b                   = [sqrt(alpha2)*Dsub1(:);Dapp(:)];
    options             = struct();
    options.verbosity   = 1;
    options.iterations  = 250;
    options.project     = @(x,weight,tau) NormNuc_project(n,2*m-1,x,tau/alpha1);
    options.primal_norm = @(x,weight)     NormNuc_primal(n,2*m-1,x);
    options.dual_norm   = @(x,weight)     NormNuc_dual(n,2*m-1,x);
    sigma               = norm(Dsub1(:))/1e5;
    [X,r,g,data]        = spgl1(Afun,b,[],sigma,[],options);
    X                   = reshape(MH'*X(:),n,m);
    Dout(:,:,i)         = X*model.W;
    SNR(i)              = -20*log10(norm(Dobs(:,:,i)-X,'fro')/norm(Dobs(:,:,i)));
end
Dout = distributed(Dout);
end

