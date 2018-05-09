function [Dout,SNR] = spg_interp1( Dobs,model,Mask,rank)

Dout         = gather(zeros(size(Dobs,1),size(model.W,2),length(model.freq)));
for i =1:size(Dobs,3) 
    Dsub1                = gather(Mask.*Dobs(:,:,i));
    MH                   = opMH(model.nrec,model.nsrc);
    nr                   = model.nrec;
    nc                   = model.nsrc; 
    params.numr          = nr;
    params.numc          = 2*nc-1;
    params.funForward    = @NLfunForward;
    params.afunT         = @(x)reshape(x,nr,2*nc-1);
    params.mode          = 1;
    params.ls            = 1;
    params.logical       = 0;
    b                    = MH*vec(Dsub1);
    params.Ind           = find(b==0);
    params.afun          = @(x)afun(x,params.Ind,params);
    params.nr            = rank;
    [U, E, V]            = svds(reshape(b, [params.numr, params.numc]),params.nr);
    Linit                = vec(U*sqrt(E));
    Rinit                = vec(V*sqrt(E)');
    xinit                = 1e-8*[Linit;Rinit];
    
    tau                  = norm(xinit,1);
    sigma                = 1e-4*norm(vec(b),'fro');
    params.funPenalty    = @funLS;
    
    opts    = spgSetParms('project', @TraceNorm_project_hassan, ...
                        'primal_norm', @TraceNorm_primal, ...
                        'dual_norm', @TraceNorm_dual, ...
                        'proxy', 1, ...
                        'ignorePErr', 1, ...
                        'iterations', 400,...
                        'verbosity', 0);
    
    [xLS,~,~,info]       = spgl1(@NLfunForward,b(:),tau,sigma,xinit,opts,params);
    e                    = params.numr*params.nr;
    L1                   = xLS(1:e);
    R1                   = xLS(e+1:end);
    L1                   = reshape(L1,params.numr,params.nr);
    R1                   = reshape(R1,params.numc,params.nr);
    X                    = reshape(MH'*vec(L1*R1'),nr,nc);
    Dout(:,:,i)          = X*model.W;
    SNR(i)               = -20*log10(norm(Dobs(:,:,i)-X,'fro')/norm(Dobs(:,:,i)));
end
Dout = distributed(Dout);
end

