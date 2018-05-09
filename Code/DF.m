function output = DF(m,Q,input,flag,model)

ot = model.o-model.nb(1,:).*model.d;
dt = model.d;
nt = model.n+2*model.nb(1,:);
[zt,xt] = odn2grid(ot,dt,nt);

% data size
nsrc   = size(Q*model.W,2);
nrec   = length(model.zrec)*length(model.xrec);
nfreq  = length(model.freq);
% define wavelet
w = exp(1i*2*pi*model.freq*model.t0);
if model.f0
    % Ricker wavelet with peak-frequency model.f0
    w = (model.freq).^2.*exp(-(model.freq/model.f0).^2).*w;
end

% mapping from source/receiver/physical grid to comp. grid
Pr = opKron(opLInterp1D(xt,model.xrec),opLInterp1D(zt,model.zrec));
Ps = opKron(opLInterp1D(xt,model.xsrc),opLInterp1D(zt,model.zsrc));
Px = opKron(opExtension(model.n(2),model.nb(2)),opExtension(model.n(1),model.nb(1)));
Pe = opKron(opExtension(model.n(2),model.nb(2),0),opExtension(model.n(1),model.nb(1),0));

% model parameter: slowness [s/m] on computational grid.
freq = distributed(model.freq);
w    = distributed(w);
m    = Px*m;

if flag==1
    % solve Helmholtz for each frequency in parallel
    spmd
        codistr   = codistributor1d(2,codistributor1d.unsetPartition,[nsrc*nrec,length(freq)]);
        freqloc   = getLocalPart(freq);
        wloc      = getLocalPart(w);
        nfreqloc  = length(freqloc);
        outputloc = zeros(nsrc*nrec,nfreqloc);
        for k = 1: nfreqloc
            [Hk,dH]   = Helm2D_opt(m,dt,nt,model.nb,model.unit,freqloc(k),model.f0);
            U0k       = Hk\(wloc(k)*(Ps'*Q*model.W));
            Sk        = -(dH*(U0k.*repmat(Px*input,1,nsrc)));
            U1k       = Hk\Sk;
            outputloc(:,k) = vec(Pr*U1k);
        end
        output = codistributed.build(outputloc,codistr,'noCommunication');
    end
    output = vec(output); 
else
    input = invvec(input,[nsrc*nrec,nfreq]);
    spmd
        freqloc   = getLocalPart(freq);
        wloc      = getLocalPart(w);
        nfreqloc  = length(freqloc);
        outputloc = zeros(prod(model.n),1);
        inputloc  = getLocalPart(input);
        inputloc  = reshape(inputloc,[nsrc*nrec,nfreqloc]);
        for k = 1:nfreqloc
            [Hk,dH]   = Helm2D_opt(m,dt,nt,model.nb,model.unit,freqloc(k),model.f0);
            dH        = 1;
            U0k       = Hk\(wloc(k)*(Ps'*Q*model.W));
            Sk        = -Pr'*reshape(inputloc(:,k),[nrec nsrc]);
            V0k       = Hk'\Sk;
            r         = real(sum(conj(U0k).*(dH'*V0k),2));
            outputloc = outputloc + Pe'*r;
        end
        output = pSPOT.utils.global_sum(outputloc);
    end
    output = output{1};
end

