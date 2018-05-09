clear;clc; close all
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
expdir = pwd;
addpath(genpath(pwd));
load('marmousiWater125x400.mat')
v1              = v1(:,1:200);
v               = v1*1e3;
n               = size(v);
clearvars v1
n               = size(v);
o               = [0 0];
d               = [10 10];
[z,x]           = odn2grid(o,d,n);
S               = opKron(opSmooth(n(2),300),opSmooth(n(1),300));
m               = 1e6./v(:).^2;
mtrue           = m;
m0              = S*m; %vec(repmat(mean(reshape(S*m,n),2),1,n(2))); %S*m is the blurred model
%% Generate data
model.o         = o;
model.d         = d;
model.n         = n;
model.nb        = [60 60;60 60];
model.xsrc      = x; %x(1:2:end); 
nsrc            = length(model.xsrc);
model.zsrc      = [d(1)];
model.xrec      = x; %x(1:2:end);    
nrec            = length(model.xrec);
model.zrec      = [d(1)];
model.f0        = 10;   %center frequency. Curt origianl set to 15
model.t0        = 0;
subf            = 0.2;
freq            = [(3:subf:4)' (4:subf:5)' (5:subf:6)' (6:subf:7)' (7:subf:8)' (8:subf:9)' (9:subf:10)',...
                    (10:subf:11)' (11:subf:12)' (12:subf:13)' (13:subf:14)' (14:subf:15)' (15:subf:16)' (16:subf:17)',...
                    (17:subf:18)' (18:subf:19)' (19:subf:20)' (20:subf:21)' (21:subf:22)' (22:subf:23)' (23:subf:24)' (24:subf:25)'];

% overlap         = 2;   %change frequency batch when change this value
% size_freq_batch = 3; %number of frequencies in a band
% freq_partition  = partition(nfreq,size_freq_batch,overlap);
Q               = eye(nsrc);
model.unit      = 's2/km2';
%Generate masking matrix randomly missing columns
% %Generate missing columns for FWI
Mask            = zeros(nrec,nsrc);
inds            = randperm(nsrc);
perc            = 0.2;
pos             = jittersamp_exact(nsrc,perc);
Mask(:,pos)     = 1;
model.nrec      = nrec;
model.nsrc      = nsrc;
rank            = floor(linspace(40,100,size(freq,2)));
%% WEMVA-FWI
for j = 1:5
    K                = 20;
    W                = randn(nsrc,K);
    model.W          = W; % sign(randn(nsrc,K));
    % generate fully sampled data
    model.freq       = freq(:,j);
    model.W          = 1;
    Dtrue            = F(mtrue, Q, model);
    Dtrue            = reshape(Dtrue, nrec, nsrc, length(model.freq));
    model.W          = W;    
    % interpolate full data without PDE constraint
    [Dfull,SNR]      = spg_interp1(Dtrue,model,Mask,rank(j));  % <****************why is this calling a mask?
    SNR
    options.maxIter  =  15;
    options.method   = 'lbfgs';
    LB               = (1e6./max(v(:)).^2)*ones(size(m0));
    UB               = (1e6./min(v(:)).^2)*ones(size(m0));
    fobj             = @(x)misfit(x,Q,Dfull(:),model);          %need to put Dfull back into vector form here
    m0               = minConf_TMP(fobj,m0,LB,UB,options);      %minConf_SPG(fh,m0,funProj,options);
    figure(1);imagesc(reshape(m0,model.n));drawnow;
end
