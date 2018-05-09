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
nfreq           = 35; %length(model.freq);
overlap         = 2;   %change frequency batch when change this value
size_freq_batch = 6; %number of frequencies in a band
freq_partition  = partition(nfreq,size_freq_batch,overlap);
Q               = eye(nsrc);
model.unit      = 's2/km2';
%% WEMVA-FWI
options.maxIter =  20;
options.method  = 'lbfgs';
LB              = (1e6./max(v(:)).^2)*ones(size(m0));
UB              = (1e6./min(v(:)).^2)*ones(size(m0));
K               = 20;         %number of PDE solves
j               = 1;
W               = randn(nsrc,K);
model.W         = W;
model.freq      = freq_partition(j,:);
Dtrue           = F(mtrue, Q, model);
fobj            = @(x)misfit(x,Q,Dtrue,model);          %need to put Dfull back into vector form here
m0              = minConf_TMP(fobj,m0,LB,UB,options);      %minConf_SPG(fh,m0,funProj,options);
figure(1);imagesc(reshape(m0,model.n))

