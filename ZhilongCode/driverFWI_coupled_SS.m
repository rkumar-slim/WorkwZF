clear;clc; close all

ToolMainDir = '../Code';

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
addpath(genpath(ToolMainDir));
% addpath(genpath('/home/mliu/Marmousi_coupled/spot-master'));
% addpath(genpath('/home/mliu/Marmousi_coupled/WAVEFORM-master'));
% addpath(genpath('/home/mliu/Marmousi_coupled/operators'));
% addpath(genpath('/home/mliu/Marmousi_coupled/minConf'));
% addpath(genpath('/home/mliu/Marmousi_coupled/spgl1'));
% addpath(genpath('/home/mliu/Marmousi_coupled/otherFunction'));

load('marmousiWater125x400.mat')

%load('marmousiReduced101x385.mat')
%v1            = v_coarse1;  %this corresponds to marmousiReduced101x385 model. Del

v=v1*1e3;
%v             = v(1:200,400:800);         %rajiv
n             = size(v);
clearvars v1
n             = size(v);
o             = [0 0];
d             = [10 10];
[z,x]         = odn2grid(o,d,n);
S             = opKron(opSmooth(n(2),300),opSmooth(n(1),300));
m             = 1e6./v(:).^2;
mtrue         = m;
m0            = S*m; %vec(repmat(mean(reshape(S*m,n),2),1,n(2))); %S*m is the blurred model
%% Generate data
model.o       = o;
model.d       = d;
model.n       = n;
model.nb      = [60 60;60 60];
model.xsrc    = x; %x(1:2:end);
nsrc          = length(model.xsrc);
model.zsrc    = [d(1)];
model.xrec    = x; %x(1:2:end);
nrec          = length(model.xrec);
model.zrec    = [d(1)];
model.f0      = 10;   %center frequency. Curt origianl set to 15
model.t0      = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nfreq           = 35; %length(model.freq);
overlap         = 2;   %change frequency batch when change this value
size_freq_batch = 3; %number of frequencies in a band
freq_partition  = partition(nfreq,size_freq_batch,overlap);



%Generate masking matrix randomly missing columns
% %Generate missing columns for FWI
Mask       = ones(nrec,nsrc);
perc       = 0.5; %0.1; %0.9; %%.001;   %percent of missing columns
misCol     = floor(perc*nsrc);  %num of missing columns/sources
ind        =  randperm(nsrc, misCol);
for i=1:length(ind)
    Mask(:,ind(i)) = zeros(nrec,1);
end

Q             = eye(nsrc);
model.unit    = 's2/km2';
%% WEMVA-FWI
options.maxIter   =  60;
options.method    = 'lbfgs';
%for j=1:3  I think you forgot to add the forloop
LB                = (1e6./max(v(:)).^2)*ones(size(m0));
UB                = (1e6./min(v(:)).^2)*ones(size(m0));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K                 = 200;         %number of PDE solves
% j               = 1;
% model.freq      = freq_partition(j,:);
% Dapprox         = F(m0, Q, model);  % m0 is the blurred model initially)
% Dapprox         = reshape(Dapprox, nrec, nsrc, length(model.freq));
% Dfull           = Dapprox;
% D               = F(mtrue, Q, model);
% D               = reshape(D, nrec, nsrc, length(model.freq));
% Dobs            = Mask .* D;

%% SPGL1

%figure(1);imagesc([real(X) real(D1) real(D1-X)]);caxis([-1 1]*100)
%%
j=1;
%for j=1:size(freq_partition,1)  %started at mest=7.
    %perform matrix completion to fill those missing columns
    W                =  sign(randn(nsrc,K));
    model.W          = W; % sign(randn(nsrc,K));
    model.freq       = freq_partition(j,:);
    Dapprox          = F(m0, Q, model);  % m0 is the blurred model initially)
    Dapprox          = reshape(Dapprox, nrec, K, length(model.freq));  %use this only if j=1,  model.freq = freq_partition(j,:)
    model.W          = 1;
    Dtrue            = F(mtrue, Q, model);
    Dtrue            = reshape(Dtrue, nrec, nsrc, length(model.freq));
    for i = 1:size(Dobs,3)
      Dobs(:,:,i)             = Mask .* Dtrue(:,:,i);
    end
    %%%%To be deleted comments: There should be two different W going in
    %%%%here?
    model.W          = W;
    Dfull            = spg_interp(Dtrue,Dobs,Dapprox,model,Mask);  % <****************why is this calling a mask?
   % fobj             = @(x)misfit(x,Q,Dfull(:),model);          %need to put Dfull back into vector form here
    %%alpha3           = 100;
   % %fobj             = @(x) misfit_regularization(x,Q,Dobs,model,alpha3);
%    m0               = minConf_TMP(fobj,m0,LB,UB,options);      %minConf_SPG(fh,m0,funProj,options);
 %  save(['mS_Couple_K350_0missing_' num2str(j) '.mat'],'m0');
 %   end
