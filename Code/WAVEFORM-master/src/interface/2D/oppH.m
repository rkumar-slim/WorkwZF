classdef oppH < oppSpot
% pSPOT wrapper for the full Hessian of F.m
%
% Usage:
%   H = oppH(m,Q,D,model,{params})
%
% see PDEfunc.m for further documentation
%
%
% You may use this code only under the conditions and terms of the
% license contained in the file LICENSE provided with this source
% code. If you do not agree to these terms you may not use this
% software.

    properties
        mt,Q,D,model,params;
    end    
    methods
       function op = oppH(mt,Q,D,model,params)
           m = numel(mt);
           n = numel(mt);
           
           op = op@oppSpot('FWI Hessian', m, n);
           op.cflag     = 0;  
           op.linear    = 1;
           op.children  = []; 
           op.sweepflag = 0;
           op.mt        = mt;
           op.Q         = Q;           
           op.D         = D;           
           op.model     = model;
           if exist('params','var')==0||isempty(params)           
               params = struct; 
           end
           params = default_fwi_params2d(params);

           params.wri = false;
           op.params = params;           
       end        
    end    
    
    methods ( Access = protected )
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % Multiply
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function y = multiply(op,x,~)           
           y = PDEfunc_dist(PDEopts.HESS,op.mt,op.Q,x,op.D,op.model,op.params);
       end        
    end     
end 

    
    
    
