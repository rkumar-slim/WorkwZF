function [x,y] = odn2grid(o,d,n)
% produces grid from {o,d,n} info
% see also grid2odn
% 
% Tristan van Leeuwen, 2011
% tleeuwen@eos.ubc.ca
%
% use:
%   [x,y,z] = odn2grid(o,d,n)
%
% input:
%   {o,d,n} - 3 vectors of length 3 describing grid in each dimension.
%
% output:
%   {x,y,z} - grid in each dimension: x = o(1) + [0:n(1)-1]*d(1), etc.
%

x = o(1) + [0:n(1)-1]*d(1);
y = o(2) + [0:n(2)-1]*d(2);
% z = o(3) + [0:n(3)-1]*d(3);



