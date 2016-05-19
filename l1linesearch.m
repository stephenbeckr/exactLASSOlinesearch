function t = l1linesearch( A, b, x0, p, lambda )
% t = l1linesearch(A,b,x0,p)
%   computes the scalar step length "t" that minimizes the 1D function:
%
%   phi(t)  = f( x0 + t*p )
% where
%   f( x ) = x'*A*x/2 - b'*x + norm(x,1);
% and A is Hermitian/symmetric positive semidefinite
% ( "A" can either be an explicit matrix or a function handle )
% If A is provided as a vector, then this function assumes this parameter
% is really A*p.  You may wish to use this form if you have pre-calculated A*p.
% It is the user's responsibility to make sure A is really symmetric.
% Note: this code has not been designed for complex numbers.
%
% t = l1linesearch(A,b,x0,p, lambda)
% uses
%   f( x ) = x'*A*x/2 - b'*x + lambda*norm(x,1);
%
% The computational complexity of this routine is on the order of the cost of
% sorting length(x0) numbers, plus the cost of multiplying A*p (if necessary).
%
% See the compantion arXiv tech report for description of the algorithm.
%
% Stephen Becker, srbecker@caltech.edu, Jan 27, 2011
%  Edited: stephen.becker@colorado.edu, May 19, 2016

if nargin < 5 || isempty(lambda)
    lambda = 1;
end

if isa(A,'function_handle')
    Ap = A(p);
elseif isvector(A)
    Ap = A;
else
    Ap = A*p;
end

% collect terms, so we can write g(t) = c1/2*t^2 + c2*t + c3 + ||...||_1
const1  = p'*Ap;     % c1 > 0 by pos def
const2  = x0'*Ap - b'*p;
if abs(const1) < 0
    error('matrix "A" must be positive semidefinite');
end

%{
 Solve for a zero in the subdifferential:
   0 \in c1*t + c2 + lambda*p'*sign( x0+t*p )
%}
lambda  = lambda/const1;
const2  = const2/const1;

if any(~p)
    % any components of p that are zero can just be removed
    x0 = x0(~~p);
    p  = p(~~p);
end

[turningPoints,ind]   = sort(-x0./p); % sorted from small to large
t0      = turningPoints(1);
if t0 < 0, t0 = 1.2*t0; else t0 = 0.8*t0; end
sgn0    = sign(x0 + t0*p ); % no more use for x0 after this point except debugging
psgn    = p.*sgn0;
psgn    = psgn(ind);
cs      = cumsum(psgn);   % this is sorted from large to small
pSum    = cs(end);      
tRHS    = -const2 +  -lambda*(pSum - 2*cs );
ind     = find( turningPoints < tRHS, 1, 'last' );
if isempty(ind)
    t    = tRHS(1);
elseif ind == length(tRHS)
    t   = tRHS(end);
else
    j = ind+1;
    t = turningPoints(j);
    s = 1/(lambda*psgn(j))*( -t - const2 - lambda*(pSum - 2*cs(j-1) ) );
    if abs(s) <= 1
        % This is fine.  It means we pick "t" such that it
        % exactly zeros out a new component in x0 + t*p.
        return;
    else
        % This is the usual case.  There are no zeros in x0 + t*p
        % so there is no ambiguity in the subdifferential.
        t = tRHS(ind);
    end
end
