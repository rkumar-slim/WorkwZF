function pos = jittersamp_exact(N,perc,seed)
    % Make a jittered sampling mask, adapted from G. Hennenfent's RSF code
    % Perc is the ratio of SELECTED positions (where mask=1)
    % In this version perc is respected much more faithfully than the non-exact version
    % Also only returns the positions
    if nargin > 2
        randn('state',seed); rand('state',seed);
    end
    
    if (perc < 0 || perc > 1)
        error('Perc must be between 0 and 1')
    end
    
    pos = generate_positions();
    % disp(['masked columns: ' int2str(length(pos))])
    % 
    % y = zeros(N,1);
    % y(pos) = 1;
    % mask = y;
    
    function pos = generate_positions()
        sub = 1/perc;
        jit = sub;
        inipos = [0:sub:N];
        pos = unique(mod(round(inipos + jit*rand(size(inipos))-jit/2),N) + 1);
        % keyboard
    end
end

