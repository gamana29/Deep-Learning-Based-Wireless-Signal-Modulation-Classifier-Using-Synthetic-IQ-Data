function y = rician_fading(x, K)
    % rician_fading: apply Rician fading with K factor
    if nargin < 2
        K = 3;
    end
    h_LOS = sqrt(K/(K+1));
    h_NLOS = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2*(K+1));
    h = h_LOS + h_NLOS;
    y = h .* x;
end

