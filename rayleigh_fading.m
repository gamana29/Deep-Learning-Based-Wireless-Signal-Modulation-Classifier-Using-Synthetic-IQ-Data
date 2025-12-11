function y = rayleigh_fading(x)
    % rayleigh_fading: apply simple per-sample Rayleigh fading
    h = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2);
    y = h .* x;
end

