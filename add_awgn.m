function y = add_awgn(x, snr_dB)
    % add_awgn: add complex AWGN to a complex or real signal
    % x : vector (row or column)
    % snr_dB : desired SNR in dB
    noise = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2);
    y = x + noise * 10^(-snr_dB/20);
end

