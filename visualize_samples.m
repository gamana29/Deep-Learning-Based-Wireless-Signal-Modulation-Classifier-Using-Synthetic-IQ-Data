clear all; close all; clc;

load X.mat
load Y.mat

mods = {"BPSK","QPSK","QAM16","AM","FM","FSK"};

% Pick random sample
idx = randi(size(X,1));
sig = X(idx, :);
label = mods{Y(idx)};

fprintf("Showing sample of class: %s\n", label);

% -------- TIME DOMAIN --------
figure;
plot(real(sig));
title([label " - Time Domain"]);
xlabel("Sample Index");
ylabel("Amplitude");

pause;  % wait for keypress



% -------- FFT SPECTRUM --------
figure;
N = length(sig);
f = linspace(-0.5, 0.5, N);
plot(f, abs(fftshift(fft(sig))));
title([label " - Frequency Spectrum"]);
xlabel("Normalized Frequency");
ylabel("Magnitude");

pause;



% -------- CONSTELLATION (IQ PLOT) --------
figure;
plot(real(sig), imag(sig), 'o', 'markersize', 2);
title([label " - Constellation"]);
xlabel("I");
ylabel("Q");
grid on;

