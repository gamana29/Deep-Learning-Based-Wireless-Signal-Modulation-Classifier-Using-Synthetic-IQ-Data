clear all;
close all;
clc;

% dataset main folder
baseDir = "dataset";

% STEP 1: settings
N = 2048;            % samples per signal
num_files = 100;     % how many samples per modulation
mods = {'BPSK','QPSK','QAM16','AM','FM','FSK'};

for m = 1:length(mods)
    modName = mods{m};

    for k = 1:num_files
        bits = randi([0 1], N, 1);

        switch modName

            % ----------------- BPSK -----------------
            case 'BPSK'
                iq = 2*bits - 1;

                clean = iq;
                awgn20 = add_awgn(iq, 20);
                rayl = rayleigh_fading(iq);
                rician = rician_fading(iq, 3);

                out = fullfile(baseDir, "BPSK");
                if ~exist(out, "dir"), mkdir(out); end

                save(fullfile(out, sprintf("BPSK_clean_%d.mat",k)), "clean");
                save(fullfile(out, sprintf("BPSK_awgn_%d.mat",k)), "awgn20");
                save(fullfile(out, sprintf("BPSK_rayleigh_%d.mat",k)), "rayl");
                save(fullfile(out, sprintf("BPSK_rician_%d.mat",k)), "rician");

            % ----------------- QPSK -----------------
            case 'QPSK'
                b = reshape(bits(1:2*floor(N/2)),[],2);
                iq = (2*b(:,1)-1 + 1i*(2*b(:,2)-1))/sqrt(2);

                clean = iq;
                awgn20 = add_awgn(iq, 20);
                rayl = rayleigh_fading(iq);
                rician = rician_fading(iq, 3);

                out = fullfile(baseDir, "QPSK");
                if ~exist(out, "dir"), mkdir(out); end

                save(fullfile(out, sprintf("QPSK_clean_%d.mat",k)), "clean");
                save(fullfile(out, sprintf("QPSK_awgn_%d.mat",k)), "awgn20");
                save(fullfile(out, sprintf("QPSK_rayleigh_%d.mat",k)), "rayl");
                save(fullfile(out, sprintf("QPSK_rician_%d.mat",k)), "rician");

            % ----------------- QAM16 -----------------
            case 'QAM16'
                M = 16;
                data = randi([0 M-1], N, 1);
                iq = qammod(data, M);
                iq = iq / sqrt(mean(abs(iq).^2));

                clean = iq;
                awgn20 = add_awgn(iq, 20);
                rayl = rayleigh_fading(iq);
                rician = rician_fading(iq, 3);

                out = fullfile(baseDir, "QAM16");
                if ~exist(out, "dir"), mkdir(out); end

                save(fullfile(out, sprintf("QAM16_clean_%d.mat",k)), "clean");
                save(fullfile(out, sprintf("QAM16_awgn_%d.mat",k)), "awgn20");
                save(fullfile(out, sprintf("QAM16_rayleigh_%d.mat",k)), "rayl");
                save(fullfile(out, sprintf("QAM16_rician_%d.mat",k)), "rician");

            % ----------------- AM -----------------
            case 'AM'
                t = linspace(0,1,N).';
                msg = double(bits);
                carrier = cos(2*pi*200*t);
                iq = (1 + 0.6*msg).*carrier;

                out = fullfile(baseDir, "AM");
                if ~exist(out, "dir"), mkdir(out); end

                save(fullfile(out, sprintf("AM_%d.mat",k)), "iq");

            % ----------------- FM -----------------
            case 'FM'
                t = linspace(0,1,N).';
                msg = double(bits)-0.5;
                kf = 50;
                phase = 2*pi*200*t + 2*pi*kf*cumsum(msg)/N;
                iq = cos(phase);

                out = fullfile(baseDir, "FM");
                if ~exist(out, "dir"), mkdir(out); end

                save(fullfile(out, sprintf("FM_%d.mat",k)), "iq");

            % ----------------- FSK -----------------
            case 'FSK'
                t = linspace(0,1,N).';
                b = double(bits);
                f0 = 100;
                f1 = 300;
                f = f0*(b==0) + f1*(b==1);
                iq = cos(2*pi.*f.*t);

                out = fullfile(baseDir, "FSK");
                if ~exist(out, "dir"), mkdir(out); end

                save(fullfile(out, sprintf("FSK_%d.mat",k)), "iq");

        end

    end
end


% -------------------- CHANNEL FUNCTIONS --------------------

function y = add_awgn(x, snr_dB)
    noise = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2);
    y = x + noise * 10^(-snr_dB/20);
end

function y = rayleigh_fading(x)
    h = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2);
    y = h .* x;
end

function y = rician_fading(x, K)
    h_LOS = sqrt(K/(K+1));
    h_NLOS = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2*(K+1));
    h = h_LOS + h_NLOS;
    y = h .* x;
end

