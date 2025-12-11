clear all;
close all;
clc;

mods = {'BPSK','QPSK','QAM16','AM','FM','FSK'};

for i = 1:length(mods)

    modName = mods{i};

    % Load first .mat file
    files = dir(fullfile(modName, '*.mat'));
    sample_file = fullfile(modName, files(1).name);

    data = load(sample_file);
    iq = data.(cell2mat(fieldnames(data)));

    % ---- 1. Waveform Plot ----
    figure;
    plot(real(iq));
    title([modName ' – Time-domain Signal']);
    xlabel('Sample Index');
    ylabel('Real(IQ)');
    grid on;

    % ---- 2. Constellation Plot (if applicable) ----
    if iscomplex(iq) || max(abs(iq)) < 2  % BPSK/QPSK/QAM16
        figure;
        scatter(real(iq), imag(iq), '.');
        title([modName ' – Constellation Diagram']);
        xlabel('In-phase (I)');
        ylabel('Quadrature (Q)');
        grid on;
        axis equal;
    else
        disp([modName ' is a real-only signal (AM/FM/FSK), no constellation.'])
    end

end

disp("All waveform + constellation figures displayed.");

