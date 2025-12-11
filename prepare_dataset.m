clear all; close all; clc;

root = "dataset";

mods = {"BPSK","QPSK","QAM16","AM","FM","FSK"};
X = [];
Y = [];

fprintf("Loading all signals...\n");

for m = 1:length(mods)
    modName = mods{m};
    folder = fullfile(root, modName);

    files = dir(fullfile(folder, "*.mat"));
    if isempty(files)
        error("No .mat files found in %s!", folder);
    end

    for k = 1:length(files)
        data = load(fullfile(folder, files(k).name));

        % Detect variable name inside .mat
        varName = fieldnames(data);
        sig = data.(varName{1});

        % --- FORCE CONSISTENT SHAPE ---
        sig = sig(:).';      % convert to row (1×N)

        % --- FIX LENGTH ---
        if length(sig) < 2048
            % pad with zeros
            sig = [sig, zeros(1,2048 - length(sig))];
        elseif length(sig) > 2048
            % trim
            sig = sig(1:2048);
        end

        % Add to dataset
        X = [X; sig];
        Y = [Y; m];

    end
end

fprintf("Total samples loaded: %d\n", size(X,1));

save("X.mat","X");
save("Y.mat","Y");

fprintf("Saved X.mat and Y.mat successfully.\n");

