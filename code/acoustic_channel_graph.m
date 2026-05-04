% Path to the audio file
filePath = 'C:\Users\dogan\Desktop\lazer_ve_ses_sinyalleri\veriseti\Data\Ses\Temiz\Duvar\S2\C1\S_D_0_C1\S_D_0_C1_A_B_C.wav';

% Read the audio file
[signal, fs] = audioread(filePath);

% Get number of channels
numChannels = size(signal, 2);

% Time vector
t = (0:length(signal)-1) / fs;

% Plot each channel in a separate subplot
figure;
for k = 1:numChannels
    subplot(numChannels, 1, k);
    plot(t, signal(:, k));
    xlabel('Time (s)');
    ylabel(sprintf('Channel %d', k));
    title(sprintf('Audio Signal - Channel %d', k));
    grid on;
end
