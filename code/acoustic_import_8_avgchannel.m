% Path to the audio file
filePath = 'C:\Users\dogan\Desktop\lazer_ve_ses_sinyalleri\veriseti\Data\Ses\Temiz\Duvar\S2\C1\S_D_0_C1\S_D_0_C1_A_B_C.wav';

% Read the audio file
[signal8ch, fs8ch] = audioread(filePath);

% Average across channels -> make it single-channel
signalSingle = mean(signal8ch, 2);

% Time vector
t = (0:length(signalSingle)-1) / fs8ch;

% Plot the single-channel signal
figure
plot(t, signalSingle);
xlabel('Time (s)');
ylabel('Amplitude');
title('Average of 8 Channels - Single-Channel Audio Signal');
grid on;
