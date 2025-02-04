clear
clf
%%% 3.1 Part A
%% Read the wav file “human_voice.wav” and write down its original sampling frequency.
% Read the file
[signal, Fs] = audioread('human_voice.wav');
fprintf('Original Sampling Frequency = %d Hz\n', Fs);

%% Plot the original signal
% Create a time vector for the original signal
t = (0:length(signal)-1) / Fs;

figure;
plot(t, signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Audio Signal');

%% Downsample the audio file to 8 kHz without using inbuilt downsampling functions
% Define the target sampling frequency
targetFs = 8000;

% Determine the downsampling factor.
% Here I assume that the original Fs is an integer multiple of 8000.
D = Fs / targetFs;
if rem(Fs, targetFs) ~= 0
    % If not an integer multiple, warn and use the nearest integer factor.
    warning('Fs is not an integer multiple of 8000 Hz; using rounded factor for downsampling.');
    D = round(D);
end

% Downsample by taking every Dth sample.
downsampled_signal = signal(1:D:end, :);

% The new sampling frequency is:
newFs = Fs / D;  % Should be 8000 Hz if D is exact

%% How many audio samples were obtained after downsampling?
numSamplesDown = size(downsampled_signal, 1);
fprintf('Number of samples after downsampling = %d\n', numSamplesDown);

%% Plot the downsampled signal
% Create a time vector for the downsampled signal
t_down = (0:numSamplesDown-1) / newFs;

figure;
plot(t_down, downsampled_signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Downsampled Audio Signal (Approx. 8 kHz)');

%%  Compare a section of the audio signal corresponding to the same time period
% Let’s choose a segment between 0.5 sec and 1.0 sec as an example.
t_start = 0.5;
t_end   = 1.0;

% Find indices corresponding to that segment for both signals
idx_orig = find(t >= t_start & t <= t_end);
idx_down = find(t_down >= t_start & t_down <= t_end);

figure(1);
subplot(2,1,1);
plot(t(idx_orig), signal(idx_orig));
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal (0.5 s to 1.0 s)');

subplot(2,1,2);
plot(t_down(idx_down), downsampled_signal(idx_down));
xlabel('Time (s)');
ylabel('Amplitude');
title('Downsampled Signal (0.5 s to 1.0 s)');

%% Observations:
% When comparing the same time segment from the original and downsampled signals,
% I notice that the downsampled version has significantly fewer samples.
% This reduction in sample density can lead to a loss in detail, especially for high-frequency
% components. The original signal appears smoother and captures more of the signal’s fine details,
% while the downsampled signal may show a more “stepped” or less smooth behavior, which can be
% attributed to the lower effective temporal resolution.


%% ENEE408I: Lab 3 - Part B: RMS, Cross-correlation, Sound Localization

% Read the audio files
[M1, Fs1] = audioread('M1.wav');
[M2, Fs2] = audioread('M2.wav');
[M3, Fs3] = audioread('M3.wav');

% Ensure all files have the same sampling rate
if ~(Fs1 == Fs2 && Fs2 == Fs3)
    error('Sampling frequencies of the files do not match.');
end

Fs = Fs1; % Set common sampling rate (8 kHz given in lab description)

%% (1) Calculate RMS values for each audio signal
RMS_M1 = sqrt(mean(M1.^2));
RMS_M2 = sqrt(mean(M2.^2));
RMS_M3 = sqrt(mean(M3.^2));

% Display the RMS values
fprintf('RMS Values:\n');
fprintf('M1: %f\n', RMS_M1);
fprintf('M2: %f\n', RMS_M2);
fprintf('M3: %f\n', RMS_M3);

%% (2) Determine which microphone is closer to the sound source
if RMS_M1 > RMS_M2
    fprintf('Microphone M1 is closer to the sound source than M2.\n');
else
    fprintf('Microphone M2 is closer to the sound source than M1.\n');
end

%% (3) Compute time delay using Cross-Correlation (Without built-in functions)
% Cross-correlation formula: Rxy[m] = Σ x[n] * y[n - m]

% Compute cross-correlation manually
L = length(M1) + length(M2) - 1;
cross_corr = zeros(L, 1);

for m = 1:L
    shift = m - length(M2);
    sum_val = 0;
    for n = 1:length(M1)
        if (n + shift > 0) && (n + shift <= length(M2))
            sum_val = sum_val + M1(n) * M2(n + shift);
        end
    end
    cross_corr(m) = sum_val;
end

% Find the lag corresponding to the maximum correlation
[~, max_index] = max(abs(cross_corr));

% Compute time delay in seconds
time_delay = (max_index - length(M2)) / Fs;

fprintf('Estimated time delay between M1 and M2: %f seconds\n', time_delay);

%% (4) Compute the angle θ for robot orientation correction
% Given r = 10 cm (distance between the microphones)
r = 0.10;  % in meters
c = 343;    % Speed of sound in air (m/s)
    
% Compute distance difference using time delay
d_diff = time_delay * c;

% Compute the angle θ (in degrees) using arcsin
theta = asind(d_diff / (2 * r));

fprintf('The robot must turn by θ = %f degrees to correct its heading.\n', theta);
