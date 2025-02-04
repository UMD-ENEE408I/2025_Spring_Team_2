%% Part A: Read and Downsample "human_voice.wav"
clear; close all; clc;

% --- Read the WAV file and display its original sampling frequency ---
[signal, Fs] = audioread('human_voice.wav');
fprintf('Original Sampling Frequency = %d Hz\n', Fs);

% --- Plot the Original Signal ---
t = (0:length(signal)-1) / Fs;  % Time vector for the original signal
figure('Name','Figure 1: Original Audio Signal','NumberTitle','off');
plot(t, signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Audio Signal');

% --- Downsample the Audio Signal to 8 kHz (without built-in downsampling) ---
targetFs = 8000;            % Target sampling frequency
D = Fs / targetFs;          % Downsampling factor
if rem(Fs, targetFs) ~= 0
    warning('Fs is not an integer multiple of 8000 Hz; using rounded factor for downsampling.');
    D = round(D);
end
downsampled_signal = signal(1:D:end, :);  % Take every Dth sample
newFs = Fs / D;                          % New sampling frequency (should be ~8 kHz)
fprintf('Number of samples after downsampling = %d\n', size(downsampled_signal, 1));

% --- Plot the Downsampled Signal ---
t_down = (0:length(downsampled_signal)-1) / newFs;
figure('Name','Figure 2: Downsampled Audio Signal (Approx. 8 kHz)','NumberTitle','off');
plot(t_down, downsampled_signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Downsampled Audio Signal (Approx. 8 kHz)');

% --- Compare a Section (0.5 s to 1.0 s) of the Original and Downsampled Signals ---
t_start = 0.5; 
t_end   = 1.0;
idx_orig = find(t >= t_start & t <= t_end);
idx_down = find(t_down >= t_start & t_down <= t_end);

figure('Name','Figure 3: Comparison of Original and Downsampled Signals','NumberTitle','off');
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

% --- Observations ---
% The downsampled signal has significantly fewer samples. This reduction in sample
% density may cause a loss of high-frequency detail, making the signal appear less smooth.

%% Part B: RMS, Cross-Correlation, and Sound Localization
% --- Read the Audio Files for the Three Microphones ---
[M1, Fs1] = audioread('M1.wav');
[M2, Fs2] = audioread('M2.wav');
[M3, Fs3] = audioread('M3.wav');

% Ensure all files have the same sampling rate
if ~(Fs1 == Fs2 && Fs2 == Fs3)
    error('Sampling frequencies of the files do not match.');
end
Fs = Fs1;  % Use the common sampling rate

% --- (1) Calculate RMS Values for Each Audio Signal ---
RMS_M1 = sqrt(mean(M1.^2));
RMS_M2 = sqrt(mean(M2.^2));
RMS_M3 = sqrt(mean(M3.^2));
fprintf('RMS Values:\n');
fprintf('  M1: %f\n  M2: %f\n  M3: %f\n', RMS_M1, RMS_M2, RMS_M3);

% --- (2) Determine Which Microphone is Closer to the Sound Source ---
if RMS_M1 > RMS_M2
    fprintf('Microphone M1 is closer to the sound source than M2.\n');
else
    fprintf('Microphone M2 is closer to the sound source than M1.\n');
end

% --- (3) Compute Time Delay Between M1 and M2 via Cross-Correlation ---
% Cross-correlation: Rxy[m] = sum_n x[n]*y[n-m]
L_corr = length(M1) + length(M2) - 1;
cross_corr = zeros(L_corr, 1);
for m = 1:L_corr
    shift = m - length(M2);
    sum_val = 0;
    for n = 1:length(M1)
        if (n + shift > 0) && (n + shift <= length(M2))
            sum_val = sum_val + M1(n) * M2(n + shift);
        end
    end
    cross_corr(m) = sum_val;
end
[~, max_index] = max(abs(cross_corr));
time_delay = (max_index - length(M2)) / Fs;
fprintf('Estimated time delay between M1 and M2: %f seconds\n', time_delay);

% --- (4) Compute the Angle for Robot Orientation Correction ---
r = 0.10;   % Distance between microphones in meters (10 cm)
c = 343;    % Speed of sound in air (m/s)
d_diff = time_delay * c;  % Difference in distance traveled by the sound
theta = asind(d_diff / (2 * r));  % Angle in degrees
fprintf('The robot must turn by θ = %f degrees to correct its heading.\n', theta);

%% Part C: Low-Pass FIR Filtering of "Cafe_with_noise.wav"
% --- Load and Plot the Original Audio Signal ---
[y, Fs] = audioread('Cafe_with_noise.wav');
t = (0:length(y)-1) / Fs;
figure('Name','Figure 4: Original Cafe_with_noise.wav Signal (Time Domain)','NumberTitle','off');
plot(t, y);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Audio Signal (Time Domain)');
grid on;

% --- Frequency Domain Analysis of the Original Signal ---
L = length(y);
Y = fft(y);
f = Fs * (0:floor(L/2)) / L;
Y_mag = abs(Y(1:floor(L/2)+1));
figure('Name','Figure 5: Magnitude Spectrum of Original Audio Signal','NumberTitle','off');
plot(f, Y_mag);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Magnitude Spectrum of Original Audio Signal');
grid on;

% --- Design a Low-Pass FIR Filter Using the Window Method ---
% Specifications:
cutoff = 1200;      % Cutoff frequency in Hz
N = 1000;           % Filter order (results in N+1 coefficients)
n = 0:N;            % Sample indices
center = N / 2;     % Center index for symmetry
omega_c = 2 * pi * (cutoff / Fs);  % Normalized cutoff frequency (radians/sample)

% Compute the ideal impulse response (sinc function)
hd = zeros(1, N+1);
for i = 1:length(n)
    if n(i) == center
        hd(i) = omega_c / pi;
    else
        hd(i) = sin(omega_c * (n(i) - center)) / (pi * (n(i) - center));
    end
end

% Compute the Hamming window manually
w = 0.54 - 0.46 * cos(2 * pi * n / N);

% Obtain the FIR filter coefficients and normalize
h = hd .* w;
h = h / sum(h);

% --- Apply the FIR Filter (via convolution) ---
y_filtered = conv(y, h, 'same');

% --- Plot the Filtered Signal (Time Domain) ---
figure('Name','Figure 6: Filtered Audio Signal (Time Domain)','NumberTitle','off');
plot(t, y_filtered);
xlabel('Time (s)');
ylabel('Amplitude');
title('Filtered Audio Signal (Time Domain)');
grid on;

% --- Frequency Domain Analysis of the Filtered Signal ---
Y_filt = fft(y_filtered);
Y_filt_mag = abs(Y_filt(1:floor(L/2)+1));
figure('Name','Figure 7: Magnitude Spectrum of Filtered Audio Signal','NumberTitle','off');
plot(f, Y_filt_mag);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Magnitude Spectrum of Filtered Audio Signal');
grid on;

