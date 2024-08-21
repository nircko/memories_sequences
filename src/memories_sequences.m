%% Parameters of the System Characteristics and Dynamics
N = 500;            % Number of neurons
q = 4;              % Number of asymmetric patterns
delta = 1;          % Difference between q and p
p = q + delta;      % Number of total patterns
duration = 100;     % Simulation time duration
lambda = 1.5;       % Scaling factor for asymmetric synapses
tau = 8;            % Time decay constant

%% Generating Random Patterns of the Neural Network
xi_s = 2 .* ceil(rand(N, delta) - 0.5) - 1;   % Symmetric patterns
xi_as = 2 .* ceil(rand(N, q) - 0.5) - 1;      % Asymmetric patterns
xi = [xi_as xi_s];                            % Combined pattern matrix

%% Synapses Construction
J = xi * xi';                 % Initial synaptic matrix
J1 = (J - diag(diag(J))) / N; % Symmetric synaptic matrix, normalized
J2 = circshift(xi_as, -1, 2) * xi_as'; % Asymmetric synaptic matrix
J2 = (J2 - diag(diag(J2))) * (lambda / N); % Asymmetric matrix, normalized

%% Initialization
S = zeros(N, duration);                        % Initialize state matrix
S(:, 1) = xi_as(:, 1);                         % Initial state
Sbar = 2 .* ceil(rand(N, 1) - 0.5) - 1;        % Averaged initial state
h1 = J1 * xi_as(:, 1);                         % Initial input from symmetric synapses
h2 = J2 * Sbar;                                % Initial input from asymmetric synapses
h = h1 + h2;                                   % Total initial input

%% Simulation Loop
for t = 1:duration
    S(:, t + 1) = sign(h);  % Update state based on total input
    w = tau^-1 * exp(-tau^-1 .* (t - (0:t)));  % Weight function
    Sbar = sum(w .* S(:, 1:t + 1), 2);  % Averaged state over time
    h1 = J1 * S(:, t + 1);   % Update input from symmetric synapses
    h2 = J2 * Sbar;          % Update input from asymmetric synapses
    h = h1 + h2;             % Update total input
end

%% Plotting Results
m = S(:, 1:duration)' * xi_as; % Overlap between state and patterns
plot(m, 'LineWidth', 1.2);
legend(strsplit(num2str(1:p)));
xlabel('Time');
xlim([0 duration]);
ylim([0 6 * N / 5]);
ylabel('m^{\nu}', 'FontSize', 24);
title('The Overlap of the \xi^{\nu} Pattern', 'FontName', 'Impact', 'FontSize', 18);

%% Loading and Processing Images
xi = zeros(50 * 50, 5);
figure_load = figure('Name', 'Counting Numbers', 'NumberTitle', 'off');
% create a path of files contating digits images
srcFiles = dir('./data/*jpg');

for i = 1:length(srcFiles)
    filename = fullfile(srcFiles(i).folder, srcFiles(i).name);
    I = im2double(im2bw(imread(filename)));
    subplot(2, 3, i);
    imagesc(I);
    colormap gray;
    drawnow;
    xi(:, i) = I(:);
end

%% Further Synapse Calculations and Simulation
xi_as = 2 .* xi - 1; % Binarize the matrix
N = size(xi, 1);
Q = (N)^-1 * (xi_as' * xi_as);
p = 8;
lambda = 2.5;
duration = 100;
tau = 8;
xi_s = 2 .* ceil(rand(N, p) - 0.5) - 1;
xi = [xi_as xi_s];
Y2 = inv((N)^-1 * (xi' * xi));
Y = inv(Q);
J1 = (xi * Y2 * xi' - diag(diag(xi * Y2 * xi'))) / N;
J2 = circshift(xi_as, -1, 2) * Y * xi_as';
J2 = (J2 - diag(diag(J2))) * (lambda / N);

S = zeros(N, duration); 
S(:, 1) = xi_as(:, 1); 
Sbar = 2 .* ceil(rand(N, 1) - 0.5) - 1; 
h1 = J1 * xi_as(:, 1); 
h2 = J2 * Sbar; 
h = h1 + h2;

for t = 1:duration
    S(:, t + 1) = sign(h);
    w = tau^-1 * exp(-tau^-1 .* (t - (0:t)));
    Sbar = sum(w .* S(:, 1:t + 1), 2);
    h1 = J1 * S(:, t + 1);
    h2 = J2 * Sbar;
    h = h1 + h2;
end

%% Image Processing and Simulation Visualization
Rimage = zeros(50, 50, duration + 1);
Sbw = floor(S / 2 + 1);

for t = 1:duration
    Rimage(:, :, t) = reshape(Sbw(:, t), 50, 50);
end

Preview = figure('Name', 'Live View', 'NumberTitle', 'off', ...
    'Position', [432 162 708 536]);

vidObj = VideoWriter('simulation.avi');
vidObj.Quality = 100;
vidObj.FrameRate = 2;
open(vidObj);

for t = 1:duration
    subplot(1, 2, 1);
    barh(Rimage(:, :, t)(:));
    title(['State of the System at t = ', num2str(t)]);
    subplot(1, 2, 2);
    imagesc(Rimage(:, :, t));
    colormap gray;
    drawnow;
    writeVideo(vidObj, getframe(gcf));
end

close(vidObj);
