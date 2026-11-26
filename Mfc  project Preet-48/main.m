%% Image Compression using Singular Value Decomposition (SVD)
clc; clear; close all;

% 1. Load an image using imread() and convert it to grayscale
img = imread('cameraman.tif'); 
if size(img, 3) == 3
    img = rgb2gray(img);
end

% 2. Represent the image as a double-precision matrix
A = double(img);
[m, n] = size(A);
original_elements = m * n;
original_norm = norm(A, 'fro');

% 3. Apply SVD using MATLAB's svd() function
[U, S, V] = svd(A);
singular_values = diag(S);

% Define different numbers of singular values (k) to test
k_values = [5, 20, 50, 100];
num_k = length(k_values);

% Preallocate arrays for metrics
rel_error = zeros(1, num_k);
compression_ratio = zeros(1, num_k);
storage_saving = zeros(1, num_k);

% 6. Display the original and compressed images using imshow()
figure('Name', 'SVD Image Compression', 'Position', [100, 100, 1200, 400]);
subplot(1, num_k + 1, 1);
imshow(uint8(A));
title(sprintf('Original\n(%dx%d)', m, n));

fprintf('%-8s | %-18s | %-18s | %-15s\n', 'Rank (k)', 'Stored Elements', 'Storage Saving (%)', 'Rel. Error (%)');
fprintf('-------------------------------------------------------------------\n');

for i = 1:num_k
    k = k_values(i);

    % 4. Reconstruct the image using k singular values
    A_k = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';

    % 5. Calculate reconstruction error (Frobenius norm) & storage requirements
    % Storage needed for rank-k: U_k (m*k) + S_k (k) + V_k (n*k) = k*(m + n + 1)
    compressed_elements = k * (m + n + 1);
    compression_ratio(i) = original_elements / compressed_elements;
    storage_saving(i) = (1 - (compressed_elements / original_elements)) * 100;
    rel_error(i) = (norm(A - A_k, 'fro') / original_norm) * 100;

    % Print metrics to Command Window
    fprintf('%-8d | %-18d | %-17.2f%% | %-14.2f%%\n', ...
        k, compressed_elements, storage_saving(i), rel_error(i));

    % Plot compressed image
    subplot(1, num_k + 1, i + 1);
    imshow(uint8(A_k));
    title(sprintf('k = %d\nError: %.1f%%', k, rel_error(i)));
end

% Plot Singular Value Decay & Reconstruction Error Graphs
figure('Name', 'SVD Error & Compression Analysis', 'Position', [150, 150, 1000, 400]);

subplot(1, 2, 1);
semilogy(singular_values, 'LineWidth', 1.5);
grid on;
xlabel('Singular Value Index (i)');
ylabel('Singular Value \sigma_i (log scale)');
title('Singular Value Decay');

subplot(1, 2, 2);
yyaxis left
plot(k_values, rel_error, '-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
ylabel('Relative Reconstruction Error (%)');

yyaxis right
plot(k_values, storage_saving, '-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
ylabel('Storage Space Saved (%)');

grid on;
xlabel('Number of Singular Values (k)');
title('Reconstruction Error vs. Storage Savings');
legend('Reconstruction Error', 'Storage Saved', 'Location', 'best');