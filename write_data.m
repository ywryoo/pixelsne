function write_data(X, no_dims, initial_dims, perplexity, theta, alg, p_method, bins)

    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('initial_dims', 'var') || isempty(initial_dims)
        initial_dims = 50;
    end
    if ~exist('perplexity', 'var') || isempty(perplexity)
        perplexity = 30;
    end
    if ~exist('theta', 'var') || isempty(theta)
        theta = 0.5;
    end
    if ~exist('alg', 'var') || isempty(alg)
        alg = 'svd';
    end
    if ~exist('p_method', 'var') || isempty(p_method)
        p_method = 1;   % 1: vp-tree, else: construct-knn
    end
    if ~exist('bins', 'var') || isempty(bins)
        bins = 512;
    end
    
    % Perform the initial dimensionality reduction using PCA
    X = double(X);
    X = bsxfun(@minus, X, mean(X, 1));
    M = pca(X,'NumComponents',initial_dims,'Algorithm',alg);
    X = X * M;
    % Run the fast diffusion SNE implementation
    write_data_file(X, no_dims, theta, perplexity, bins, p_method);
end

function write_data_file(X, no_dims, theta, perplexity, bins, p_method)
    [n, d] = size(X);
    h = fopen('data.dat', 'wb');
	fwrite(h, n, 'integer*4');
	fwrite(h, d, 'integer*4');
    fwrite(h, theta, 'double');
    fwrite(h, perplexity, 'double');
	fwrite(h, no_dims, 'integer*4');
    fwrite(h, p_method, 'integer*4');
    fwrite(h, bins, 'integer*4');
    fwrite(h, X', 'double');
	fclose(h);
end