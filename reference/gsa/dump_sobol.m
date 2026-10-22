% dump_sobol.m -- direction numbers of Dynare's qmc_sequence (Sobol, 1111 dimensions),
% extracted from the sequence itself: with Gray-code ordering, the point at seed
% 2^(j+1)-1 is v_j alone, so m(d,j+1) = x(d) * 2^(j+1) is the j-th direction integer.
% Writes sobol_directions.txt (1111 lines x NBITS integers) and sobol_check.json.
% Run from a private copy: /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_sobol.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
D = 1111; NBITS = 24; chunk = 2^14;
want = 2.^(1:NBITS) - 1;            % seeds holding the single direction numbers
m = zeros(D, NBITS);
extra_seed = 1000001; extra = [];
seed = int64(1); pos = 1;            % pos = seed of the first row of the chunk
total = 2^NBITS;
tic
while pos <= total
    n = min(chunk, total - pos + 1);
    [a, seed] = qmc_sequence(D, seed, 0, n);
    rows = pos:pos+n-1;
    for j = 1:NBITS
        k = find(rows == want(j));
        if ~isempty(k)
            m(:, j) = a(:, k) * 2^j;
        end
    end
    k = find(rows == extra_seed);
    if ~isempty(k), extra = a(:, k); end
    pos = pos + n;
end
toc
fid = fopen('sobol_directions.txt', 'w');
fprintf(fid, '%% Dynare qmc_sequence direction integers m(d,j), d=1..%d, j=1..%d (v_j = m/2^j)\n', D, NBITS);
for d = 1:D
    fprintf(fid, '%d ', m(d, 1:end-1)); fprintf(fid, '%d\n', m(d, end));
end
fclose(fid);
out = struct();
out.first = qmc_sequence(40, int64(1), 0, 1024)';          % 1024 x 40
b = qmc_sequence(D, int64(1), 0, 64)'; out.high = b(:, 1090:D); % 64 x 22
out.extra_seed = extra_seed; out.extra = extra;              % seed 1000001, all dims
% scrambled-free sample as stability_mapping draws it: qmc_sequence(np, int64(1), 0, Nsam)'
out.sample_np7_n2048 = qmc_sequence(7, int64(1), 0, 2048)';
fid = fopen('sobol_check.json', 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
