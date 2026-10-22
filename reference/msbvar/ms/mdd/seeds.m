% Dynare's Muller / bridge on the same 5000 simulation draws, 20 seeds of the 10000 proposal draws, types 1-6
fid = fopen('seeds.txt', 'w');
for pt = 1:6
  for seed = 1:20
    ms_sbvar_command_line(sprintf('-mdd -seed %d -ft msd -pf simulation_msd.out -d 10000 -pt %d -l 0.1 -h 0.9', seed, pt));
    txt = fileread(sprintf('mdd_t%d_msd.out', pt));
    m = regexp(txt, 'Muller log\(mdd\) = (\S+)', 'tokens');
    b = regexp(txt, 'Bridge log\(mdd\) = (\S+)', 'tokens');
    fprintf(fid, '%d %d %s %s\n', pt, seed, m{1}{1}, b{1}{1});
  end
end
fclose(fid);
