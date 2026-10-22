dynare msd nolog;
for pt = [1 3 5 2 6]
  ms_sbvar_command_line(sprintf('-mdd -seed 0 -ft msd -pf simulation_msd.out -d 10000 -pt %d -l 0.1 -h 0.9 -print', pt));
end
ms_sbvar_command_line('-mdd -seed 0 -ft msd -pf simulation_msd.out -d 10000 -pt 4 -l 0.1 -h 0.9 -print');
