droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
base = getenv('E72_WORK'); if isempty(base), base = fullfile(tempdir, 'e72', 'm'); end
addpath(base);
cd(fullfile(base,'schorfheide_2000'));
dynare schorfheide_2000 nograph nointeractive
r = est_dump(M_, oo_, options_, bayestopt_);
fid=fopen(fullfile(base,'schorfheide_2000.json'),'w'); fprintf(fid,'%s',jsonencode(r,'ConvertInfAndNaN',true)); fclose(fid);
