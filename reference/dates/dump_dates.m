% dump_dates.m -- Dynare 7.2 (dseries) reference values for weekly and daily dates
% parsing, char/year/subperiod, shifts, differences, ranges,
% the dates('W', y, w) constructor, sameperiodpreviousyear and weekly/daily ydiff/ygrowth.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_dates.m')"
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
here = fileparts(mfilename('fullpath'));
cd(here);
out = struct();

strs = {'0W1','0W52','1W1','4W53','1950W1','1950W52','1950W53','1951W1','2003W53','2004W53','2004W52', ...
        '2005W1','2009W53','2015W53','2016W1','2020W52','2020W53','2021W1','2021W52','2026W53','2400W1', ...
        '1999W53','2000W52','-1950W2','-1W1','2020W54','2020W0','2020w01','2020w5', ...
        '2020-01-01','2020-02-29','2021-02-29','2019-12-31','1900-02-29','2000-02-29','0000-01-01', ...
        '-0001-12-31','1970-01-01','2020-1-01','2020-13-01', '2004S2', '2004H1'};
out.parse = {};
for i = 1:numel(strs)
    s = strs{i};
    r = struct('s', s, 'ok', false, 'freq', NaN, 'time', NaN, 'label', '', 'year', NaN, 'subperiod', NaN);
    try
        if isdate(s)
            d = dates(s);
            r.ok = true; r.freq = d.freq; r.time = d.time; r.label = char(d);
            r.year = year(d);
            try, r.subperiod = subperiod(d); catch, r.subperiod = NaN; end
        end
    catch
        r.ok = false;
    end
    out.parse{end+1} = r;
end

shifts = {{'2004W52',1},{'2004W52',2},{'2004W53',1},{'2005W1',-1},{'2005W1',-53},{'2020W1',53},{'2020W1',52}, ...
          {'1950W1',-52},{'1950W1',1000},{'2020W53',-53},{'0W2',-1},{'2020-02-28',1},{'2020-02-28',2}, ...
          {'2019-12-31',1},{'2020-03-01',-1},{'2021-03-01',-1},{'1999-12-31',366},{'2000-01-01',-365}};
out.shifts = {};
for i = 1:numel(shifts)
    s = shifts{i};
    d = dates(s{1}) + s{2};
    out.shifts{end+1} = struct('s', s{1}, 'k', s{2}, 'time', d.time, 'label', char(d), 'year', year(d));
end

pairs = {{'2004W1','2005W1'},{'2020W1','2021W1'},{'2019W1','2021W52'},{'1950W5','2050W5'},{'0W1','2026W43'}, ...
         {'2020-01-01','2021-01-01'},{'2019-01-01','2020-01-01'},{'1900-01-01','2000-01-01'}};
out.diffs = {};
for i = 1:numel(pairs)
    p = pairs{i};
    out.diffs{end+1} = struct('a', p{1}, 'b', p{2}, 'n', dates(p{2}) - dates(p{1}));
end

ranges = {{'2020W50','2021W3'},{'2004W51','2005W2'},{'2015W52','2016W2'},{'1950W1','1951W2'},{'2019-12-30','2020-01-02'}, ...
          {'2020-02-27','2020-03-01'}};
out.ranges = {};
for i = 1:numel(ranges)
    p = ranges{i};
    d = dates(p{1}):dates(p{2});
    r = struct('a', p{1}, 'b', p{2});
    r.labels = strings(d);
    r.years = year(d);
    try, r.subperiods = subperiod(d); catch, r.subperiods = []; end
    out.ranges{end+1} = r;
end

ctor = {{2020,53},{2021,53},{0,1},{-1,1},{1950,52},{2004,53},{2026,1}};
out.ctor = {};
for i = 1:numel(ctor)
    c = ctor{i};
    r = struct('y', c{1}, 'w', c{2}, 'ok', false, 'time', NaN, 'label', '');
    try
        d = dates('W', c{1}, c{2});
        r.ok = true; r.time = d.time; r.label = char(d);
    catch
    end
    out.ctor{end+1} = r;
end

spy = {{'2019W1','2021W52'},{'2015W40','2016W10'},{'2019-01-01','2021-12-31'},{'2020Q1','2021Q4'}};
out.spy = {};
for i = 1:numel(spy)
    p = spy{i};
    d = dates(p{1}):dates(p{2});
    id = sameperiodpreviousyear(d);
    out.spy{end+1} = struct('a', p{1}, 'b', p{2}, 'id', id);
end

d = dates('2019W1'):dates('2021W52');
ts = dseries(transpose(1.01.^(1:d.ndat())), '2019W1', {'A1'});
out.weekly_ydiff = ts.ydiff.data;
out.weekly_ygrowth = ts.ygrowth.data;
d = dates('2019-01-01'):dates('2021-12-31');
ts = dseries(transpose(1.0001.^(1:d.ndat())), '2019-01-01', {'A1'});
out.daily_ydiff = ts.ydiff.data;
out.daily_ygrowth = ts.ygrowth.data;

fid = fopen('ref_dates.json', 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
