function s = fullify(s)
if issparse(s), s = full(s);
elseif isstruct(s)
  for k = 1:numel(s), f = fieldnames(s);
    for j=1:numel(f), s(k).(f{j}) = fullify(s(k).(f{j})); end
  end
elseif iscell(s)
  for k=1:numel(s), s{k} = fullify(s{k}); end
elseif isa(s,'function_handle'), s = func2str(s);
end
end
