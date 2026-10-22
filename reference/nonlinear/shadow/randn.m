function varargout = randn(varargin)
% Logging wrapper around the builtin randn (dump_nonlinear.m): while PF_LOG.on, every
% draw is appended (column-major) to PF_LOG.normals so dynare-rust can replay them.
global PF_LOG
[varargout{1:max(nargout,1)}] = builtin('randn', varargin{:});
if ~isempty(PF_LOG) && PF_LOG.on
    PF_LOG.normals{end+1} = varargout{1}(:);
end
