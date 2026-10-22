function varargout = rand(varargin)
% Logging wrapper around the builtin rand (dump_nonlinear.m): while PF_LOG.on, every
% draw is appended (column-major) to PF_LOG.uniforms so dynare-rust can replay them.
global PF_LOG
[varargout{1:max(nargout,1)}] = builtin('rand', varargin{:});
if ~isempty(PF_LOG) && PF_LOG.on
    PF_LOG.uniforms{end+1} = varargout{1}(:);
end
