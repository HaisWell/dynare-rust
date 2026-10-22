// Stream X: macro-language features, checked against dynare-preprocessor's savemacro output
// (x_macro_macroexp.mod, from `dynare-preprocessor x_macro.mod savemacro onlymacro`).
@#define flag
@#define x = 5
@#define y = "US"
@#define v = [ 1, 2, 4 ]
@#define w = [ "US", "EA" ]
@#define z = 3 + v[2]
@#define t = ("US" in w)
@#define f(a) = " " + a + y
@#define g(a, b) = a*10 + b
@#define third = 1/3
@#define evens = [ i in 1:6 when mod(i, 2) == 0 ]
@#define squares = [ i^2 for i in 1:4 ]
@#define pairs = [ (j, i+1) for (i,j) in (1:2)^2 ]
@#define pairs2 = [ (j, i+1) for (i,j) in (1:2)*(1:2) when i < j ]
@#define u = [1, 2, 3] | [3, 4]
@#define n = [1, 2, 3] & [2, 3, 5]
@#define d = [1, 2, 3] - [2]
@#define s = "abcdef"
@#define c1 = (real) "2.5" + 1
@#define c2 = (string) (3 + 4)
@#define c3 = (array) 5 + (array) 6
var a b;
verbatim;
@#if flag && t
z=@{z} x=@{x} f=@{f("D")} g=@{g(2, 3)} third=@{third}
evens=@{evens} squares=@{squares} pairs=@{pairs} pairs2=@{pairs2}
u=@{u} n=@{n} d=@{d} sub=@{s[2:4]} c1=@{c1} c2=@{c2} c3=@{c3} len=@{length(v)} sum=@{sum(v)}
fns=@{floor(2.7)} @{sign(-3)} @{normcdf(0)} @{exp(1)} @{isreal(x)} @{isstring(y)} @{isempty([])}
@#endif
@#define countries = ["US", "FR", "JA"]
@#for co in countries when co != "US"
co_@{co}
@#endfor
@#for (i, j) in ["GDP"] * ["home", "foreign"]
@{i}_@{j}
@#endfor
end;
parameters p;
p = @{g(1, 1)};
model;
a = p*a(-1);
b = a;
end;
