// Stream X: macro-language features, checked against dynare-preprocessor's savemacro output
// (x_macro_macroexp.mod, from `dynare-preprocessor x_macro.mod savemacro onlymacro`).
var a b;
verbatim;
z=5 x=5 f= DUS g=23 third=0.333333333333333
evens=[2, 4, 6] squares=[1, 4, 9, 16] pairs=[(1, 2), (2, 2), (1, 3), (2, 3)] pairs2=[(2, 2)]
u=[1, 2, 3, 4] n=[2, 3] d=[1, 3] sub=bcd c1=3.5 c2=7 c3=[5, 6] len=3 sum=7
fns=2 -1 0.5 2.71828182845905 true true true
co_FR
co_JA
GDP_home
GDP_foreign
end;
parameters p;
p = 11;
model;
a = p*a(-1);
b = a;
end;
