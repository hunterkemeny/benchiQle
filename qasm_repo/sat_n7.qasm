//SAT: Should obtain 11 in very high probability, see the paper "Automatically Solving NP-Complete Problems on a Quantum Computer"
OPENQASM 2.0;
include "qelib1.inc";

qreg var[3];
qreg conj[3];
qreg anci[1];
creg ans[2];

h var[1];
h var[2];
x conj[0];
x conj[1];
x conj[2];
x var[1];
x var[2];
ccx var[1], var[2], conj[0];
x var[2];
ccx var[1], var[2], conj[1];
x var[1];
x var[2];
ccx var[1], var[2], conj[2];
x var[2];
ccx conj[0], conj[1], anci[0];
ccx conj[2], anci[0], var[0];
ccx conj[0], conj[1], anci[0];
x var[2];
ccx var[1], var[2], conj[2];
x var[1];
x var[2];
ccx var[1], var[2], conj[1];
x var[2];
ccx var[1], var[2], conj[0];
x var[1];
x var[2];
h var[1];
h var[2];
x var[0];
x var[1];
x var[2];
h var[0];
ccx var[1], var[2], var[0];
h var[0];
x var[0];
x var[1];
x var[2];
h var[0];
h var[1];
h var[2];

measure var[1] -> ans[0];
measure var[2] -> ans[1];
