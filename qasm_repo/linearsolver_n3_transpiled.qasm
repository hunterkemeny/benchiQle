OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
cx q[2],q[1];
rz(-pi) q[2];
sx q[2];
rz(2.5615927) q[2];
sx q[2];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[2];
rz(0.99079633) q[2];
sx q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
