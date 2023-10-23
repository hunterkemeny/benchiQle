OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(-pi) q[0];
sx q[0];
rz(-0.64543986) q[1];
sx q[1];
rz(-2.9644047) q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
sx q[1];
rz(3.53207685057836) q[1];
sx q[1];
rz(7*pi/2) q[1];
sx q[2];
rz(2.7511085) q[2];
sx q[2];
rz(-pi) q[2];
cx q[1],q[2];
rz(-pi) q[1];
sx q[1];
rz(-2.4272417) q[1];
cx q[0],q[1];
rz(3*pi/2) q[0];
sx q[0];
rz(4.17279559527435) q[0];
sx q[0];
rz(7*pi/2) q[0];
sx q[1];
rz(2.1103897) q[1];
sx q[1];
rz(-pi) q[1];
cx q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.073482312) q[0];
sx q[0];
rz(1.1389922) q[1];
sx q[1];
rz(0.093717342) q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
sx q[1];
rz(3.87252240097961) q[1];
sx q[1];
rz(7*pi/2) q[1];
sx q[2];
rz(2.4106629) q[2];
sx q[2];
rz(-pi) q[2];
cx q[1],q[2];
rz(-pi) q[1];
sx q[1];
rz(-0.3949138) q[1];
cx q[0],q[1];
rz(3*pi/2) q[0];
sx q[0];
rz(4.17279559527435) q[0];
sx q[0];
rz(7*pi/2) q[0];
sx q[1];
rz(2.1103897) q[1];
sx q[1];
rz(-pi) q[1];
cx q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(2.6996279) q[0];
rz(-2.5706465) q[1];
sx q[1];
rz(-pi) q[1];
x q[2];
rz(-1.8158889) q[2];
cx q[1],q[2];
rz(3*pi/2) q[1];
sx q[1];
rz(3.53207685057836) q[1];
sx q[1];
rz(7*pi/2) q[1];
sx q[2];
rz(2.7511085) q[2];
sx q[2];
rz(-pi) q[2];
cx q[1],q[2];
sx q[1];
rz(-1.081822) q[1];
x q[2];
rz(-1.9044498) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];