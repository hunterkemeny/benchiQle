OPENQASM 2.0;
include "qelib1.inc";
qreg q0[67];
creg c0[1];
rz(pi/2) q0[0];
sx q0[0];
rz(pi/2) q0[0];
rz(-pi) q0[1];
sx q0[1];
rz(2.1199875) q0[1];
sx q0[1];
rz(-pi) q0[2];
sx q0[2];
rz(2.9643996) q0[2];
sx q0[2];
rz(-pi) q0[3];
sx q0[3];
rz(2.8957076) q0[3];
sx q0[3];
rz(-pi) q0[4];
sx q0[4];
rz(2.7678877) q0[4];
sx q0[4];
rz(-pi) q0[5];
sx q0[5];
rz(1.6647266) q0[5];
sx q0[5];
rz(-pi) q0[6];
sx q0[6];
rz(2.974879) q0[6];
sx q0[6];
rz(-pi) q0[7];
sx q0[7];
rz(2.1924675) q0[7];
sx q0[7];
rz(-pi) q0[8];
sx q0[8];
rz(2.7637507) q0[8];
sx q0[8];
rz(-pi) q0[9];
sx q0[9];
rz(1.5589408) q0[9];
sx q0[9];
rz(-pi) q0[10];
sx q0[10];
rz(0.39912445) q0[10];
sx q0[10];
rz(-pi) q0[11];
sx q0[11];
rz(2.1291289) q0[11];
sx q0[11];
rz(-pi) q0[12];
sx q0[12];
rz(1.6985669) q0[12];
sx q0[12];
rz(-pi) q0[13];
sx q0[13];
rz(2.3267282) q0[13];
sx q0[13];
rz(-pi) q0[14];
sx q0[14];
rz(3.0606925) q0[14];
sx q0[14];
rz(-pi) q0[15];
sx q0[15];
rz(1.9034792) q0[15];
sx q0[15];
rz(-pi) q0[16];
sx q0[16];
rz(0.43254955) q0[16];
sx q0[16];
rz(-pi) q0[17];
sx q0[17];
rz(2.5147754) q0[17];
sx q0[17];
rz(-pi) q0[18];
sx q0[18];
rz(2.9613377) q0[18];
sx q0[18];
rz(-pi) q0[19];
sx q0[19];
rz(2.7853223) q0[19];
sx q0[19];
rz(-pi) q0[20];
sx q0[20];
rz(1.4604031) q0[20];
sx q0[20];
rz(-pi) q0[21];
sx q0[21];
rz(2.6470844) q0[21];
sx q0[21];
rz(-pi) q0[22];
sx q0[22];
rz(0.52256755) q0[22];
sx q0[22];
rz(-pi) q0[23];
sx q0[23];
rz(1.2142906) q0[23];
sx q0[23];
rz(-pi) q0[24];
sx q0[24];
rz(0.015084354) q0[24];
sx q0[24];
rz(-pi) q0[25];
sx q0[25];
rz(0.34689975) q0[25];
sx q0[25];
rz(-pi) q0[26];
sx q0[26];
rz(1.209805) q0[26];
sx q0[26];
rz(-pi) q0[27];
sx q0[27];
rz(1.5756979) q0[27];
sx q0[27];
rz(-pi) q0[28];
sx q0[28];
rz(0.58005835) q0[28];
sx q0[28];
rz(-pi) q0[29];
sx q0[29];
rz(1.0505506) q0[29];
sx q0[29];
rz(-pi) q0[30];
sx q0[30];
rz(0.64174615) q0[30];
sx q0[30];
rz(-pi) q0[31];
sx q0[31];
rz(2.4817246) q0[31];
sx q0[31];
rz(-pi) q0[32];
sx q0[32];
rz(0.62089055) q0[32];
sx q0[32];
rz(-pi) q0[33];
sx q0[33];
rz(2.683352) q0[33];
sx q0[33];
rz(-pi) q0[34];
sx q0[34];
rz(2.4897089) q0[34];
sx q0[34];
cx q0[34],q0[1];
rz(pi/2) q0[34];
sx q0[34];
rz(pi/2) q0[34];
cx q0[1],q0[34];
rz(-pi/4) q0[34];
cx q0[0],q0[34];
rz(pi/4) q0[34];
cx q0[1],q0[34];
rz(pi/4) q0[1];
rz(-pi/4) q0[34];
cx q0[0],q0[34];
cx q0[0],q0[1];
rz(pi/4) q0[0];
rz(-pi/4) q0[1];
cx q0[0],q0[1];
rz(3*pi/4) q0[34];
sx q0[34];
rz(pi/2) q0[34];
cx q0[34],q0[1];
rz(-pi) q0[35];
sx q0[35];
rz(1.9864067) q0[35];
sx q0[35];
cx q0[35],q0[2];
rz(pi/2) q0[35];
sx q0[35];
rz(pi/2) q0[35];
cx q0[2],q0[35];
rz(-pi/4) q0[35];
cx q0[0],q0[35];
rz(pi/4) q0[35];
cx q0[2],q0[35];
rz(pi/4) q0[2];
rz(-pi/4) q0[35];
cx q0[0],q0[35];
cx q0[0],q0[2];
rz(pi/4) q0[0];
rz(-pi/4) q0[2];
cx q0[0],q0[2];
rz(3*pi/4) q0[35];
sx q0[35];
rz(pi/2) q0[35];
cx q0[35],q0[2];
rz(-pi) q0[36];
sx q0[36];
rz(3.1104143) q0[36];
sx q0[36];
cx q0[36],q0[3];
rz(pi/2) q0[36];
sx q0[36];
rz(pi/2) q0[36];
cx q0[3],q0[36];
rz(-pi/4) q0[36];
cx q0[0],q0[36];
rz(pi/4) q0[36];
cx q0[3],q0[36];
rz(pi/4) q0[3];
rz(-pi/4) q0[36];
cx q0[0],q0[36];
cx q0[0],q0[3];
rz(pi/4) q0[0];
rz(-pi/4) q0[3];
cx q0[0],q0[3];
rz(3*pi/4) q0[36];
sx q0[36];
rz(pi/2) q0[36];
cx q0[36],q0[3];
rz(-pi) q0[37];
sx q0[37];
rz(1.3995543) q0[37];
sx q0[37];
cx q0[37],q0[4];
rz(pi/2) q0[37];
sx q0[37];
rz(pi/2) q0[37];
cx q0[4],q0[37];
rz(-pi/4) q0[37];
cx q0[0],q0[37];
rz(pi/4) q0[37];
cx q0[4],q0[37];
rz(-pi/4) q0[37];
cx q0[0],q0[37];
rz(3*pi/4) q0[37];
sx q0[37];
rz(pi/2) q0[37];
rz(pi/4) q0[4];
cx q0[0],q0[4];
rz(pi/4) q0[0];
rz(-pi/4) q0[4];
cx q0[0],q0[4];
cx q0[37],q0[4];
rz(-pi) q0[38];
sx q0[38];
rz(0.45923125) q0[38];
sx q0[38];
cx q0[38],q0[5];
rz(pi/2) q0[38];
sx q0[38];
rz(pi/2) q0[38];
cx q0[5],q0[38];
rz(-pi/4) q0[38];
cx q0[0],q0[38];
rz(pi/4) q0[38];
cx q0[5],q0[38];
rz(-pi/4) q0[38];
cx q0[0],q0[38];
rz(3*pi/4) q0[38];
sx q0[38];
rz(pi/2) q0[38];
rz(pi/4) q0[5];
cx q0[0],q0[5];
rz(pi/4) q0[0];
rz(-pi/4) q0[5];
cx q0[0],q0[5];
cx q0[38],q0[5];
rz(-pi) q0[39];
sx q0[39];
rz(1.5698768) q0[39];
sx q0[39];
cx q0[39],q0[6];
rz(pi/2) q0[39];
sx q0[39];
rz(pi/2) q0[39];
cx q0[6],q0[39];
rz(-pi/4) q0[39];
cx q0[0],q0[39];
rz(pi/4) q0[39];
cx q0[6],q0[39];
rz(-pi/4) q0[39];
cx q0[0],q0[39];
rz(3*pi/4) q0[39];
sx q0[39];
rz(pi/2) q0[39];
rz(pi/4) q0[6];
cx q0[0],q0[6];
rz(pi/4) q0[0];
rz(-pi/4) q0[6];
cx q0[0],q0[6];
cx q0[39],q0[6];
rz(-pi) q0[40];
sx q0[40];
rz(3.0071407) q0[40];
sx q0[40];
cx q0[40],q0[7];
rz(pi/2) q0[40];
sx q0[40];
rz(pi/2) q0[40];
cx q0[7],q0[40];
rz(-pi/4) q0[40];
cx q0[0],q0[40];
rz(pi/4) q0[40];
cx q0[7],q0[40];
rz(-pi/4) q0[40];
cx q0[0],q0[40];
rz(3*pi/4) q0[40];
sx q0[40];
rz(pi/2) q0[40];
rz(pi/4) q0[7];
cx q0[0],q0[7];
rz(pi/4) q0[0];
rz(-pi/4) q0[7];
cx q0[0],q0[7];
cx q0[40],q0[7];
rz(-pi) q0[41];
sx q0[41];
rz(0.072703654) q0[41];
sx q0[41];
cx q0[41],q0[8];
rz(pi/2) q0[41];
sx q0[41];
rz(pi/2) q0[41];
cx q0[8],q0[41];
rz(-pi/4) q0[41];
cx q0[0],q0[41];
rz(pi/4) q0[41];
cx q0[8],q0[41];
rz(-pi/4) q0[41];
cx q0[0],q0[41];
rz(3*pi/4) q0[41];
sx q0[41];
rz(pi/2) q0[41];
rz(pi/4) q0[8];
cx q0[0],q0[8];
rz(pi/4) q0[0];
rz(-pi/4) q0[8];
cx q0[0],q0[8];
cx q0[41],q0[8];
rz(-pi) q0[42];
sx q0[42];
rz(0.94056205) q0[42];
sx q0[42];
cx q0[42],q0[9];
rz(pi/2) q0[42];
sx q0[42];
rz(pi/2) q0[42];
cx q0[9],q0[42];
rz(-pi/4) q0[42];
cx q0[0],q0[42];
rz(pi/4) q0[42];
cx q0[9],q0[42];
rz(-pi/4) q0[42];
cx q0[0],q0[42];
rz(3*pi/4) q0[42];
sx q0[42];
rz(pi/2) q0[42];
rz(pi/4) q0[9];
cx q0[0],q0[9];
rz(pi/4) q0[0];
rz(-pi/4) q0[9];
cx q0[0],q0[9];
cx q0[42],q0[9];
rz(-pi) q0[43];
sx q0[43];
rz(3.0933901) q0[43];
sx q0[43];
cx q0[43],q0[10];
rz(pi/2) q0[43];
sx q0[43];
rz(pi/2) q0[43];
cx q0[10],q0[43];
rz(-pi/4) q0[43];
cx q0[0],q0[43];
rz(pi/4) q0[43];
cx q0[10],q0[43];
rz(pi/4) q0[10];
rz(-pi/4) q0[43];
cx q0[0],q0[43];
cx q0[0],q0[10];
rz(pi/4) q0[0];
rz(-pi/4) q0[10];
cx q0[0],q0[10];
rz(3*pi/4) q0[43];
sx q0[43];
rz(pi/2) q0[43];
cx q0[43],q0[10];
rz(-pi) q0[44];
sx q0[44];
rz(2.8882807) q0[44];
sx q0[44];
cx q0[44],q0[11];
rz(pi/2) q0[44];
sx q0[44];
rz(pi/2) q0[44];
cx q0[11],q0[44];
rz(-pi/4) q0[44];
cx q0[0],q0[44];
rz(pi/4) q0[44];
cx q0[11],q0[44];
rz(pi/4) q0[11];
rz(-pi/4) q0[44];
cx q0[0],q0[44];
cx q0[0],q0[11];
rz(pi/4) q0[0];
rz(-pi/4) q0[11];
cx q0[0],q0[11];
rz(3*pi/4) q0[44];
sx q0[44];
rz(pi/2) q0[44];
cx q0[44],q0[11];
rz(-pi) q0[45];
sx q0[45];
rz(1.7981791) q0[45];
sx q0[45];
cx q0[45],q0[12];
rz(pi/2) q0[45];
sx q0[45];
rz(pi/2) q0[45];
cx q0[12],q0[45];
rz(-pi/4) q0[45];
cx q0[0],q0[45];
rz(pi/4) q0[45];
cx q0[12],q0[45];
rz(pi/4) q0[12];
rz(-pi/4) q0[45];
cx q0[0],q0[45];
cx q0[0],q0[12];
rz(pi/4) q0[0];
rz(-pi/4) q0[12];
cx q0[0],q0[12];
rz(3*pi/4) q0[45];
sx q0[45];
rz(pi/2) q0[45];
cx q0[45],q0[12];
rz(-pi) q0[46];
sx q0[46];
rz(1.7299686) q0[46];
sx q0[46];
cx q0[46],q0[13];
rz(pi/2) q0[46];
sx q0[46];
rz(pi/2) q0[46];
cx q0[13],q0[46];
rz(-pi/4) q0[46];
cx q0[0],q0[46];
rz(pi/4) q0[46];
cx q0[13],q0[46];
rz(pi/4) q0[13];
rz(-pi/4) q0[46];
cx q0[0],q0[46];
cx q0[0],q0[13];
rz(pi/4) q0[0];
rz(-pi/4) q0[13];
cx q0[0],q0[13];
rz(3*pi/4) q0[46];
sx q0[46];
rz(pi/2) q0[46];
cx q0[46],q0[13];
rz(-pi) q0[47];
sx q0[47];
rz(1.2954459) q0[47];
sx q0[47];
cx q0[47],q0[14];
rz(pi/2) q0[47];
sx q0[47];
rz(pi/2) q0[47];
cx q0[14],q0[47];
rz(-pi/4) q0[47];
cx q0[0],q0[47];
rz(pi/4) q0[47];
cx q0[14],q0[47];
rz(pi/4) q0[14];
rz(-pi/4) q0[47];
cx q0[0],q0[47];
cx q0[0],q0[14];
rz(pi/4) q0[0];
rz(-pi/4) q0[14];
cx q0[0],q0[14];
rz(3*pi/4) q0[47];
sx q0[47];
rz(pi/2) q0[47];
cx q0[47],q0[14];
rz(-pi) q0[48];
sx q0[48];
rz(1.8923155) q0[48];
sx q0[48];
cx q0[48],q0[15];
rz(pi/2) q0[48];
sx q0[48];
rz(pi/2) q0[48];
cx q0[15],q0[48];
rz(-pi/4) q0[48];
cx q0[0],q0[48];
rz(pi/4) q0[48];
cx q0[15],q0[48];
rz(pi/4) q0[15];
rz(-pi/4) q0[48];
cx q0[0],q0[48];
cx q0[0],q0[15];
rz(pi/4) q0[0];
rz(-pi/4) q0[15];
cx q0[0],q0[15];
rz(3*pi/4) q0[48];
sx q0[48];
rz(pi/2) q0[48];
cx q0[48],q0[15];
rz(-pi) q0[49];
sx q0[49];
rz(2.085722) q0[49];
sx q0[49];
cx q0[49],q0[16];
rz(pi/2) q0[49];
sx q0[49];
rz(pi/2) q0[49];
cx q0[16],q0[49];
rz(-pi/4) q0[49];
cx q0[0],q0[49];
rz(pi/4) q0[49];
cx q0[16],q0[49];
rz(pi/4) q0[16];
rz(-pi/4) q0[49];
cx q0[0],q0[49];
cx q0[0],q0[16];
rz(pi/4) q0[0];
rz(-pi/4) q0[16];
cx q0[0],q0[16];
rz(3*pi/4) q0[49];
sx q0[49];
rz(pi/2) q0[49];
cx q0[49],q0[16];
rz(-pi) q0[50];
sx q0[50];
rz(0.55436445) q0[50];
sx q0[50];
cx q0[50],q0[17];
rz(pi/2) q0[50];
sx q0[50];
rz(pi/2) q0[50];
cx q0[17],q0[50];
rz(-pi/4) q0[50];
cx q0[0],q0[50];
rz(pi/4) q0[50];
cx q0[17],q0[50];
rz(pi/4) q0[17];
rz(-pi/4) q0[50];
cx q0[0],q0[50];
cx q0[0],q0[17];
rz(pi/4) q0[0];
rz(-pi/4) q0[17];
cx q0[0],q0[17];
rz(3*pi/4) q0[50];
sx q0[50];
rz(pi/2) q0[50];
cx q0[50],q0[17];
rz(-pi) q0[51];
sx q0[51];
rz(1.0406347) q0[51];
sx q0[51];
cx q0[51],q0[18];
rz(pi/2) q0[51];
sx q0[51];
rz(pi/2) q0[51];
cx q0[18],q0[51];
rz(-pi/4) q0[51];
cx q0[0],q0[51];
rz(pi/4) q0[51];
cx q0[18],q0[51];
rz(pi/4) q0[18];
rz(-pi/4) q0[51];
cx q0[0],q0[51];
cx q0[0],q0[18];
rz(pi/4) q0[0];
rz(-pi/4) q0[18];
cx q0[0],q0[18];
rz(3*pi/4) q0[51];
sx q0[51];
rz(pi/2) q0[51];
cx q0[51],q0[18];
rz(-pi) q0[52];
sx q0[52];
rz(1.5848364) q0[52];
sx q0[52];
cx q0[52],q0[19];
rz(pi/2) q0[52];
sx q0[52];
rz(pi/2) q0[52];
cx q0[19],q0[52];
rz(-pi/4) q0[52];
cx q0[0],q0[52];
rz(pi/4) q0[52];
cx q0[19],q0[52];
rz(pi/4) q0[19];
rz(-pi/4) q0[52];
cx q0[0],q0[52];
cx q0[0],q0[19];
rz(pi/4) q0[0];
rz(-pi/4) q0[19];
cx q0[0],q0[19];
rz(3*pi/4) q0[52];
sx q0[52];
rz(pi/2) q0[52];
cx q0[52],q0[19];
rz(-pi) q0[53];
sx q0[53];
rz(0.84730035) q0[53];
sx q0[53];
cx q0[53],q0[20];
rz(pi/2) q0[53];
sx q0[53];
rz(pi/2) q0[53];
cx q0[20],q0[53];
rz(-pi/4) q0[53];
cx q0[0],q0[53];
rz(pi/4) q0[53];
cx q0[20],q0[53];
rz(pi/4) q0[20];
rz(-pi/4) q0[53];
cx q0[0],q0[53];
cx q0[0],q0[20];
rz(pi/4) q0[0];
rz(-pi/4) q0[20];
cx q0[0],q0[20];
rz(3*pi/4) q0[53];
sx q0[53];
rz(pi/2) q0[53];
cx q0[53],q0[20];
rz(-pi) q0[54];
sx q0[54];
rz(1.0187866) q0[54];
sx q0[54];
cx q0[54],q0[21];
rz(pi/2) q0[54];
sx q0[54];
rz(pi/2) q0[54];
cx q0[21],q0[54];
rz(-pi/4) q0[54];
cx q0[0],q0[54];
rz(pi/4) q0[54];
cx q0[21],q0[54];
rz(pi/4) q0[21];
rz(-pi/4) q0[54];
cx q0[0],q0[54];
cx q0[0],q0[21];
rz(pi/4) q0[0];
rz(-pi/4) q0[21];
cx q0[0],q0[21];
rz(3*pi/4) q0[54];
sx q0[54];
rz(pi/2) q0[54];
cx q0[54],q0[21];
rz(-pi) q0[55];
sx q0[55];
rz(1.4078884) q0[55];
sx q0[55];
cx q0[55],q0[22];
rz(pi/2) q0[55];
sx q0[55];
rz(pi/2) q0[55];
cx q0[22],q0[55];
rz(-pi/4) q0[55];
cx q0[0],q0[55];
rz(pi/4) q0[55];
cx q0[22],q0[55];
rz(pi/4) q0[22];
rz(-pi/4) q0[55];
cx q0[0],q0[55];
cx q0[0],q0[22];
rz(pi/4) q0[0];
rz(-pi/4) q0[22];
cx q0[0],q0[22];
rz(3*pi/4) q0[55];
sx q0[55];
rz(pi/2) q0[55];
cx q0[55],q0[22];
rz(-pi) q0[56];
sx q0[56];
rz(3.0103216) q0[56];
sx q0[56];
cx q0[56],q0[23];
rz(pi/2) q0[56];
sx q0[56];
rz(pi/2) q0[56];
cx q0[23],q0[56];
rz(-pi/4) q0[56];
cx q0[0],q0[56];
rz(pi/4) q0[56];
cx q0[23],q0[56];
rz(pi/4) q0[23];
rz(-pi/4) q0[56];
cx q0[0],q0[56];
cx q0[0],q0[23];
rz(pi/4) q0[0];
rz(-pi/4) q0[23];
cx q0[0],q0[23];
rz(3*pi/4) q0[56];
sx q0[56];
rz(pi/2) q0[56];
cx q0[56],q0[23];
rz(-pi) q0[57];
sx q0[57];
rz(2.703608) q0[57];
sx q0[57];
cx q0[57],q0[24];
rz(pi/2) q0[57];
sx q0[57];
rz(pi/2) q0[57];
cx q0[24],q0[57];
rz(-pi/4) q0[57];
cx q0[0],q0[57];
rz(pi/4) q0[57];
cx q0[24],q0[57];
rz(pi/4) q0[24];
rz(-pi/4) q0[57];
cx q0[0],q0[57];
cx q0[0],q0[24];
rz(pi/4) q0[0];
rz(-pi/4) q0[24];
cx q0[0],q0[24];
rz(3*pi/4) q0[57];
sx q0[57];
rz(pi/2) q0[57];
cx q0[57],q0[24];
rz(-pi) q0[58];
sx q0[58];
rz(3.0296793) q0[58];
sx q0[58];
cx q0[58],q0[25];
rz(pi/2) q0[58];
sx q0[58];
rz(pi/2) q0[58];
cx q0[25],q0[58];
rz(-pi/4) q0[58];
cx q0[0],q0[58];
rz(pi/4) q0[58];
cx q0[25],q0[58];
rz(pi/4) q0[25];
rz(-pi/4) q0[58];
cx q0[0],q0[58];
cx q0[0],q0[25];
rz(pi/4) q0[0];
rz(-pi/4) q0[25];
cx q0[0],q0[25];
rz(3*pi/4) q0[58];
sx q0[58];
rz(pi/2) q0[58];
cx q0[58],q0[25];
rz(-pi) q0[59];
sx q0[59];
rz(1.8511426) q0[59];
sx q0[59];
cx q0[59],q0[26];
rz(pi/2) q0[59];
sx q0[59];
rz(pi/2) q0[59];
cx q0[26],q0[59];
rz(-pi/4) q0[59];
cx q0[0],q0[59];
rz(pi/4) q0[59];
cx q0[26],q0[59];
rz(pi/4) q0[26];
rz(-pi/4) q0[59];
cx q0[0],q0[59];
cx q0[0],q0[26];
rz(pi/4) q0[0];
rz(-pi/4) q0[26];
cx q0[0],q0[26];
rz(3*pi/4) q0[59];
sx q0[59];
rz(pi/2) q0[59];
cx q0[59],q0[26];
rz(-pi) q0[60];
sx q0[60];
rz(0.51437195) q0[60];
sx q0[60];
cx q0[60],q0[27];
rz(pi/2) q0[60];
sx q0[60];
rz(pi/2) q0[60];
cx q0[27],q0[60];
rz(-pi/4) q0[60];
cx q0[0],q0[60];
rz(pi/4) q0[60];
cx q0[27],q0[60];
rz(pi/4) q0[27];
rz(-pi/4) q0[60];
cx q0[0],q0[60];
cx q0[0],q0[27];
rz(pi/4) q0[0];
rz(-pi/4) q0[27];
cx q0[0],q0[27];
rz(3*pi/4) q0[60];
sx q0[60];
rz(pi/2) q0[60];
cx q0[60],q0[27];
rz(-pi) q0[61];
sx q0[61];
rz(1.5343856) q0[61];
sx q0[61];
cx q0[61],q0[28];
rz(pi/2) q0[61];
sx q0[61];
rz(pi/2) q0[61];
cx q0[28],q0[61];
rz(-pi/4) q0[61];
cx q0[0],q0[61];
rz(pi/4) q0[61];
cx q0[28],q0[61];
rz(pi/4) q0[28];
rz(-pi/4) q0[61];
cx q0[0],q0[61];
cx q0[0],q0[28];
rz(pi/4) q0[0];
rz(-pi/4) q0[28];
cx q0[0],q0[28];
rz(3*pi/4) q0[61];
sx q0[61];
rz(pi/2) q0[61];
cx q0[61],q0[28];
rz(-pi) q0[62];
sx q0[62];
rz(0.52056245) q0[62];
sx q0[62];
cx q0[62],q0[29];
rz(pi/2) q0[62];
sx q0[62];
rz(pi/2) q0[62];
cx q0[29],q0[62];
rz(-pi/4) q0[62];
cx q0[0],q0[62];
rz(pi/4) q0[62];
cx q0[29],q0[62];
rz(pi/4) q0[29];
rz(-pi/4) q0[62];
cx q0[0],q0[62];
cx q0[0],q0[29];
rz(pi/4) q0[0];
rz(-pi/4) q0[29];
cx q0[0],q0[29];
rz(3*pi/4) q0[62];
sx q0[62];
rz(pi/2) q0[62];
cx q0[62],q0[29];
rz(-pi) q0[63];
sx q0[63];
rz(0.35085345) q0[63];
sx q0[63];
cx q0[63],q0[30];
rz(pi/2) q0[63];
sx q0[63];
rz(pi/2) q0[63];
cx q0[30],q0[63];
rz(-pi/4) q0[63];
cx q0[0],q0[63];
rz(pi/4) q0[63];
cx q0[30],q0[63];
rz(pi/4) q0[30];
rz(-pi/4) q0[63];
cx q0[0],q0[63];
cx q0[0],q0[30];
rz(pi/4) q0[0];
rz(-pi/4) q0[30];
cx q0[0],q0[30];
rz(3*pi/4) q0[63];
sx q0[63];
rz(pi/2) q0[63];
cx q0[63],q0[30];
rz(-pi) q0[64];
sx q0[64];
rz(2.1894482) q0[64];
sx q0[64];
cx q0[64],q0[31];
rz(pi/2) q0[64];
sx q0[64];
rz(pi/2) q0[64];
cx q0[31],q0[64];
rz(-pi/4) q0[64];
cx q0[0],q0[64];
rz(pi/4) q0[64];
cx q0[31],q0[64];
rz(pi/4) q0[31];
rz(-pi/4) q0[64];
cx q0[0],q0[64];
cx q0[0],q0[31];
rz(pi/4) q0[0];
rz(-pi/4) q0[31];
cx q0[0],q0[31];
rz(3*pi/4) q0[64];
sx q0[64];
rz(pi/2) q0[64];
cx q0[64],q0[31];
rz(-pi) q0[65];
sx q0[65];
rz(2.107859) q0[65];
sx q0[65];
cx q0[65],q0[32];
rz(pi/2) q0[65];
sx q0[65];
rz(pi/2) q0[65];
cx q0[32],q0[65];
rz(-pi/4) q0[65];
cx q0[0],q0[65];
rz(pi/4) q0[65];
cx q0[32],q0[65];
rz(pi/4) q0[32];
rz(-pi/4) q0[65];
cx q0[0],q0[65];
cx q0[0],q0[32];
rz(pi/4) q0[0];
rz(-pi/4) q0[32];
cx q0[0],q0[32];
rz(3*pi/4) q0[65];
sx q0[65];
rz(pi/2) q0[65];
cx q0[65],q0[32];
rz(-pi) q0[66];
sx q0[66];
rz(1.3992582) q0[66];
sx q0[66];
cx q0[66],q0[33];
rz(pi/2) q0[66];
sx q0[66];
rz(pi/2) q0[66];
cx q0[33],q0[66];
rz(-pi/4) q0[66];
cx q0[0],q0[66];
rz(pi/4) q0[66];
cx q0[33],q0[66];
rz(pi/4) q0[33];
rz(-pi/4) q0[66];
cx q0[0],q0[66];
cx q0[0],q0[33];
rz(pi/4) q0[0];
rz(-pi/4) q0[33];
cx q0[0],q0[33];
rz(pi/2) q0[0];
sx q0[0];
rz(pi/2) q0[0];
rz(3*pi/4) q0[66];
sx q0[66];
rz(pi/2) q0[66];
cx q0[66],q0[33];
measure q0[0] -> c0[0];