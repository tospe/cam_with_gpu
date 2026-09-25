FFMA/cyc/SM counts FFMA instructions per thread (1 FFMA = 2 FLOP). Candidate -trace_opcode_latency_initiation_sp 3,1.

| set | case | HW median cycles/iter | HW FFMA/cyc/SM | sim cycles/iter | sim FFMA/cyc/SM | error (time) | verdict |
|---|---|---|---|---|---|---|---|
| fresh | F1 K1 W2 (dependency) | 4.461 | 14.3 | 6.000 | 10.7 | +34.5% | FAIL |
| fresh | F2 K4 W8 (intermediate) | 9.415 | 108.8 | 8.625 | 118.7 | -8.4% | PASS |
| fresh | F3 K16 W16 (saturated) | 70.109 | 116.8 | 68.000 | 120.5 | -3.0% | PASS |
| regr | R1 chain 4096/16384 (fit case) | 4.438 | — | 6.000 | — | +35.2% | FAIL |
| regr | R1 chain 1024/8192 | 4.438 | — | 6.000 | — | +35.2% | FAIL |
| regr | R1 chain 2048/32768 | 4.438 | — | 6.000 | — | +35.2% | FAIL |
| regr | R2 K2 W4 | 5.155 | 49.7 | 6.375 | 40.2 | +23.7% | FAIL |
| regr | R2 K2 W32 | 18.391 | 111.4 | 18.000 | 113.8 | -2.1% | PASS |
| regr | R2 K8 W4 | 12.907 | 79.3 | 10.500 | 97.5 | -18.6% | FAIL |
| regr | R2 K8 W32 | 76.123 | 107.6 | 72.000 | 113.8 | -5.4% | PASS |
| regr | R3 arithmetic ring (µs, cold) | 57.44 | — | 59.69 | — | +3.9% | PASS |
