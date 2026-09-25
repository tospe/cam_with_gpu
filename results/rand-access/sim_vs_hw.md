| warps/SM | lanes/warp | chains | HW cycles/hop | sim cycles/hop | err | sim DRAM reads/hop (H2) | sim L2 acc/hop (H2) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 114 | 698 | 649 | -7.0% | 1.000 | 1.50 |
| 1 | 4 | 456 | 892 | 733 | -17.9% | 1.000 | 1.50 |
| 1 | 32 | 3648 | 1106 | 758 | -31.5% | 1.000 | 1.50 |
| 2 | 16 | 3648 | 1023 | 749 | -26.8% | 1.000 | 1.50 |
| 4 | 8 | 3648 | 954 | 745 | -21.9% | 1.000 | 1.50 |
| 8 | 4 | 3648 | 894 | 733 | -18.0% | 1.000 | 1.50 |
| 16 | 2 | 3648 | 830 | 700 | -15.7% | 1.000 | 1.50 |
| 32 | 1 | 3648 | 764 | 652 | -14.6% | 1.000 | 1.50 |

Interpretation (2026-09-25): at fixed 3648 chains the dev-config simulator, which has no per-access DRAM
latency variance, still rises 652 -> 758 cycles/hop (+106) from 32x1 to 1x32; HW rises 764 -> 1106 (+342).
So at least ~1/3 of the HW lanes effect is a per-warp cost unrelated to latency variance (e.g. divergent request
handling, partition queueing). "Slowest-lane effects dominate" is NOT supported as stated; at most the remaining
~2/3 could come from variance, which is not demonstrated. Resident warps at 1 lane: sim flat (649 -> 652),
HW +66 (698 -> 764). No HW transaction counters available (ncu absent, profiling admin-only); HW outstanding =
chains by construction; sim DRAM reads/hop = 1.000 and L2 accesses/hop = 1.50 in every case.
