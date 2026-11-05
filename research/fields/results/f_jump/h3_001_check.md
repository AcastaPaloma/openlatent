# H3-001 closing check (F.md section 24); B=1000

## (1) Pabc shuffled null of ctrl_noise16, per staged fold

| fold | U Pabc (stored) | U Pabc (recomputed) | U shuffled-label null (Pabc) | |dU| | control holds (<= 0.01) | V: U / null |
|---|---|---|---|---|---|---|
| blk | 0.0732 | 0.0732 | 0.0762 | 0.0030 | True (reproducible True) | 0.2469 / 0.2536 |
| s1 | 0.2309 | 0.2309 | 0.2546 | 0.0238 | False (reproducible True) | 0.5689 / 0.5999 |
| s9 | 0.1325 | 0.1325 | 0.1480 | 0.0155 | False (reproducible True) | 0.3921 / 0.4260 |
| s11 | 0.1113 | 0.1113 | 0.1023 | 0.0090 | True (reproducible True) | 0.3598 / 0.3197 |

Pabc fold blk with the corrected control: controls ok=True (U(noise16) 0.073 vs Pabc null 0.076), valid=False, powered=False, trained b100 adapter=no

Pabc fold s1 with the corrected control: controls ok=False (U(noise16) 0.231 vs Pabc null 0.255), valid=False, powered=False, trained b100 adapter=no

Pabc fold s9 with the corrected control: controls ok=False (U(noise16) 0.133 vs Pabc null 0.148), valid=False, powered=False, trained b100 adapter=no

Pabc fold s11 with the corrected control: controls ok=True (U(noise16) 0.111 vs Pabc null 0.102), valid=False, powered=False, trained b100 adapter=no

**verdict_tightened (Pabc) with the corrected control: INVALID** (valid and powered folds [])

## (2) seen-source reference, P0: **R-A SEEN-SOURCE GAIN, UNSEEN-SOURCE FAILURE**

valid=True (controls True: U(noise16) 0.043 vs null 0.044, attacker 0.102; powered=True MDE80 b10 0.0013, b100 0.0016; trained b10 ['mlp_b10_d8', 'mlp_b10_d8_r0.5'], b100 ['mlp_b100_d8', 'mlp_b100_d8_r0.5']); headroom 0.060 (ceiling mlp_b100_d8 U_H 0.172, best floor ldaM_b100_8 U_H 0.111)

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b10 | mlp_b10_d8 | 0.127 [0.33] | 0.0162 [0.0153, 0.0171] (0.0121, 0.0143) | ldaM_b100_8 | True | mlp_b10_d8 | 0.0269 [0.0260, 0.0278] | True | 0.027 |
| b100 | mlp_b100_d8 | 0.172 [0.44] | 0.0603 [0.0594, 0.0614] (0.0467, 0.0517) | ldaM_b100_8 | True | mlp_b100_d8 | 0.0603 [0.0594, 0.0614] | True | 0.060 |

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b10_d8 | 0.32 | 7.2 | 0.294 | 0.144 | 0.039 | True |
| mlp_b10_d8_r0.5 | 0.32 | 7.2 | 0.265 | 0.144 | 0.039 | True |
| mlp_b100_d8 | 0.59 | 6.5 | 0.370 | 0.153 | 0.046 | True |
| mlp_b100_d8_r0.5 | 0.59 | 6.5 | 0.316 | 0.153 | 0.046 | True |

## (2) seen-source reference, Pabc: **R-A SEEN-SOURCE GAIN, UNSEEN-SOURCE FAILURE**

valid=True (controls True: U(noise16) 0.111 vs null 0.120, attacker 0.102; powered=True MDE80 b10 0.0016, b100 0.0023; trained b10 ['mlp_b10_d8', 'mlp_b10_d8_r0.5'], b100 ['mlp_b100_d8', 'mlp_b100_d8_r0.5']); headroom 0.107 (ceiling mlp_b100_d8 U_H 0.295, best floor ldaP_8 U_H 0.188)

| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |
|---|---|---|---|---|---|---|---|---|---|
| b10 | mlp_b10_d8 | 0.216 [0.33] | 0.0283 [0.0271, 0.0294] (0.0242, 0.0287) | ldaP_8 | True | mlp_b10_d8 | 0.0467 [0.0455, 0.0479] | True | 0.047 |
| b100 | mlp_b100_d8 | 0.295 [0.44] | 0.1068 [0.1053, 0.1082] (0.0903, 0.0998) | ldaP_8 | True | mlp_b100_d8 | 0.1067 [0.1052, 0.1082] | True | 0.107 |

| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |
|---|---|---|---|---|---|---|
| mlp_b10_d8 | 0.32 | 7.2 | 0.451 | 0.311 | 0.103 | True |
| mlp_b10_d8_r0.5 | 0.32 | 7.2 | 0.429 | 0.311 | 0.103 | True |
| mlp_b100_d8 | 0.59 | 6.5 | 0.550 | 0.299 | 0.102 | True |
| mlp_b100_d8_r0.5 | 0.59 | 6.5 | 0.502 | 0.299 | 0.102 | True |
