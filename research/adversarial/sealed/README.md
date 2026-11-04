# Sealed-split salts (custodian: main agent)
A salt is committed here only AFTER the builder's frozen floor list is in git, so floors cannot be tuned to the sealed fold.
- t_cxr_salt.txt: committed after the R-CXR floor list (37 image floors + labels14 code floor, sha256 41c6b059...c01d38) was frozen in acdeb7a. Roles = SHA-256 of `salt:pid` per T.md.
- h_imu_salt.txt: committed after the R-IMU floor list (research/adversarial/src/h_imu/FLOOR_LIST.json, sha256 fc1ba9c8...c5064, verified by the custodian) was frozen in ca9b9df. Partition: per (subject, activity) trials ranked by sha256("{subject}|{activity}|{trial}|{salt}"), fold = rank mod 3. The salt applies only to defenders retrained inside the programme; the shipped-checkpoint fold-O result is a burned-split audit.
The salt is not secret (it is public in git); what protects the fold is the ORDER: floors frozen first, salt second.
