# Sealed-split salts (custodian: main agent)
A salt is committed here only AFTER the builder's frozen floor list is in git, so floors cannot be tuned to the sealed fold.
- t_cxr_salt.txt: committed after the R-CXR floor list (37 image floors + labels14 code floor, sha256 41c6b059...c01d38) was frozen in acdeb7a. Roles = SHA-256 of `salt:pid` per T.md.
- h_imu salt: NOT yet committed (waiting for Builder H to report that its floor list is frozen).
The salt is not secret (it is public in git); what protects the fold is the ORDER: floors frozen first, salt second.
