# Troubleshooting

Common issues

- Wrong cache dir: local `cache/tusz/`, Modal `/results/cache/tusz/`
- No seizures in batches: enable `use_balanced_sampling`
- NaN losses on 4090: set `mixed_precision: false`
- Modal stuck: increase CPU (24) and RAM (96GB)
- PyG install fails: use prebuilt wheels
