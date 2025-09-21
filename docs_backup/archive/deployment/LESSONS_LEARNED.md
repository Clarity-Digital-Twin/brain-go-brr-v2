# Modal Deployment Lessons Learned (Archived)

- Image build order: add_local_dir/file last
- Use CUDA devel images for mamba-ssm
- Install PyTorch → numpy → mamba-ssm (order matters)
- First build compiles kernels; subsequent runs are cached
- Prefer A100-80GB for training; T4 for debug

Refer to docs/deployment/MODAL_MAMBA_DEPLOYMENT_ISSUES.md for active issues/notes.

