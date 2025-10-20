# WSL2 Notes

- Set `UV_LINK_MODE=copy` so uv installs don’t create hard links on NTFS.
- Prefer `data.num_workers: 0`, `pin_memory: false`, `persistent_workers: false` to avoid dataloader hangs on the 9P filesystem.
- **CRITICAL**: Keep the mmap cache on a native ext4 volume, not `/mnt/c` or `/mnt/d`. See `docs/08-operations/wsl2-sigbus-fix.md` for the full migration guide and SIGBUS root-cause analysis.
