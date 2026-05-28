# Gauge experiment matrix

The canonical command generator is:

```bash
python3 -m pebble_gauge.run_sweep --phase all --mode dry-run
```

RRM runs once per environment, alpha mode, seed, and final-activation family:
`none` covers `none/mean_buf/mean_ref`, while `no_tanh` covers
`no_tanh/no_tanh_mean_buf/no_tanh_mean_ref`.

PerfG runs once per active gauge.

Metaworld jobs need MuJoCo library paths. `run_sweep.py --mode slurm`
exports:

```bash
MUJOCO_GL=egl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zifan/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

Default run outputs are written under
`/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/pg_gauge`, and Slurm
stdout/stderr logs are written under
`/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/logs`.
