# Dynamic-k Neighbor Selection with PPO

This branch trains a distance-pointer policy for ACS flocking. For every ego
agent, the policy selects one active agent as a distance pointer. All active
agents no farther away than that pointer become directed neighbors, so the
number of selected neighbors `k` changes with the observation and the policy
output. Selecting the ego itself means zero external neighbors; equal-distance
ties are included.

The policy uses rotated ego-centric observations and a shared
Transformer/pointer network. Training is fixed to 20 agents by default, while
the environment and parameter-only model remain shape-compatible with other
fixed swarm sizes.

## Default training profile

The checked-in profile was validated on an i9-9900KF and one RTX 3090.

| Setting | Default |
|---|---:|
| Swarm size | 20 |
| Environment steps | 6,000,000 |
| Safety time limit | 18 hours |
| Rollout workers / envs per worker | 8 / 2 |
| Rollout fragment / train batch | 512 / 8,192 |
| SGD minibatch / passes | 512 / 7 |
| Learning rate | `2e-5` to `1e-7` over 6M steps |
| Checkpoints | every 8 iterations, best 5 plus final |

The reference run completed 6,004,736 steps in 12.93 hours. Every value above
can be overridden with the environment variables listed by
`./docker/run_train.sh --help`.

## Build and configure W&B

The host needs Docker, the NVIDIA Container Toolkit, and a compatible NVIDIA
driver. Build the Python 3.9 / Ray 2.1 / CUDA 11.3 image with:

```bash
./docker/build.sh
```

W&B logging is enabled by default. Store the API key in a private file; the
runner mounts it read-only and never places the key in Docker environment
variables or command-line arguments.

```bash
mkdir -p ~/.config/wandb
chmod 700 ~/.config/wandb
${EDITOR:-vi} ~/.config/wandb/api_key
chmod 600 ~/.config/wandb/api_key
```

The default project is `nb-selection-distance-pointer`. Override it with
`WANDB_PROJECT`, or set `WANDB_ENABLED=false` to train without W&B.

## Durable background training

Start a named run in a detached container:

```bash
./docker/run_train.sh start --run-id dynamic-k-n20-seed42
```

Build the image automatically when starting, or inspect the fully resolved
Docker command without changing state:

```bash
./docker/run_train.sh start --build --run-id dynamic-k-n20-seed42
./docker/run_train.sh start --dry-run --run-id dynamic-k-n20-seed42
```

Monitor or deliberately stop it with:

```bash
./docker/run_train.sh status
./docker/run_train.sh logs
./docker/run_train.sh logs --no-follow
./docker/run_train.sh stop
```

Results are stored under `test_results/<run-id>/` on the host. The container
uses `restart=unless-stopped`, and Ray Tune uses `AUTO+ERRORED`, so an
unexpected process, container, or host restart resumes the latest available
checkpoint. A failure before the first checkpoint starts again from step zero.
A deliberate `stop` is not automatically restarted; resume that same container
with `docker start dynamic-k-nn-training`.

After successful training, the service writes `.training_complete` in the run
directory and remains idle, preventing a Docker restart loop. Use a different
`CONTAINER_NAME` for concurrent or subsequent retained containers.

Example overrides:

```bash
WANDB_PROJECT=my-project BASE_ENV_SEED=7 \
TOTAL_TRAINING_TIMESTEPS=1000000 \
CONTAINER_NAME=dynamic-k-seed7 \
./docker/run_train.sh start --run-id dynamic-k-n20-seed7
```

## Tests

Run the dynamic action, pointer conversion, padding, cross-size strict-load,
short PPO rollout, and legacy binary regression tests in the image:

```bash
docker run --rm --init --shm-size 4g \
  --workdir /workspace/source \
  --env START_SSHD=0 \
  --mount type=bind,src="$(pwd)",dst=/workspace/source,readonly \
  uom-neighbor-selection \
  python -m unittest -v test_distance_pointer
```
