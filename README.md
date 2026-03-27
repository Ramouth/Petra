# Petra

## HPC Access (DTU gbar)

**SSH alias:** `gbar` (key at `~/.ssh/gbar`)

**From campus:** SSH works directly, no VPN needed.

**From home:** SSH key authentication works without VPN.
Requires: ssh-key + passphrase + DTU password.
```bash
ssh gbar
rsync -av gbar:~/Petra-Phase1/models/zigzag/r2/best.pt ~/Documents/Code/Petra-Phase1/models/zigzag/r2/
```

**VPN (openconnect):** Currently broken — server rejects client as outdated version.
Workarounds tried: `--version-string "4.10.07073"`, `--os win`. Both fail with
`anyconnect_unsupported_version.html`. SSH key from outside is the reliable path.

**Stockfish on HPC:** `/zhome/81/b/206091/bin/stockfish`

**Python venv:** `~/petra-env`

## Pulling results after a job

```bash
# Pull model + logs
rsync -av gbar:~/Petra-Phase1/models/zigzag/r2/best.pt ~/Documents/Code/Petra-Phase1/models/zigzag/r2/
rsync -av "gbar:~/Petra-Phase1/logs/lsf_r2_*.log" ~/Documents/Code/Petra-Phase1/logs/

# Delete intermediate data on HPC (compliance)
ssh gbar "rm -f ~/Petra-Phase1/data/selfplay_r2*.pt"
```
