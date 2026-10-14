#!/usr/bin/env bash
#SBATCH --job-name=psro_stage
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2
#SBATCH --partition=project
#SBATCH --qos=qos_project_preempt
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00
#SBATCH --output=/zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/stage_%j.log
# Build the Docker CONTEXT: tar the two conda envs (vllm30 is node-local -> must run on gpu2),
# the code dirs, role_phase, the pi0 LoRA and the SFT data. Each tar's top entry is the dir name
# so `ADD ctx/<x>.tar <dest>/` extracts to the pinned absolute path.
set -u -o pipefail
CTX=/zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/ctx
mkdir -p $CTX
tarit () {  # <out>  <parent-dir>  <name>
  local out=$1 parent=$2 name=$3
  echo "[stage] $name ($(du -sh -L $parent/$name 2>/dev/null | cut -f1)) -> $out"
  tar -C "$parent" -cf "$CTX/$out" "$name" && echo "  ok $(du -h $CTX/$out | cut -f1)"
}
tarit vllm30.tar       /scratch/wentsec/envs                      vllm30
tarit srt.tar          /zfsauton/scratch/wentsec/envs             srt
tarit ssp_v10xm_g2.tar /zfsauton/scratch/wentsec                 ssp_v10xm_g2
tarit work_v9zp_g2.tar /zfsauton/scratch/wentsec/coldpsro_slurm  work_v9zp_g2
tarit role_phase.tar   /zfsauton/scratch/wentsec/coldpsro_slurm/async_prof  role_phase_v9sb_flame.sh
tarit sft_data.tar     /zfsauton/scratch/wentsec                 sft_data
echo "=== context ready in $CTX ==="; ls -la $CTX; du -sh $CTX
