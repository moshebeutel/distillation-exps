# Starts a Ray instace on Slurm

function start_head_on_matrix() {
	PORT=6379
	DIRECTORY="${DDIST_EXPS_DIR}/ray-logs/"
	MEMORY_GB=100
	GPU_NUM=2
	CPU_NUM=16
	NODELIST=matrix-2-37
	# OBJ_STORE_MEMORY="${MEMORY_GB}000000000" # k: 1e3, m: 1e6, g: 1e9
	# NODELIST=matrix-2-37,matrix-2-33
	# CPU_PER_GPU=2

	mkdir -p ${DIRECTORY}
	echo "Output file: ${DIRECTORY}/slurm.log"
	# export RAY_GRAFANA_HOST="0.0.0.0:9090"
	# Sleep for some time so that we can ensure prometheus starts
	cmd_ray="ray start --head --temp-dir=${DIRECTORY} --port $PORT --block"
	cmd_ray="${cmd_ray}  --num-cpus=${CPU_NUM} --num-gpus=${GPU_NUM}"
	# cmd_ray="${cmd_ray} --object-store-memory=$OBJ_STORE_MEMORY"
	# cmd_promethius="prometheus --config.file=${DIRECTORY}/session_latest/metrics/prometheus/prometheus.yml"
	cmd_sleep="echo Sleeping ; sleep 10 ; echo Waking "
	# cmd_gafana="grafana-server --config ${DIRECTORY}/session_latest/metrics/grafana/grafana.ini web"
	SLURM="sbatch --job-name=RayHead --partition=smith_reserved "
	SLURM="${SLURM} --nodelist=${NODELIST} "
	SLURM="${SLURM} --mem=${MEMORY_GB}GB --gpus-per-node=${GPU_NUM}  "
	SLURM="${SLURM} --cpus-per-task=${CPU_NUM} -o ${DIRECTORY}/slurm.log "
	SLURM="${SLURM} /home/donkurid/runner.sh "
	SLURM="${SLURM}'${cmd_sleep}; ${cmd_ray}'"
	echo -e "\nSlurm command:\n"
	echo $SLURM
	echo ""
	eval $SLURM
}

HOSTNAME=$(uname -n)
case $HOSTNAME in
pollux)
	echo "Pollux does not have a script setup"
	;;
matrix*)
	echo "Starting ray head for matrix servers"
	start_head_on_matrix
	;;
*)
	echo "Unknown hostname: ${HOSTNAME}"
	;;
esac
