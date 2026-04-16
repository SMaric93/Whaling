#!/bin/bash
# Monitor HPCC job 5644165 every 30 minutes
JOB_ID=5644165
LOG="/Users/stefanmaric/Papers/Whaling/scripts/hpcc_monitor.log"
REMOTE="maricste@hpcc.msu.edu"
REMOTE_DIR="/mnt/research/CEO_Complementarities/maricste"

echo "$(date): Starting monitor for job ${JOB_ID}" | tee -a "$LOG"

while true; do
    STATE=$(ssh -o ConnectTimeout=10 $REMOTE "squeue -j $JOB_ID --noheader --format='%T' 2>/dev/null" 2>/dev/null)
    
    if [ -z "$STATE" ]; then
        # Job not in queue anymore — check if it completed
        RESULT=$(ssh -o ConnectTimeout=10 $REMOTE "sacct -j $JOB_ID --format=State,ExitCode,Elapsed --noheader 2>/dev/null | head -1" 2>/dev/null)
        echo "$(date): Job $JOB_ID LEFT QUEUE. sacct: $RESULT" | tee -a "$LOG"
        
        # Check for output files
        ssh -o ConnectTimeout=10 $REMOTE "echo '=== V3 OUTPUT ==='; ls -lh ${REMOTE_DIR}/data/extracted/wsl_events_1894.jsonl ${REMOTE_DIR}/data/extracted/wsl_manifest_1894.jsonl ${REMOTE_DIR}/data/extracted/extraction_v3_1894.log 2>/dev/null; echo '=== SLURM OUT (last 40 lines) ==='; tail -40 ${REMOTE_DIR}/wsl_v3-${JOB_ID}*.SLURMout 2>/dev/null || echo 'No SLURM output'" | tee -a "$LOG"
        
        echo "$(date): Monitor complete." | tee -a "$LOG"
        break
    else
        echo "$(date): Job $JOB_ID state=$STATE" | tee -a "$LOG"
    fi
    
    sleep 1800  # 30 minutes
done
