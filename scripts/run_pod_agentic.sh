#!/bin/bash
# Drive the agentic reconstruction loop on the Isaac Sim pod.
#
# - Waits for any pod under the deployment to be Running+Ready (re-resolves
#   the pod name on each iteration so deployment recreates don't hang us).
# - Syncs git HEAD into the repo on the pod's PVC (only tracked files).
# - Pipes the OpenAI key from the `openai-key` k8s secret into the pod over
#   stdin (never lands on local disk).
# - Idempotently installs the bundled-python deps that Isaac's image lacks.
# - Runs reconstruct_scene.py with --agent agentic --enable-render-tool.
# - Moves prediction.json/usd, agentic_trace.json, agentic_run.log into a
#   versioned dir on the pod, then `kubectl cp`s back to local.
# - Optionally pulls the mid-loop render snapshots from /tmp on the pod.
#
# Usage:   RUNDIR=agentic_run_v9 bash scripts/run_pod_agentic.sh
# Default: RUNDIR=agentic_run_v9 (bump per version so prior runs are preserved).

set -uo pipefail

NS=nsf-maica
LABEL=k8s-app=yizhan-isaacsim-dep
SAMPLE=samples/u_conveyor_default
RUNDIR=${RUNDIR:-agentic_run_v9}
LOCAL_REPO=${LOCAL_REPO:-/mnt/d/dev/Isaac_Sim_Scene}

step() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. wait for any pod under the deployment to be Running+Ready
step "waiting for $LABEL Running+Ready..."
while true; do
  POD=$(kubectl get pod -n "$NS" -l "$LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$POD" ]; then
    PHASE=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
    READY=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
    [ "$PHASE" = "Running" ] && [ "$READY" = "true" ] && { step "pod=$POD"; break; }
  fi
  sleep 30
done

# 2. locate repo on the PVC
REPO=$(kubectl exec -n "$NS" "$POD" -- bash -c '
  for p in /workspace/Isaac_Sim_Scene /workspace/dev/Isaac_Sim_Scene /workspace; do
    [ -d "$p/src/isaacsim_bench" ] && echo "$p" && exit 0
  done
  echo MISSING' 2>/dev/null)
[ "$REPO" = "MISSING" ] && { step "ERROR: repo missing on PVC"; exit 2; }
step "repo=$REPO"

# 3. sync git HEAD into the repo
step "syncing git-tracked files..."
( cd "$LOCAL_REPO" && git ls-files | tar -cf - -T - ) | \
  kubectl exec -i -n "$NS" "$POD" -- bash -c "mkdir -p $REPO && cd $REPO && tar -xf -"

# 4. pipe OpenAI key from the k8s secret into the pod (never on local disk)
step "writing /workspace/.openai_key..."
kubectl get secret openai-key -n "$NS" -o jsonpath='{.data.OPENAI_API_KEY}' | base64 -d | \
  kubectl exec -i -n "$NS" "$POD" -- bash -c 'umask 077; cat > /workspace/.openai_key'

# 5. install bundled-python deps (idempotent)
step "ensuring openai/open-clip-torch/click in /isaac-sim/python.sh..."
kubectl exec -n "$NS" "$POD" -- /isaac-sim/python.sh -m pip install -q \
  "openai==1.99.9" open-clip-torch click 2>&1 | tail -3 || true

# 6. run agentic with render
step "launching agentic loop -> $SAMPLE/$RUNDIR..."
kubectl exec -n "$NS" "$POD" -- bash -c "
  set +e
  cd $REPO
  export OPENAI_API_KEY=\$(cat /workspace/.openai_key)
  rm -rf $SAMPLE/$RUNDIR
  rm -f $SAMPLE/prediction.json $SAMPLE/prediction.usd $SAMPLE/agentic_trace.json $SAMPLE/agentic_run.log
  /isaac-sim/python.sh scripts/reconstruct_scene.py $SAMPLE/renders \
    --agent agentic --enable-render-tool --evaluate \
    >$SAMPLE/agentic_run.log 2>&1
  echo \"  agentic exit=\$?\"
  mkdir -p $SAMPLE/$RUNDIR
  for f in prediction.json prediction.usd agentic_trace.json agentic_run.log; do
    [ -f $SAMPLE/\$f ] && mv $SAMPLE/\$f $SAMPLE/$RUNDIR/
  done
"
step "agentic loop returned"

# 7. pull artifacts back
step "pulling artifacts to $LOCAL_REPO/$SAMPLE/$RUNDIR/..."
mkdir -p "$LOCAL_REPO/$SAMPLE/$RUNDIR"
kubectl cp "$NS/$POD:$REPO/$SAMPLE/$RUNDIR/." "$LOCAL_REPO/$SAMPLE/$RUNDIR/"

# 8. (optional) pull mid-loop render snapshots
step "pulling mid-loop render snapshots..."
for d in $(kubectl exec -n "$NS" "$POD" -- ls /tmp 2>/dev/null | grep agentic_render || true); do
  mkdir -p "$LOCAL_REPO/$SAMPLE/$RUNDIR/renders/$d"
  kubectl cp "$NS/$POD:/tmp/$d/." \
    "$LOCAL_REPO/$SAMPLE/$RUNDIR/renders/$d/" 2>/dev/null
done

step "DONE. local artifacts:"
ls -la "$LOCAL_REPO/$SAMPLE/$RUNDIR/"
