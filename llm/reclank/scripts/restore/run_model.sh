#!/bin/bash
# Inside the sandbox: one model over every /in/*.wav (or the files named), outputs in /out/MODEL/NAME.wav, a line
# of timing each.   run_model.sh MODEL [NAME ...]
set -u
m=$1; shift
names=${*:-$(cd /in && ls *.wav | sed 's/\.wav$//')}
mkdir -p /out/$m
for n in $names; do
  [ -s /out/$m/$n.wav ] && { echo "$m $n done already"; continue; }
  t0=$(date +%s.%N)
  case $m in
    restore*)
      PYTHONPATH=/repo SAFE_GLOBALS=demucs.htdemucs:HTDemucs,fractions:Fraction python /drv/safe_run.py /repo/infer.py --checkpoint /repo/checkpoint.th --input /in/$n.wav \
        --output-dir /out/$m/$n.stems --device ${DEV:-cuda} ${RESTORE_ARGS:-} && cp /out/$m/$n.stems/restored.wav /out/$m/$n.wav ;;
    apollo)
      PYTHONPATH=/repo SAFE_GLOBALS=omegaconf.nodes:AnyNode,omegaconf.base:ContainerMetadata,omegaconf.base:Metadata,omegaconf.listconfig:ListConfig,omegaconf.dictconfig:DictConfig,torch.torch_version:TorchVersion,collections:defaultdict,typing:Any,builtins:list,builtins:int,builtins:dict \
        python /drv/safe_run.py /repo/inference.py --in_wav /in/$n.wav --out_wav /out/$m/$n.wav \
        --checkpoint /models/pytorch_model.bin --device ${DEV:-cuda} --chunk-seconds 10 --chunk-pad-seconds 1 ;;
    a2sb)
      python /drv/a2sb_one.py /in/$n.wav /out/$m/$n.wav ;;
    audiosr)
      HF_HOME=/hf python /drv/audiosr_one.py /in/$n.wav /out/$m/$n.wav ;;
  esac
  echo "$m $n rc=$? $(python -c "import time; print('%.0f s' % (time.time() - $t0))")"
done
