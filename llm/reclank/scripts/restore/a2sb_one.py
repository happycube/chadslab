"""A2SB bandwidth extension of one file (what inference/A2SB_upsample_api.py does, with its configs written to /tmp
instead of the read-only repo): the 99 % spectral rolloff as the cutoff, the two-split ensemble, 50 steps.
    python a2sb_one.py IN.wav OUT.wav [N_STEPS]"""
import os, sys
import numpy as np, yaml, librosa

REPO, CK = "/repo", "/models/ckpt"
src, dst = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
ens = yaml.safe_load(open(os.path.join(REPO, "configs/ensemble_2split_sampling.yaml")))
ens["trainer"]["strategy"] = "auto"
ens["trainer"]["accelerator"] = "gpu" if os.environ.get("DEV", "cuda") == "cuda" else "cpu"
ens["trainer"]["plugins"] = None
ens["model"]["pretrained_checkpoints"] = [CK + "/A2SB_twosplit_0.0_0.5_release.ckpt", CK + "/A2SB_twosplit_0.5_1.0_release.ckpt"]
ens["data"]["num_workers"] = 0
inf = yaml.safe_load(open(os.path.join(REPO, "configs/inference_files_upsampling.yaml")))
inf["data"]["predict_filelist"] = [{"filepath": src, "output_subdir": "."}]
y, sr = librosa.load(src, sr=None)
cut = int(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)[0]))
print("cutoff (99 %% rolloff): %d Hz" % cut, flush=True)
inf["data"]["transforms_aug"][0]["init_args"]["upsample_mask_kwargs"] = {"min_cutoff_freq": cut, "max_cutoff_freq": cut}
os.makedirs("/tmp/a2sb", exist_ok=True)
yaml.safe_dump(ens, open("/tmp/a2sb/ens.yaml", "w")); yaml.safe_dump(inf, open("/tmp/a2sb/inf.yaml", "w"))
os.makedirs("/tmp/a2sb/run", exist_ok=True)
os.chdir("/tmp/a2sb/run")   # it writes logs to ../debug
sys.path.insert(0, REPO)
sys.path.insert(0, "/shims")
sys.argv = ["/drv/safe_run.py", os.path.join(REPO, "ensembled_inference_api.py"), "predict", "-c", "/tmp/a2sb/ens.yaml",
            "-c", "/tmp/a2sb/inf.yaml", "--checkpoint_callback.dirpath=/tmp/a2sb/debug", "--model.predict_n_steps=%d" % steps, "--model.predict_batch_size=%s" % os.environ.get("A2SB_BATCH", "2"), "--model.output_audio_filename=%s" % dst]
exec(open("/drv/safe_run.py").read())
