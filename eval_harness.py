import subprocess
import os
import re
import json
import csv
import hashlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import time

MAX_WALLCLOCK_SECONDS="300"
VAL_LOSS_EVERY="50"
TRAIN_LOG_EVERY="25"
FAST_VAL_TOKENS="5000000"

# CSV database file
RESULTS_CSV = "experiment_results.csv"
METRICS_CACHE_DIR = Path("runs/metrics_cache")

# --- Configuration ---
# 10L Int5-MLP + BigramHash(10240) SOTA and Baseline
EXPERIMENTS = {
    # # 10L Int5-MLP + BigramHash(10240): 1.1428 BPB
    "SOTA_Int5MLP_BigramHash": {
        "env": {
            "RUN_ID": "SOTA_Int5MLP_BigramHash_10min",
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": MAX_WALLCLOCK_SECONDS,
            "VAL_LOSS_EVERY": VAL_LOSS_EVERY,  # Periodic validation for monitoring
            # "FAST_VAL_TOKENS": FAST_VAL_TOKENS,
            "TRAIN_LOG_EVERY": TRAIN_LOG_EVERY,
            "EVAL_STRIDE": "1024", # No sliding window for now while testing (must match seq len)

            # --- 2. Shrink the Compute Payload (The Big Speedup) ---
            "TRAIN_BATCH_TOKENS": "524288", #  "131072", # "262144", # "524288", # Down from 786k (1/3 the tokens per step)
            "TRAIN_SEQ_LEN": "1024",        # Down from 2048 (4x faster attention)

            # --- 3. Trim the Architecture ---
            "NUM_LAYERS": "5",              # Down from 10
            "MLP_MULT": "2.0",              # Down from 3.0 (makes Int5 compression faster)
            
            # --- 4. Boost Weight Decay (Crucial for 16MB) ---
            # Since we are doing ~1200 steps instead of ~4000, we need to apply 
            # 3x more weight decay per step to hit the same sparsity target by the end.
            # "WEIGHT_DECAY": "0.12",         
            
            # --- FASTER LEARNING TWEAKS ---
            "MATRIX_LR": "0.03",              # Up from 0.02 (Bigger steps)
            "TIED_EMBED_LR": "0.05",          # Up from 0.03 (Learn embeddings faster)
            "BETA2": "0.90",                  # Down from 0.95 (Faster Adam adaptation)
            
            "MUON_BACKEND_STEPS": "3",        # Less overhead per step
            "MUON_MOMENTUM_WARMUP_STEPS": "300", # Ramp up momentum 5x faster!
            
            # --- CRITICAL: Settle the loss before the 10-min buzzer ---
            "WARMDOWN_ITERS": "250",          # Decay LR over the last ~2 minutes
            "SWA_ENABLED": "0",               # Disable for short evals
        },
        "cmd": ["torchrun", "--standalone", "--nproc_per_node=1",
                "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"],
        "description": "10L Int5-MLP + BigramHash(10240), SWA(0.4), WD=0.04 (Score: 1.1428)"
    },
    "SOTA_Int5MLP_BigramHash_with_multi_hash": {
        "env": {
            "RUN_ID": "SOTA_Int5MLP_BigramHash_with_multi_hash",
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": MAX_WALLCLOCK_SECONDS,
            "VAL_LOSS_EVERY": VAL_LOSS_EVERY,  # Periodic validation for monitoring
            # "FAST_VAL_TOKENS": FAST_VAL_TOKENS,
            "TRAIN_LOG_EVERY": TRAIN_LOG_EVERY,
            "EVAL_STRIDE": "1024", # No sliding window for now while testing (must match seq len)

            # --- 2. Shrink the Compute Payload (The Big Speedup) ---
            "TRAIN_BATCH_TOKENS": "524288", #  "131072", # "262144", # "524288", # Down from 786k (1/3 the tokens per step)
            "TRAIN_SEQ_LEN": "1024",        # Down from 2048 (4x faster attention)

            # --- 3. Trim the Architecture ---
            "NUM_LAYERS": "5",              # Down from 10
            "MLP_MULT": "2.0",              # Down from 3.0 (makes Int5 compression faster)
            # --- FASTER LEARNING TWEAKS ---
            "MATRIX_LR": "0.03",              # Up from 0.02 (Bigger steps)
            "TIED_EMBED_LR": "0.05",          # Up from 0.03 (Learn embeddings faster)
            "BETA2": "0.90",                  # Down from 0.95 (Faster Adam adaptation)
            
            "MUON_BACKEND_STEPS": "3",        # Less overhead per step
            "MUON_MOMENTUM_WARMUP_STEPS": "300", # Ramp up momentum 5x faster!
            
            # --- CRITICAL: Settle the loss before the 10-min buzzer ---
            "WARMDOWN_ITERS": "250",          # Decay LR over the last ~2 minutes
            "SWA_ENABLED": "0",              # Disable for short evals

            "MULTI_HASH_ENABLED": "1",
        },
        "cmd": ["torchrun", "--standalone", "--nproc_per_node=1",
                "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"],
        "description": "10L Int5-MLP + BigramHash(10240), SWA(0.4), WD=0.04 (Score: 1.1428)"
    },
    "SOTA_Int5MLP_BigramHash_with_multi_hash_layer6": {
        "env": {
            "RUN_ID": "SOTA_Int5MLP_BigramHash_with_multi_hash_layer6",
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": MAX_WALLCLOCK_SECONDS,
            "VAL_LOSS_EVERY": VAL_LOSS_EVERY,  # Periodic validation for monitoring
            # "FAST_VAL_TOKENS": FAST_VAL_TOKENS,
            "TRAIN_LOG_EVERY": TRAIN_LOG_EVERY,
            "EVAL_STRIDE": "1024", # No sliding window for now while testing (must match seq len)

            # --- 2. Shrink the Compute Payload (The Big Speedup) ---
            "TRAIN_BATCH_TOKENS": "524288", #  "131072", # "262144", # "524288", # Down from 786k (1/3 the tokens per step)
            "TRAIN_SEQ_LEN": "1024",        # Down from 2048 (4x faster attention)

            # --- 3. Trim the Architecture ---
            "NUM_LAYERS": "6",              # Down from 10
            "MLP_MULT": "2.0",              # Down from 3.0 (makes Int5 compression faster)
            # --- FASTER LEARNING TWEAKS ---
            "MATRIX_LR": "0.03",              # Up from 0.02 (Bigger steps)
            "TIED_EMBED_LR": "0.05",          # Up from 0.03 (Learn embeddings faster)
            "BETA2": "0.90",                  # Down from 0.95 (Faster Adam adaptation)
            
            "MUON_BACKEND_STEPS": "3",        # Less overhead per step
            "MUON_MOMENTUM_WARMUP_STEPS": "300", # Ramp up momentum 5x faster!
            
            # --- CRITICAL: Settle the loss before the 10-min buzzer ---
            "WARMDOWN_ITERS": "250",          # Decay LR over the last ~2 minutes
            "SWA_ENABLED": "0",              # Disable for short evals

            "MULTI_HASH_ENABLED": "1",
        },
        "cmd": ["torchrun", "--standalone", "--nproc_per_node=1",
                "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"],
        "description": "10L Int5-MLP + BigramHash(10240), SWA(0.4), WD=0.04 (Score: 1.1428)"
    },
    "SOTA_Int5MLP_BigramHash_with_multi_hash_less_batch_size": {
        "env": {
            "RUN_ID": "SOTA_Int5MLP_BigramHash_with_multi_hash_less_batch_size",
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": MAX_WALLCLOCK_SECONDS,
            "VAL_LOSS_EVERY": VAL_LOSS_EVERY,  # Periodic validation for monitoring
            # "FAST_VAL_TOKENS": FAST_VAL_TOKENS,
            "TRAIN_LOG_EVERY": TRAIN_LOG_EVERY,
            "EVAL_STRIDE": "1024", # No sliding window for now while testing (must match seq len)

            # --- 2. Shrink the Compute Payload (The Big Speedup) ---
            "TRAIN_BATCH_TOKENS": "131072", #  "131072", # "262144", # "524288", # Down from 786k (1/3 the tokens per step)
            "TRAIN_SEQ_LEN": "1024",        # Down from 2048 (4x faster attention)

            # --- 3. Trim the Architecture ---
            "NUM_LAYERS": "5",              # Down from 10
            "MLP_MULT": "2.0",              # Down from 3.0 (makes Int5 compression faster)
            # --- FASTER LEARNING TWEAKS ---
            "MATRIX_LR": "0.03",              # Up from 0.02 (Bigger steps)
            "TIED_EMBED_LR": "0.05",          # Up from 0.03 (Learn embeddings faster)
            "BETA2": "0.90",                  # Down from 0.95 (Faster Adam adaptation)
            
            "MUON_BACKEND_STEPS": "3",        # Less overhead per step
            "MUON_MOMENTUM_WARMUP_STEPS": "300", # Ramp up momentum 5x faster!
            
            # --- CRITICAL: Settle the loss before the 10-min buzzer ---
            "WARMDOWN_ITERS": "250",          # Decay LR over the last ~2 minutes
            "SWA_ENABLED": "0",              # Disable for short evals

            "MULTI_HASH_ENABLED": "1",
        },
        "cmd": ["torchrun", "--standalone", "--nproc_per_node=1",
                "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"],
        "description": "10L Int5-MLP + BigramHash(10240), SWA(0.4), WD=0.04 (Score: 1.1428)"
    },
    # Baseline for comparison
    "baseline_10min": {
    "env": {
        "RUN_ID": "baseline_10min",
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": MAX_WALLCLOCK_SECONDS,  # 10 minutes
        "VAL_LOSS_EVERY": VAL_LOSS_EVERY,  # Periodic validation for monitoring
        "TRAIN_LOG_EVERY": TRAIN_LOG_EVERY,
        # "FAST_VAL_TOKENS": FAST_VAL_TOKENS
    },
    "cmd": ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
    "description": "Baseline (Score: ~1.22)"
    },
}

# Updated regex patterns to match actual train_gpt.py output format
REGEX_TRAIN = re.compile(r"step:(\d+)/\d+ train_loss:([\d\.]+)")
REGEX_VAL = re.compile(r"step:(\d+)/\d+ val_loss:([\d\.]+) val_bpb:([\d\.]+)")
REGEX_FINAL_BPB = re.compile(r"final_int8_zlib_roundtrip\s+val_loss:([\d\.]+) val_bpb:([\d\.]+)")
REGEX_FINAL_INT6 = re.compile(r"final_int6.*?roundtrip\s+val_loss:([\d\.]+) val_bpb:([\d\.]+)")
REGEX_FINAL_EXACT = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d\.]+) val_bpb:([\d\.]+)")
REGEX_SIZE_BYTES = re.compile(r"Total submission size.*?:\s*([\d]+)\s*bytes")
REGEX_SLIDING = re.compile(r"final_sliding.*?val_loss:([\d\.]+) val_bpb:([\d\.]+)")
REGEX_TTT = re.compile(r"legal_ttt.*?val_loss:([\d\.]+) val_bpb:([\d\.]+)")

def get_git_commit():
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def compute_config_hash(config):
    """Compute deterministic hash of experiment config."""
    # Sort env vars for consistent hashing
    env_str = json.dumps(config["env"], sort_keys=True)
    cmd_str = " ".join(config["cmd"])
    combined = f"{env_str}|{cmd_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]

def load_existing_results():
    """Load existing results from CSV."""
    if not Path(RESULTS_CSV).exists():
        return {}
    
    results = {}
    with open(RESULTS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config_hash = row['config_hash']
            results[config_hash] = row
    return results

def get_metrics_cache_path(config_hash):
    """Return stable cache path for full metrics payload."""
    return METRICS_CACHE_DIR / f"{config_hash}.json"

def load_cached_metrics(config_hash):
    """Load full cached metrics for plotting skipped experiments."""
    cache_path = get_metrics_cache_path(config_hash)
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load cached metrics from {cache_path}: {e}")
        return None

def save_cached_metrics(config_hash, metrics):
    """Persist full metrics payload for future skipped runs."""
    try:
        METRICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = get_metrics_cache_path(config_hash)
        with open(cache_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Full metrics cache saved to {cache_path}")
    except Exception as e:
        print(f"⚠️  Failed to save cached metrics for {config_hash}: {e}")

def save_result_to_csv(exp_name, config, metrics, plot_filename, config_hash, run_time_seconds):
    """Append or update result in CSV database."""
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()
    
    # Flatten config for CSV columns
    row = {
        'timestamp': timestamp,
        'experiment_name': exp_name,
        'config_hash': config_hash,
        'git_commit': git_commit,
        'description': config.get('description', ''),
        'plot_filename': plot_filename,
        'run_time_seconds': f"{run_time_seconds:.1f}",
        
        # Results
        'final_bpb': metrics.get('final_bpb', ''),
        'final_bpb_exact': metrics.get('final_bpb_exact', ''),
        'final_loss': metrics.get('final_loss', ''),
        'final_size_mb': metrics.get('final_size_mb', ''),
        'sliding_bpb': metrics.get('sliding_bpb', ''),
        'ttt_bpb': metrics.get('ttt_bpb', ''),
        'max_train_steps': max(metrics['train_steps']) if metrics['train_steps'] else '',
        
        # Config parameters (flattened)
        **{f"env_{k}": v for k, v in config['env'].items()},
        'cmd': ' '.join(config['cmd'])
    }
    
    # Check if file exists to determine if we need headers
    file_exists = Path(RESULTS_CSV).exists()
    
    # Read existing data if file exists
    existing_rows = []
    fieldnames = list(row.keys())
    if file_exists:
        with open(RESULTS_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                # Merge fieldnames to handle new columns
                fieldnames = list(dict.fromkeys(list(reader.fieldnames) + fieldnames))
            existing_rows = [r for r in reader if r.get('config_hash') != config_hash]
    
    # Write back all data including new row
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
    
    print(f"💾 Result saved to {RESULTS_CSV}")

def run_experiment(name, config, force_run=False, timestamp=None):
    """Run experiment and track results."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = compute_config_hash(config)
    existing_results = load_existing_results()
    
    # Check if we can skip this run
    if not force_run and config_hash in existing_results:
        cached = existing_results[config_hash]
        print(f"\n{'='*70}")
        print(f"⏭️  SKIPPING: {name} (config unchanged)")
        print(f"   Config hash: {config_hash}")
        print(f"   Previous run: {cached.get('timestamp', 'unknown')}")
        print(f"   Previous BPB: {cached.get('final_bpb', 'N/A')}")
        print(f"   Use --force to re-run")
        print(f"{'='*70}")
        
        cached_metrics = load_cached_metrics(config_hash)
        if cached_metrics is not None:
            cached_metrics['cached'] = True
            cached_metrics['description'] = cached.get('description', cached_metrics.get('description', ''))
            return cached_metrics
        
        # Fallback to scalar-only cached metrics if no full time-series cache exists
        return {
            'final_bpb': float(cached['final_bpb']) if cached.get('final_bpb') else None,
            'final_bpb_exact': float(cached['final_bpb_exact']) if cached.get('final_bpb_exact') else None,
            'final_loss': float(cached['final_loss']) if cached.get('final_loss') else None,
            'final_size_mb': float(cached['final_size_mb']) if cached.get('final_size_mb') else None,
            'sliding_bpb': float(cached['sliding_bpb']) if cached.get('sliding_bpb') else None,
            'ttt_bpb': float(cached['ttt_bpb']) if cached.get('ttt_bpb') else None,
            'description': cached.get('description', ''),
            'cached': True,
            'train_steps': [], 'train_loss': [], 'train_times': [],
            'val_steps': [], 'val_loss': [], 'val_bpb': [], 'val_times': []
        }
    
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {name}")
    print(f"Config hash: {config_hash}")
    if "description" in config:
        print(f"Description: {config['description']}")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    
    # Merge current system env with experiment env
    env = os.environ.copy()
    env.update(config["env"])
    
    metrics = {
        "train_steps": [], "train_loss": [], "train_times": [],
        "val_steps": [], "val_loss": [], "val_bpb": [], "val_times": [],
        "final_loss": None, "final_bpb": None, "final_bpb_exact": None,
        "final_size_mb": None, "sliding_bpb": None, "ttt_bpb": None,
        "description": config.get("description", ""),
        "cached": False
    }
    
    # Create output directory for this run
    run_dir = Path(f"runs/{name}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to capture stdout/stderr
    stdout_file = run_dir / "stdout.log"
    stderr_file = run_dir / "stderr.log"
    combined_file = run_dir / "combined.log"
    
    # Save configuration
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump({
            "name": name,
            "config": config,
            "config_hash": config_hash,
            "timestamp": timestamp,
            "git_commit": get_git_commit()
        }, f, indent=2)
    print(f"📁 Run directory: {run_dir}")
    
    # Run the process and capture output
    try:
        process = subprocess.Popen(
            config["cmd"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd="/workspace/parameter-golf"
        )
        
        # Open log files
        with open(stdout_file, "w") as stdout_f, \
             open(stderr_file, "w") as stderr_f, \
             open(combined_file, "w") as combined_f:
            
            if process.stdout and process.stderr:
                import select
                
                # Track which streams are still open
                streams_to_read = [process.stdout, process.stderr]
                
                # Track wallclock time
                while streams_to_read:
                    # Use select to read from both stdout and stderr
                    readable, _, _ = select.select(streams_to_read, [], [], 0.1)
                    
                    if not readable:
                        if process.poll() is not None:
                            break
                        continue
                    
                    for stream in readable:
                        line = stream.readline()
                        if not line:
                            # Stream reached EOF, remove it from the list
                            streams_to_read.remove(stream)
                            continue
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Calculate elapsed time
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        
                        # Write to appropriate log files
                        if stream == process.stdout:
                            stdout_f.write(line + "\n")
                            stdout_f.flush()
                        else:
                            stderr_f.write(line + "\n")
                            stderr_f.flush()
                        
                        combined_f.write(f"[{elapsed_time:.2f}s] {line}\n")
                        combined_f.flush()
                        
                        # Print to console for real-time monitoring
                        print(f"[{name}] {line}")
                        
                        # Parse Train Loss
                        train_match = REGEX_TRAIN.search(line)
                        if train_match:
                            metrics["train_steps"].append(int(train_match.group(1)))
                            metrics["train_loss"].append(float(train_match.group(2)))
                            metrics["train_times"].append(elapsed_time)
                            
                        # Parse Val Loss & BPB
                        val_match = REGEX_VAL.search(line)
                        if val_match:
                            metrics["val_steps"].append(int(val_match.group(1)))
                            metrics["val_loss"].append(float(val_match.group(2)))
                            metrics["val_bpb"].append(float(val_match.group(3)))
                            metrics["val_times"].append(elapsed_time)
                            
                        # Parse Final BPB (int8 roundtrip)
                        final_match = REGEX_FINAL_BPB.search(line)
                        if final_match:
                            metrics["final_loss"] = float(final_match.group(1))
                            metrics["final_bpb"] = float(final_match.group(2))
                            
                        # Parse Final BPB (int6 roundtrip - some submissions use this)
                        int6_match = REGEX_FINAL_INT6.search(line)
                        if int6_match and not metrics["final_bpb"]:
                            metrics["final_loss"] = float(int6_match.group(1))
                            metrics["final_bpb"] = float(int6_match.group(2))
                            
                        # Parse Final BPB Exact
                        exact_match = REGEX_FINAL_EXACT.search(line)
                        if exact_match:
                            metrics["final_bpb_exact"] = float(exact_match.group(2))
                            
                        # Parse Artifact Size
                        size_match = REGEX_SIZE_BYTES.search(line)
                        if size_match:
                            metrics["final_size_mb"] = int(size_match.group(1)) / 1_000_000
                            
                        # Parse Sliding Window BPB
                        sliding_match = REGEX_SLIDING.search(line)
                        if sliding_match:
                            metrics["sliding_bpb"] = float(sliding_match.group(2))
                            
                        # Parse TTT BPB (test-time training)
                        ttt_match = REGEX_TTT.search(line)
                        if ttt_match:
                            metrics["ttt_bpb"] = float(ttt_match.group(2))

        process.wait()
        
        run_time = (datetime.now() - start_time).total_seconds()
        
        # Validation checks
        if metrics["final_size_mb"] and metrics["final_size_mb"] > 16.0:
            print(f"\n⚠️  WARNING: Artifact exceeds 16MB limit ({metrics['final_size_mb']:.2f} MB)")
        elif metrics["final_size_mb"]:
            print(f"\n✅ Artifact size OK: {metrics['final_size_mb']:.2f} MB")
        else:
            print(f"\n⚠️  No size info captured")
        
        # Save to JSON in run directory
        json_path = run_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Metrics saved to {json_path}")
        
        # Also save to legacy location for backward compatibility
        legacy_json_path = f"metrics_{name}.json"
        with open(legacy_json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Store metadata for CSV saving
        metrics['_config_hash'] = config_hash
        metrics['_run_time'] = run_time
        metrics['_run_dir'] = str(run_dir)
        
        save_cached_metrics(config_hash, metrics)
        
        print(f"📝 Logs saved:")
        print(f"   - stdout: {stdout_file}")
        print(f"   - stderr: {stderr_file}")
        print(f"   - combined: {combined_file}")
        print(f"   - config: {config_file}")
            
    except Exception as e:
        print(f"\n❌ Error running {name}: {e}")
        
    return metrics

def plot_metrics(results, timestamp):
    """Generate comparison plots and return filename."""
    # Create 2x2 subplot layout for all metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    has_data = False
    for name, data in results.items():
        if not data.get("train_steps") and not data.get("val_steps"):
            continue
        
        has_data = True
        
        # Plot 1: Loss vs Steps (original)
        ax1.plot(data["train_steps"], data["train_loss"], alpha=0.3, label=f"{name} (Train)")
        if data["val_steps"]:
            ax1.plot(data["val_steps"], data["val_loss"], marker='o', linewidth=2, label=f"{name} (Val)")
        
        # Plot 2: BPB vs Steps (original)
        if data["val_steps"]:
            ax2.plot(data["val_steps"], data["val_bpb"], marker='x', linewidth=2, label=f"{name}")
        
        # Plot 3: Loss vs Wallclock Time (NEW)
        if data.get("train_times"):
            ax3.plot(data["train_times"], data["train_loss"], alpha=0.3, label=f"{name} (Train)")
        if data.get("val_times"):
            ax3.plot(data["val_times"], data["val_loss"], marker='o', linewidth=2, label=f"{name} (Val)")
        
        # Plot 4: BPB vs Wallclock Time (NEW)
        if data.get("val_times") and data.get("val_bpb"):
            ax4.plot(data["val_times"], data["val_bpb"], marker='x', linewidth=2, label=f"{name}")

    if not has_data:
        plt.close()
        return ""

    # Configure Plot 1: Loss vs Steps
    ax1.set_title("Loss vs Steps")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configure Plot 2: BPB vs Steps
    ax2.set_title("Validation BPB vs Steps (Lower is Better)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Bits Per Byte")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Configure Plot 3: Loss vs Wallclock Time
    ax3.set_title("Loss vs Wallclock Time")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Configure Plot 4: BPB vs Wallclock Time
    ax4.set_title("Validation BPB vs Wallclock Time (Lower is Better)")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Bits Per Byte")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = f"eval_comparison_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to {plot_path}")
    plt.close()
    
    # Also save individual plots for each experiment in their run directories
    for name, data in results.items():
        if not data.get("train_steps") and not data.get("val_steps"):
            continue
        
        run_dir = Path(data.get('_run_dir', f"runs/{name}_{timestamp}"))
        
        # Create individual experiment plot
        fig_ind, ((ax1_ind, ax2_ind), (ax3_ind, ax4_ind)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Loss vs Steps
        ax1_ind.plot(data["train_steps"], data["train_loss"], alpha=0.5, label="Train Loss", color='blue')
        if data["val_steps"]:
            ax1_ind.plot(data["val_steps"], data["val_loss"], marker='o', linewidth=2, label="Val Loss", color='red')
        ax1_ind.set_title(f"{name}: Loss vs Steps")
        ax1_ind.set_xlabel("Steps")
        ax1_ind.set_ylabel("Loss")
        ax1_ind.legend()
        ax1_ind.grid(True, alpha=0.3)
        
        # Plot 2: BPB vs Steps
        if data["val_steps"]:
            ax2_ind.plot(data["val_steps"], data["val_bpb"], marker='x', linewidth=2, color='green')
        ax2_ind.set_title(f"{name}: Validation BPB vs Steps")
        ax2_ind.set_xlabel("Steps")
        ax2_ind.set_ylabel("Bits Per Byte")
        ax2_ind.grid(True, alpha=0.3)
        
        # Plot 3: Loss vs Wallclock Time
        if data.get("train_times"):
            ax3_ind.plot(data["train_times"], data["train_loss"], alpha=0.5, label="Train Loss", color='blue')
        if data.get("val_times"):
            ax3_ind.plot(data["val_times"], data["val_loss"], marker='o', linewidth=2, label="Val Loss", color='red')
        ax3_ind.set_title(f"{name}: Loss vs Wallclock Time")
        ax3_ind.set_xlabel("Time (seconds)")
        ax3_ind.set_ylabel("Loss")
        ax3_ind.legend()
        ax3_ind.grid(True, alpha=0.3)
        
        # Plot 4: BPB vs Wallclock Time
        if data.get("val_times") and data.get("val_bpb"):
            ax4_ind.plot(data["val_times"], data["val_bpb"], marker='x', linewidth=2, color='green')
        ax4_ind.set_title(f"{name}: Validation BPB vs Wallclock Time")
        ax4_ind.set_xlabel("Time (seconds)")
        ax4_ind.set_ylabel("Bits Per Byte")
        ax4_ind.grid(True, alpha=0.3)
        
        # Save individual plot
        individual_plot_path = run_dir / "plots.png"
        plt.tight_layout()
        plt.savefig(individual_plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Individual plot saved to {individual_plot_path}")
        plt.close()
    
    return plot_path

if __name__ == "__main__":
    import sys
    force_run = "--force" in sys.argv
    
    if force_run:
        print("🔄 Force mode enabled - will re-run all experiments")
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for exp_name, exp_config in EXPERIMENTS.items():
        metrics = run_experiment(exp_name, exp_config, force_run=force_run, timestamp=timestamp)
        all_results[exp_name] = metrics
    
    # Generate comparison plot
    plot_filename = ""
    plot_filename = plot_metrics(all_results, timestamp)
    if not plot_filename:
        print("\n📊 No plottable data found")
    
    # Save results to CSV
    for exp_name, metrics in all_results.items():
        if not metrics.get('cached'):
            save_result_to_csv(
                exp_name,
                EXPERIMENTS[exp_name],
                metrics,
                plot_filename,
                metrics.get('_config_hash', ''),
                metrics.get('_run_time', 0)
            )
    
    # Print final leaderboard summary
    print("\n" + "="*90)
    print("🏆 EVALUATION SUMMARY")
    print("="*90)
    print(f"{'Run':<30} | {'Final BPB':<10} | {'Size (MB)':<10} | {'Sliding':<10} | {'TTT':<10} | {'Status':<10}")
    print("-"*90)
    for name, data in all_results.items():
        final_bpb = f"{data['final_bpb']:.4f}" if data.get("final_bpb") else "N/A"
        size = f"{data['final_size_mb']:.2f}" if data.get("final_size_mb") else "N/A"
        sliding = f"{data['sliding_bpb']:.4f}" if data.get("sliding_bpb") else "N/A"
        ttt = f"{data['ttt_bpb']:.4f}" if data.get("ttt_bpb") else "N/A"
        status = "CACHED" if data.get('cached') else "NEW"
        print(f"{name:<30} | {final_bpb:<10} | {size:<10} | {sliding:<10} | {ttt:<10} | {status:<10}")
        if data.get("description"):
            print(f"  └─ {data['description']}")
    print("="*90)
    print(f"\n📊 Results database: {RESULTS_CSV}")
    if plot_filename:
        print(f"📈 Latest plot: {plot_filename}")

# Made with Bob
