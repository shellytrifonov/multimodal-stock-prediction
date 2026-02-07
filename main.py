import subprocess
import sys
import time

def print_banner(text):
    """Print a large banner."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")

def run_pipeline(script_name, description):
    """Run a pipeline script and handle errors."""
    print_banner(f"STAGE: {description}")
    print(f"Executing: python {script_name}")
    print("-" * 70)
    
    start_time = time.time()
    result = subprocess.run(f"python {script_name}", shell=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print("\n" + "=" * 70)
        print(f"ERROR: {description} FAILED")
        print("=" * 70)
        print(f"\nPipeline stopped at: {script_name}")
        print(f"Return code: {result.returncode}")
        print("\nPlease check the error messages above and fix any issues.")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully in {elapsed_time:.1f} seconds")
    return elapsed_time

if __name__ == "__main__":
    print_banner("STARTING FULL SYSTEM PIPELINE")
    print("This will execute all pipelines in sequence:")
    print("  1. Stock Data Processing & Training")
    print("  2. Twitter Sentiment Analysis & Training")
    print("  3. Hybrid Fusion Model Training")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
        sys.exit(0)
    
    total_start_time = time.time()
    
    # Pipeline stages
    stages = [
        ("scripts/run_stock_pipeline.py", "Stock Pipeline"),
        ("scripts/run_twitter_pipeline.py", "Twitter Pipeline"),
        ("scripts/run_hybrid_pipeline.py", "Hybrid Fusion Pipeline")
    ]
    
    stage_times = []
    
    for script, description in stages:
        elapsed = run_pipeline(script, description)
        stage_times.append((description, elapsed))
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print_banner("SYSTEM TRAINED SUCCESSFULLY")
    
    print("Pipeline Execution Summary:")
    print("-" * 70)
    for description, elapsed in stage_times:
        print(f"  {description:<40} {elapsed:>8.1f}s")
    print("-" * 70)
    print(f"  {'Total Time':<40} {total_time:>8.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 70)
    
    print("\nTrained Models:")
    print("  • models/trained/stock_lstm_trained.pth")
    print("  • models/trained/twitter_lstm_trained.pth")
    print("  • models/trained/hybrid_fusion_trained.pth")
    
    print("\nYour hybrid stock prediction system is ready!")
    print("Use the fusion model for production predictions.")
    print("=" * 70 + "\n")
