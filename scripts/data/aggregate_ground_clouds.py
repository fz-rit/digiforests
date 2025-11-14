# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
from pathlib import Path

# Add src directory to path so digiforests_dataloader can be imported
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import typer
from typing import List
from digiforests_dataloader.utils.logging import logger

# Import the aggregate function from the other script
from aggregate_clouds_and_labels import aggregate as single_aggregate

app = typer.Typer(rich_markup_mode="markdown")


def find_ground_experiment_folders(base_path: Path) -> List[Path]:
    """
    Find all experiment folders within digiforests-ground-* directories.
    
    Expected structure:
    digiforests-ground-XX/raw/{train,val}/YYYY-MM/expNN-XX/
    
    Returns:
        List of paths to experiment folders (deepest level containing ground_clouds/)
    """
    experiment_folders = []
    
    # Find all digiforests-ground-* directories
    ground_dirs = sorted(base_path.glob("digiforests-ground-*"))
    
    for ground_dir in ground_dirs:
        # Look for experiment folders (containing ground_clouds directory)
        for exp_folder in ground_dir.rglob("*"):
            if exp_folder.is_dir() and (exp_folder / "ground_clouds").exists():
                # Verify it also has poses.txt and labels/
                if (exp_folder / "poses.txt").exists() and (exp_folder / "labels").exists():
                    experiment_folders.append(exp_folder)
                    logger.info(f"Found experiment folder: {exp_folder}")
                else:
                    logger.warning(f"Skipping {exp_folder} - missing poses.txt or labels/")
    
    return sorted(experiment_folders)


@app.command()
def batch_aggregate(
    base_path: Path = typer.Argument(
        ..., 
        help="Base path containing digiforests-ground-* folders (e.g., /path/to/DigiForests/)"
    ),
    output_folder: Path = typer.Argument(
        ..., 
        help="Path to the output folder where all aggregated clouds will be saved."
    ),
    denoise: bool = typer.Option(
        True, 
        help="Apply denoising to the aggregated point clouds if set to True."
    ),
    voxel_down_sample_size: float = typer.Option(
        0.01, 
        help="Voxel down-sampling size for denoising the point cloud."
    ),
    dry_run: bool = typer.Option(
        False,
        help="If True, only list the folders that would be processed without running aggregation."
    ),
):
    """
    Batch process all digiforests-ground experiment folders.
    
    This script automatically discovers all ground-based experiment folders
    within the specified base path and processes them in batch.
    
    \n\n**Workflow:**\n
    1. Scan base_path for all digiforests-ground-* directories\n
    2. Identify experiment folders (containing ground_clouds/, poses.txt, labels/)\n
    3. Process each folder using the aggregate function\n
    4. Generate descriptive filenames based on folder structure
    
    \n\n**Example:**\n
    ```bash
    python aggregate_ground_clouds.py /path/to/DigiForests /path/to/output
    ```
    
    This will process folders like:
    - digiforests-ground-c1/raw/val/2024-07/exp11-c1/ → grd_c1_val_2024_07_exp11_c1_aggr.ply
    - digiforests-ground-d2/raw/train/2023-03/exp20-d2/ → grd_d2_train_2023_03_exp20_d2_aggr.ply
    """
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment folders
    logger.info(f"Scanning {base_path} for ground experiment folders...")
    experiment_folders = find_ground_experiment_folders(base_path)
    
    if not experiment_folders:
        logger.error(f"No experiment folders found in {base_path}")
        raise typer.Exit(code=1)
    
    logger.info(f"Found {len(experiment_folders)} experiment folder(s) to process")
    
    if dry_run:
        logger.info("DRY RUN - Would process the following folders:")
        for folder in experiment_folders:
            logger.info(f"  - {folder}")
        return
    
    # Process each folder
    successful = 0
    failed = 0
    
    for i, exp_folder in enumerate(experiment_folders, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(experiment_folders)}: {exp_folder}")
        logger.info(f"{'='*80}")
        
        try:
            # Call the single aggregation function
            single_aggregate(
                exp_folder=exp_folder,
                output_folder=output_folder,
                denoise=denoise,
                voxel_down_sample_size=voxel_down_sample_size,
                output_filename=None  # Auto-generate from path
            )
            successful += 1
            logger.info(f"✓ Successfully processed {exp_folder.name}")
            
        except Exception as e:
            failed += 1
            logger.error(f"✗ Failed to process {exp_folder}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total folders: {len(experiment_folders)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {output_folder}")


if __name__ == "__main__":
    app()
