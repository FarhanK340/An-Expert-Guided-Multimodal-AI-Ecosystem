@echo off
:: ============================================================
:: run_fusion_training.bat
:: Trains the MoME+ fusion / gating network.
::
:: Prerequisites:
::   - All 4 expert checkpoints in experiments\checkpoints\experts\
::       expert_T1_best.pth, expert_T1ce_best.pth,
::       expert_T2_best.pth, expert_FLAIR_best.pth
::   - Per-modality H5 files in ..\data\preprocessed\
::       brats2024_gli_T1_train.h5  etc.
:: ============================================================

cd /d %~dp0

echo ============================================================
echo  MoME+  Fusion Network Training
echo ============================================================

python -m src.training.train_fusion ^
    --expert_t1    experiments/checkpoints/experts/expert_T1_best.pth ^
    --expert_t1ce  experiments/checkpoints/experts/expert_T1ce_best.pth ^
    --expert_t2    experiments/checkpoints/experts/expert_T2_best.pth ^
    --expert_flair experiments/checkpoints/experts/expert_FLAIR_best.pth ^
    --t1_train     ../data/preprocessed/brats2024_gli_T1_train.h5 ^
    --t1ce_train   ../data/preprocessed/brats2024_gli_T1ce_train.h5 ^
    --t2_train     ../data/preprocessed/brats2024_gli_T2_train.h5 ^
    --flair_train  ../data/preprocessed/brats2024_gli_FLAIR_train.h5 ^
    --t1_val       ../data/preprocessed/brats2024_gli_T1_val.h5 ^
    --t1ce_val     ../data/preprocessed/brats2024_gli_T1ce_val.h5 ^
    --t2_val       ../data/preprocessed/brats2024_gli_T2_val.h5 ^
    --flair_val    ../data/preprocessed/brats2024_gli_FLAIR_val.h5 ^
    --epochs       50 ^
    --batch_size   4 ^
    --lr           1e-4 ^
    --weight_decay 1e-4 ^
    --patience     15 ^
    --val_freq     5 ^
    --num_workers  0 ^
    --output_dir   experiments/checkpoints

echo.
echo Training finished.
pause
