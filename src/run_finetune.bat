@echo off
REM GOPT Fine-tuning Script (Windows)
REM Usage: run_finetune.bat

echo Starting GOPT Fine-tuning...

REM ============================================================
REM Configuration
REM ============================================================

REM Learning rate (lower for fine-tuning)
set lr=5e-4

REM Model architecture (MUST match pretrained model)
set depth=3
set head=1
set embed_dim=24

REM Training parameters
set batch_size=25
set n_epochs=50
set model=gopt
set am=librispeech

REM Loss weights
set loss_w_phn=1.0
set loss_w_word=1.0
set loss_w_utt=1.0

REM Data augmentation
set noise=0.3

REM Pretrained model path
set pretrained=..\pretrained_models\gopt_librispeech\best_audio_model.pth

REM Output directory
set exp_dir=..\exp\finetune-lr%lr%-%am%-noise%noise%

REM ============================================================
REM Advanced Options
REM ============================================================

REM Freezing strategies (uncomment to enable)
REM set freeze_blocks=--freeze_blocks
REM set freeze_embeddings=--freeze_embeddings
REM set progressive_unfreeze=--progressive_unfreeze

REM Early stopping (0 to disable)
set early_stopping=15

REM Learning rate scheduler
set lr_scheduler=multistep

REM ============================================================
REM Run Training
REM ============================================================

if not exist "%exp_dir%" mkdir "%exp_dir%"

python traintest_finetune.py ^
  --lr %lr% ^
  --exp-dir %exp_dir% ^
  --pretrained %pretrained% ^
  --goptdepth %depth% ^
  --goptheads %head% ^
  --batch_size %batch_size% ^
  --embed_dim %embed_dim% ^
  --n-epochs %n_epochs% ^
  --model %model% ^
  --am %am% ^
  --loss_w_phn %loss_w_phn% ^
  --loss_w_word %loss_w_word% ^
  --loss_w_utt %loss_w_utt% ^
  --noise %noise% ^
  --early_stopping %early_stopping% ^
  --lr_scheduler %lr_scheduler% ^
  %freeze_blocks% ^
  %freeze_embeddings% ^
  %progressive_unfreeze%

echo.
echo Fine-tuning completed!
echo Results saved to: %exp_dir%
echo Best model: %exp_dir%\models\best_audio_model.pth
echo Training log: %exp_dir%\result.csv
echo.
pause

