# Golden arguments of the paper's runs

`tests/test_paper_scripts.py` checks that `scripts/` runs these exact configurations.

- `train/<run>.json`: the parsed `train.py` arguments of each training run in `scripts/train_*.sh`.
  They come from the launcher that trained the paper's models (`scripts/mplc_v2_train.sh`, removed
  after commit `399c2bc`). Where the run directory was kept, its `info.log` recorded the same
  arguments, except for the settings that change where results are written, which are ignored.
  There are two exceptions. The clean twin (`supervlad_clean_s0`) and the closest-positive run
  (`supervlad_araft_closest_s0`) were trained with validation every 2 epochs, patience 8 and
  lr-plateau patience 5. The closest-positive run switched to 3 / 6 / 3 when it was resumed at epoch 60.
  The golden files and the scripts use the paper's 3 / 6 / 3 for both.
- `eval/<model>__<attack>__<dataset>__<tag>.json`: the `eval.py` arguments recorded in the
  `info.log` of the evaluation behind each reported number (`recorded_in` lists the directories).
  Model paths are given in the `logs/paper/<run>` naming. Plain AT under rank-APGD was evaluated in
  two runs, at 4/255 and later at 2/255; its `epsilons` are the union of the two.

`--selection_clean_budgets` is ignored. It only decides which extra `best_model_budget<b>.pth`
files are saved. The scripts keep budgets 1, 3, 5, 7 and 10 for every run.
