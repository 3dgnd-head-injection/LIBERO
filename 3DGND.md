# LIBERO for 3DGnd

This fork of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is the version used to
generate the training data and to run the LIBERO / LIBERO-PRO evaluation of *Direct Action-Head
Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization* (3DGnd, CoRL 2026,
[project page](https://3dgnd-head-injection.github.io)). The 3DGnd code includes it as the git
submodule `external_dependencies/LIBERO`.

The fork is upstream LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` plus the changes
below (`git log 8f1084e..HEAD` lists their commits):

1. [Sub-goal switch](#sub-goal-switch): `ObjectState.grasped()`.
2. [Placement targets of 19 LIBERO-90 tasks](#placement-targets-of-19-libero-90-tasks) in their
   task files.
3. [LIBERO-PRO suites](#libero-pro-suites): four suites of LIBERO-PRO with their 40 task files.
4. [Compatibility](#compatibility): `torch.load(..., weights_only=False)` for the init states and
   an empty `libero/__init__.py`.

Apart from these changes, this file, a note at the top of `README.md` and the license of the
LIBERO-PRO task files (`licenses/LIBERO-PRO.txt`), the fork is identical to upstream; in particular,
the scenes, goals and init states of the LIBERO tasks are unchanged.

## Sub-goal switch

`ObjectState.grasped()` in `libero/libero/envs/object_states/base_object_states.py` selects the
sub-goal of a pick-and-place task: the 3DGnd grounding target is the placement target while
`grasped()` of the object to pick returns True, otherwise the object to pick (both are read from
`:obj_of_interest`, see below). `grasped()` returns False for its first 12 calls, records the
object's height at the 12th call and then returns whether the object is more than 1 cm above that
height (not latched). It counts its calls: the 3DGnd data generation calls it once per recorded
frame (dropped no-op steps are not counted), the evaluation once per environment observation. The
code is unchanged from the version that produced the dataset and the paper results.

## Placement targets of 19 LIBERO-90 tasks

3DGnd reads `:obj_of_interest` as (object to pick, placement target). In 19 LIBERO-90 tasks of the
training data the placement target named the whole fixture or reference object; the edits name the
region the object has to go to instead, as the LIBERO-Goal task files already do (e.g.
`flat_stove_1_cook_region`). For `KITCHEN_SCENE6` the object to pick is also listed first. Only
`:obj_of_interest` changes, not the scene or the goal. The edits are required to regenerate the
training data; the evaluation suites do not use them.

| Task (`libero/libero/bddl_files/libero_90/<task>.bddl`) | Upstream `:obj_of_interest` | 3DGnd |
|---|---|---|
| `KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet` | `akita_black_bowl_1` `wooden_cabinet_1` | `akita_black_bowl_1` `wooden_cabinet_1_top_side` |
| `KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet` | `akita_black_bowl_2` `wooden_cabinet_1` | `akita_black_bowl_2` `wooden_cabinet_1_top_side` |
| `KITCHEN_SCENE3_put_the_frying_pan_on_the_stove` | `chefmate_8_frypan_1` `flat_stove_1` | `chefmate_8_frypan_1` `flat_stove_1_cook_region` |
| `KITCHEN_SCENE3_put_the_moka_pot_on_the_stove` | `moka_pot_1` `flat_stove_1` | `moka_pot_1` `flat_stove_1_cook_region` |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet` | `akita_black_bowl_1` `white_cabinet_1` | `akita_black_bowl_1` `white_cabinet_1_bottom_region` |
| `KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet` | `akita_black_bowl_1` `white_cabinet_1` | `akita_black_bowl_1` `white_cabinet_1_top_side` |
| `KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet` | `wine_bottle_1` `white_cabinet_1` | `wine_bottle_1` `white_cabinet_1_bottom_region` |
| `KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | `akita_black_bowl_1` `white_cabinet_1` | `akita_black_bowl_1` `white_cabinet_1_top_region` |
| `KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet` | `akita_black_bowl_1` `white_cabinet_1` | `akita_black_bowl_1` `white_cabinet_1_top_side` |
| `KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet` | `ketchup_1` `white_cabinet_1` | `ketchup_1` `white_cabinet_1_top_region` |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug` | `porcelain_mug_1` `white_yellow_mug_1` | `white_yellow_mug_1` `kitchen_table_porcelain_mug_front_region` |
| `KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate` | `white_bowl_1` `plate_1` | `white_bowl_1` `kitchen_table_plate_right_region` |
| `KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove` | `moka_pot_1` `flat_stove_1` | `moka_pot_1` `flat_stove_1_cook_region` |
| `KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet` | `chefmate_8_frypan_1` `wooden_two_layer_shelf_1` | `chefmate_8_frypan_1` `wooden_two_layer_shelf_1_top_side` |
| `KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet` | `white_bowl_1` `wooden_two_layer_shelf_1` | `white_bowl_1` `wooden_two_layer_shelf_1_top_side` |
| `KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | `akita_black_bowl_1` `wooden_cabinet_1` | `akita_black_bowl_1` `wooden_cabinet_1_top_region` |
| `LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate` | `chocolate_pudding_1` `plate_1` | `chocolate_pudding_1` `living_room_table_plate_left_region` |
| `LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate` | `chocolate_pudding_1` `plate_1` | `chocolate_pudding_1` `living_room_table_plate_right_region` |
| `STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` | `yellow_book_2` `wooden_two_layer_shelf_1` | `yellow_book_2` `wooden_two_layer_shelf_1_top_side` |

## LIBERO-PRO suites

`libero/libero/benchmark/` registers four suites of
[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) like the LIBERO suites, e.g.
`benchmark.get_benchmark_dict()["libero_pro_object_task"]()`:

| Suite | Perturbation | Tasks |
|---|---|---|
| `libero_pro_object_task` | task | the 10 LIBERO-Object tasks |
| `libero_pro_spatial_task` | task | the 10 LIBERO-Spatial tasks |
| `libero_pro_object_swap` | position | the 10 LIBERO-Object tasks |
| `libero_pro_spatial_swap` | position | the 10 LIBERO-Spatial tasks |

Task names are `<suite>_<LIBERO task>`, e.g.
`libero_pro_object_task_pick_up_the_alphabet_soup_and_place_it_in_the_basket`, and task files
`libero/libero/bddl_files/<suite>/<task name>.bddl`. The instruction of a task is the `:language` of
its task file (`env.language_instruction`), which the 3DGnd evaluation uses. `task.language` is
derived from the task name as in LIBERO, so it is the instruction of the original LIBERO task; in
the task-perturbation suites that is a different task.

The task files come from the LIBERO-PRO dataset on Hugging Face
([`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro), revision
`3a33dc4cfbea7ff5eac6764e9c557500972b6259`): its files
`bddl_files/libero_{object,spatial}_{task,swap}/<LIBERO task>.bddl`, renamed to the task names above
and edited as follows (used for all evaluations in the paper):

- `object_task`, `object_swap`: `(:language Pick the ...)` becomes `(:language pick up the ...)`.
- `object_task`: the placement target in `:obj_of_interest` is `basket_1` instead of
  `basket_1_contain_region` (the goal is unchanged).
- `spatial_task`, `..._between_the_plate_and_the_ramekin_...`: `plate_1` is added to
  `:obj_of_interest`, which otherwise lists only the bowl.
- `spatial_swap`: unchanged.

The LIBERO-PRO init-state files are not included, so `Benchmark.get_task_init_states()` does not
work for these suites. The 3DGnd evaluation does not need them: it samples the object placements
from the regions of the task file at every reset.

## Compatibility

- `Benchmark.get_task_init_states()` loads the init states with
  `torch.load(..., weights_only=False)`. The files hold NumPy arrays, which `torch.load` refuses by
  default since torch 2.6; the argument requires torch 1.13 or later.
- An empty `libero/__init__.py` makes `libero` a regular package, so that `import libero` also
  works from any directory after a regular (non-compat) editable install (`pip install -e .`).

## LIBERO config

LIBERO reads its paths from `$LIBERO_CONFIG_PATH/config.yaml`, by default `~/.libero/config.yaml`,
which may point to another LIBERO installation without these changes. Set `LIBERO_CONFIG_PATH` to
a directory whose `config.yaml` maps the paths into this checkout:

```yaml
assets: <checkout>/libero/libero/assets
bddl_files: <checkout>/libero/libero/bddl_files
benchmark_root: <checkout>/libero/libero
datasets: <checkout>/libero/datasets
init_states: <checkout>/libero/libero/init_files
```

## License and attribution

- **LIBERO**: MIT License, Copyright (c) 2023 Lifelong Robot Learning ([`LICENSE`](LICENSE)).
- **LIBERO-PRO**: the 40 task files in `libero/libero/bddl_files/libero_pro_*/` come from the
  dataset above, MIT License ([`licenses/LIBERO-PRO.txt`](licenses/LIBERO-PRO.txt)), with the
  renaming and edits listed above.

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}

@article{zhou2025liberopro,
  title={LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization},
  author={Zhou, Xueyang and Xu, Yangming and Tie, Guiyao and Chen, Yongchao and Zhang, Guowen and Chu, Duanfeng and Zhou, Pan and Sun, Lichao},
  journal={arXiv preprint arXiv:2510.03827},
  year={2025}
}
```
