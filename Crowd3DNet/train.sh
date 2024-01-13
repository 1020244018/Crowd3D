#!/bin/bash

python -u -m crowd3d.train --configs_yml=configs/train/foot.yml >train_log/foot.log 2>&1
wait

python -u -m crowd3d.train --configs_yml=configs/train/all.yml >train_log/all.log 2>&1

