import pytest
from xcell.cls.data import CustomLoader
from ..mappers import mapper_from_name
import os
import shutil
import yaml

# Just run this test in Glamdring
if os.uname()[1] != 'glamdring':
    pytest.skip('Skipping this test as it can only be run in Glamdring',
                allow_module_level=True)

with open('input/master.yml') as f:
    master_data = yaml.load(f, CustomLoader)

path_rerun = 'xcell/tests/cls/input_master_path_rerun_tmp'
nside = 32
coords = 'G'

def cleanup():
    if os.path.isfile(path_rerun):
        shutil.rmtree(path_rerun)

def test_smoke():
    for trn, config in master_data['tracers'].items():
        config['path_rerun'] = path_rerun
        config['nside'] = nside
        if 'coords' not in config:
            config['coords'] = coords
        mapper_class = config['mapper_class']
        tr = mapper_from_name(mapper_class)(config)

    tr.get_signal_map()
    tr.get_mask()

cleanup()
