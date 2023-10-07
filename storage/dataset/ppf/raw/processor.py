from dataclasses import dataclass
import os
import pandas as pd

def create_dir(dir_path, force=False):
    if not os.path.exists(dir_path):
        os.system(f'mkdir {dir_path}')
    elif force:
        os.system(f'rm -rf {dir_path}')
        os.system(f'mkdir {dir_path}')

def process(sourse_file):
    _, file_name = os.path.split(sourse_file)
    name, _ = os.path.splitext(file_name)
    file_dir = os.path.join(ouput_dir, f'{name}_processed')
    create_dir(file_dir)
    df = pd.read_csv(sourse_file, usecols=['original_text', 'reframed_text', 'strategy'])
    group = df.groupby('strategy')
    unknow_index = 1
    for name, indices in group.indices.items():
        try:
            name = '-'.join(eval(name))
            if name is '':
                raise ValueError
        except:
            name = f'unknow_{unknow_index}'
            unknow_index += 1
        file_path = os.path.join(file_dir, f'{name}.csv')
        df.iloc[indices].to_csv(file_path, index_label='index')
# # df[df['strategy']=="['growth', 'impermanence', 'neutralizing']"].to_csv('test.csv', index_label='index')
# ouput_dir = 'preprocessed'
# create_dir(ouput_dir, force=True)
#
# process('wholedev.csv')
# process('wholetest.csv')
# process('wholetrain.csv')
pass

@dataclass
class C:
    a : int
    pass

print(C.__name__)
pass
