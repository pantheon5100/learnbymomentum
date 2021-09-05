import shutil
import os

def backup(cfg):
    version = cfg.versions["version"]
    if cfg.wandb:
        experimentdir = f"./code/{cfg.method}_{cfg.project}_{cfg.name}_{version}"
    else:
        experimentdir = f"./code/{cfg.method}_{cfg.project}_{cfg.name}_test"

    if not os.path.exists("./code"):
        os.mkdir("./code")

    if os.path.exists(experimentdir):
        print(experimentdir + ' : exists. overwrite it.')
        shutil.rmtree(experimentdir)
        os.mkdir(experimentdir)
    else:
        os.mkdir(experimentdir)

    shutil.copytree(f"./datasets", os.path.join(experimentdir, 'datasets'))
    shutil.copytree(f"./eval", os.path.join(experimentdir, 'eval'))
    shutil.copytree(f"./methods", os.path.join(experimentdir, 'methods'))
    shutil.copytree(f"./utils", os.path.join(experimentdir, 'utils'))
    shutil.copyfile(f"./train.py", os.path.join(experimentdir, 'train.py'))
    shutil.copyfile(f"./train.py", os.path.join(experimentdir, 'train.py'))
    shutil.copyfile(f"./start_train.sh", os.path.join(experimentdir, 'start_train.sh'))
