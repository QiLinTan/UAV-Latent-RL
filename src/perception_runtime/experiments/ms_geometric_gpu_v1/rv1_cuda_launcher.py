"""Run unmodified RV1 formal trainer under deterministic CUDA and disk guard."""
import os,shutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
from experiments.rv1_range_cnn import train_formal
original_save=train_formal.save
def guarded_save(*args,**kwargs):
    if shutil.disk_usage('/mnt/d').free<20*1024**3:
        raise OSError('D: has under 20 GiB free; checkpoint halted')
    return original_save(*args,**kwargs)
train_formal.save=guarded_save
if __name__=='__main__':train_formal.main()
