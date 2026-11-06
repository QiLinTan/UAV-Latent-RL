"""Acceptance-only harness: in-memory small training settings; frozen files untouched."""
import copy
import experiments.rv1_range_cnn.train_formal as trainer
original=trainer.load_config

def test_config():
    c=copy.deepcopy(original())
    c['td3'].update(warmup=0,update_after=1,batch_size=8,buffer_size=32,success_replay_size=16)
    return c
trainer.load_config=test_config
trainer.main()
