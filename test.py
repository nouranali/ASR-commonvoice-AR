import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import OmegaConf


config=OmegaConf.load("/home/nouran.ali/commonvoice/code/conf/citrinet_512.yaml")

config.model.tokenizer.dir="/home/nouran.ali/commonvoice/code/tokenizers/tokenizer_spe_bpe_v512"
config.model.tokenizer.type = "bpe"


config.model.train_ds.manifest_filepath="/home/nouran.ali/commonvoice/code/manifests/ar/commonvoice_train_manifest_processed.json"
config.model.validation_ds.manifest_filepath="/home/nouran.ali/commonvoice/code/manifests/ar/commonvoice_dev_manifest_processed.json"
config.model.test_ds.manifest_filepath="/home/nouran.ali/commonvoice/code/manifests/ar/commonvoice_test_manifest_processed.json"



trainer = pl.Trainer(
    devices= 4, 
    max_steps= -1, 
    num_nodes= 1,
    accelerator= 'gpu',
    strategy='ddp',
    accumulate_grad_batches= 1,
    enable_checkpointing= False, 
    logger= False, 
    log_every_n_steps= 100, 
    val_check_interval= 1.0, 
    check_val_every_n_epoch= 1,
    precision= 32,
    sync_batchnorm= False,
    benchmark= False
)

model = nemo_asr.models.EncDecCTCModelBPE.restore_from("/home/nouran.ali/output/citrinet512bpe_500epochs/Citrinet-512-8x-Stride/checkpoints/Citrinet-512-8x-Stride.nemo")
model.setup_training_data(config.model.train_ds)
model.setup_validation_data(config.model.validation_ds)
model.setup_test_data(config.model.test_ds)
model.set_trainer(trainer)
trainer.test(model)
