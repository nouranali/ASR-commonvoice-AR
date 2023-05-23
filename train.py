import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.utils import exp_manager

config=OmegaConf.load("/home/nouran.ali/commonvoice/code/conf/citrinet_512.yaml")
config.model.train_ds.batch_size=32
config.model.validation_ds.batch_size=16
config.model.tokenizer.dir="/home/nouran.ali/commonvoice/code/tokenizers/tokenizer_spe_bpe_v512"
config.model.tokenizer.type="bpe"
config.model.train_ds.manifest_filepath="/home/nouran.ali/commonvoice/code/manifests/ar/commonvoice_train_manifest_processed.json"
config.model.validation_ds.manifest_filepath="/home/nouran.ali/commonvoice/code/manifests/ar/commonvoice_dev_manifest_processed.json"
config.model.test_ds.manifest_filepath="/home/nouran.ali/commonvoice/code/manifests/ar/commonvoice_test_manifest_processed.json"

trainer=pl.Trainer(devices=4,
                  accelerator='gpu',
                  max_epochs=500,    
                  accumulate_grad_batches= 1,
                  enable_checkpointing= False,
                  logger= False,  
                  log_every_n_steps= 5,  
                  val_check_interval= 1.0, 
                  check_val_every_n_epoch= 1,
                  strategy='ddp'
                 )
first_asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=config.model, trainer=trainer)
config.exp_manager.exp_dir = "/home/nouran.ali/output/citrinet512bpe_500epochs/"
config.exp_manager.resume_if_exists = "true"
config.exp_manager.resume_ignore_no_checkpoint= "true"
experiment_manager = exp_manager.exp_manager(trainer=trainer, cfg=config.exp_manager)
trainer.fit(first_asr_model)
print("Saving model...")
first_asr_model.save_to("first_model.nemo")