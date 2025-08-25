import os
from source.trainer import DPGSTrainer
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from argparse import Namespace
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    _ = wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                   tags=[cfg.wandb.tag], 
                   name = cfg.wandb.name,
                   mode = cfg.wandb.mode)
    omegaconf.OmegaConf.resolve(cfg)
    set_seed(cfg.seed)
    # print(cfg)

    # Init output folder
    print("Output folder: {}".format(cfg.gs.dataset.model_path))
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)

    with open(os.path.join(cfg.gs.dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        # 1. 从最可靠的来源 cfg.gs.dataset 开始，创建一个字典
        args_dict = OmegaConf.to_container(cfg.gs.dataset, resolve=True)

        # 2. 添加或覆盖其他必要的参数
        args_dict['sh_degree'] = cfg.gs.sh_degree
        
        # 3. 根据需要，可以添加其他在配置文件中可能不存在的默认值
        args_dict.setdefault("eval", False)
        args_dict.setdefault("depths", "")
        args_dict.setdefault("convert_SHs_python", False)
        args_dict.setdefault("compute_cov3D_python", False)
        args_dict.setdefault("debug", False)

        # 4. 使用这个最终合并好的字典创建Namespace 对象
        final_namespace_obj = Namespace(**args_dict)
        cfg_log_f.write(repr(final_namespace_obj))

    # Init both agents
    gs = hydra.utils.instantiate(cfg.gs) 
    # Init trainer and launch training
    trainer = DPGSTrainer(GS=gs,
        training_config=cfg.gs.opt,
        dataset_config=cfg.gs.dataset,
        device=cfg.device)
    
    trainer.load_checkpoints(cfg.load)
    trainer.timer.start()
    trainer.init_with_corr(cfg.init_wC)      
    trainer.train(cfg.train)

if __name__ == "__main__":
    main()