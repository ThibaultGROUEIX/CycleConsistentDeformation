import sys

sys.path.append("./auxiliary/")
sys.path.append("./extension/")
import argument_parser
import my_utils

opt = argument_parser.parser()
my_utils.plant_seeds(randomized_seed=False)

if opt.train_atlasnet:
    import trainer_atlasnet as trainer
else:
    import trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset_train()
trainer.build_dataset_test()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()

for epoch in range(opt.nepoch):
    trainer.train_epoch()
    trainer.test_epoch()
    trainer.dump_stats()
    trainer.save_network()
    trainer.increment_epoch()

trainer.save_new_experiments_results()
