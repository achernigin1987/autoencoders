import argparse
import os
import shutil
import sys
from typing import Sequence


def run_training(options, build_model_fn, process_record_fn):
    if not options.eval_data:
        options.eval_data = options.train_data

    if options.clear_dir:
        shutil.rmtree(options.model_dir, ignore_errors=True)
    os.makedirs(options.model_dir, exist_ok=True)

    train_dataset = ImageDataset(options.train_data, mode="train")
    eval_dataset = ImageDataset(options.eval_data, mode="eval")

    data_features = [feature for feature in options.features
                     if feature[0] in dif.DATA_FEATURES]

    model_fn = build_model_fn(ModelParams(stage_sizes=options.stage_sizes,
                                          features_info=data_features,
                                          normalize_color=options.normalize_color,
                                          loss_scale=1))

    estimator = denoiser.create_estimator(
        options,
        derived_model_fn=model_fn,
        dataset_size=train_dataset.size,
        is_training=True)

    def _do_nothing(inputs, target):
        return inputs, target

    process_fn = _do_nothing if options.dont_preprocess else process_record_fn

    num_cycles = options.train_epochs // options.epochs_between_evals
    for cycle in range(num_cycles):
        print("Starting a training cycle {} from {}."
              .format(cycle + 1, num_cycles))
        # train_hooks = hooks_helper.get_train_hooks(
        #     options.hooks, batch_size=options.batch_size)

        def input_fn_train():
            return train_dataset.process(
                is_training=True,
                features_info=data_features,
                is_sequence=options.is_sequence,
                process_record_fn=process_fn,
                batch_size=options.batch_size,
                shuffle_buffer_size=200,
                num_cpu_cores=options.num_cpu_cores,
                num_epochs=options.epochs_between_evals,
                num_parallel_batches=4)

        print(f'Training stage: dataset_size={train_dataset.size}')

        estimator.train(input_fn=input_fn_train, hooks=train_hooks)

        def input_fn_eval():
            return eval_dataset.process(
                is_training=True,
                features_info=data_features,
                is_sequence=options.is_sequence,
                process_record_fn=process_fn,
                batch_size=options.batch_size,
                shuffle_buffer_size=50,
                num_cpu_cores=options.num_cpu_cores,
                num_epochs=1,
                num_parallel_batches=4)

        print(f'Evaluation stage: dataset_size={train_dataset.size}')

        results = estimator.evaluate(input_fn=input_fn_eval)
        print(results)