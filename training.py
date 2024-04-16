import os
import torch
import numpy as np
import pandas as pd
import pickle
from utils import resize, normalize, load_config, save_pickle, CustomSummaryTracker
import init_training
import argparse

def train(dataset, models, training_pipeline, logging, cfg):
    # Unpack
    trainloader = dataset['trainloader']
    example_batch = dataset['example_batch']
    optimizer = models['optimizer']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    training_loss = logging['training_loss']
    training_summary = logging['training_summary']
    validation_summary = logging['validation_summary']
    example_output = logging['example_output']
    tb_writer = logging['tensorboard_writer']

    # Make dir
    if not os.path.exists(cfg['save_path']):
        os.makedirs(cfg['save_path'])

    # Set torch deterministic (not possible on every GPU)
    if cfg['use_deterministic_algorithms']:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)

    # Training loop
    best_validation_performance = np.inf
    not_improved_count = 0
    epoch = 0
    while epoch <= cfg['epochs'] and not_improved_count < cfg['early_stop_criterium']:
        print(f'\nepoch {epoch}')

        training_loss.reset()
        for batch_idx, batch in enumerate(trainloader, 1):  # range(100):

            # Forward pass
            model_output = forward(batch, models, cfg)
            total_loss = compound_loss_func(model_output)['total']

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            optimizer.step()

            # Track the loss summary
            training_loss.update(compound_loss_func.items())

            if batch_idx % (len(trainloader) // cfg['trainstats_per_epoch']) == 0:
                # Get the average loss over last batches
                training_performance = training_loss.get()
                training_loss.reset()

                # Store and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                training_summary.update({**timestamp, **training_performance})
                tb_writer.add_scalars('loss/training', training_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-tr ' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in training_performance.items()]))

                # Process example batch
                with torch.no_grad():
                    model_output = forward(example_batch, models, cfg, to_cpu=True)

                # Store examples in the summary trackers
                example_output.update(model_output)
                for key in cfg['save_output']:
                    shape = model_output[key].shape
                    if len(shape) == 4:  # Image batch (N,C,H,W)
                        tb_writer.add_images(key,
                                             normalize(model_output[key]),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 5:  # Video batch (N, C, T, H, W)
                        img_batch = model_output[key][0].permute(1,0,2,3) # First video as img batch
                        tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 2: # (N, P)
                        tb_writer.add_histogram(key, model_output[key]) # Stimulation

            if batch_idx % (len(trainloader) // cfg['validations_per_epoch']) == 0:
                # Run validation loop
                validation_performance = validation(dataset, models, training_pipeline, logging, cfg)

                # Track and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                validation_summary.update({**timestamp, **validation_performance})
                tb_writer.add_scalars('/loss/validation', validation_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-val' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in validation_performance.items()]))

                if validation_performance['total'] < best_validation_performance:
                    best_validation_performance = validation_performance['total']
                    print("Model has improved")
                    not_improved_count = 0
                    save_models(models, cfg, prefix='best')

                else:
                    not_improved_count += 1
                    print(f"Not improved during last {not_improved_count} validations")
                    if not_improved_count >= cfg['early_stop_criterium']:
                        break
        epoch += 1

    print("--- Finished training ---\n")


def validation(dataset, models, training_pipeline, logging, cfg):
    # Unpack
    valloader = dataset['valloader']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    validation_loss = logging['validation_loss']

    # Set models to eval
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.eval()

    # Loop over validation set and calculate validation loss
    validation_loss.reset()
    for batch_idx, batch in enumerate(valloader, 1):  # range(100):

        # Forward pass
        with torch.no_grad():
            model_output = forward(batch, models, cfg)
            loss = compound_loss_func(model_output)

        # Update running stats
        validation_loss.update(compound_loss_func.items())

    # Get the average loss over last batches
    validation_performance = validation_loss.get()

    # Reset models to training mode
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.train()
    return validation_performance

def get_timestamp(epoch, batch_idx, total_batches_per_epoch, batch_size):
    timestamp = {'timestamp': f'E{epoch:02d}-B{batch_idx:03d}',
                 'epochs': epoch + batch_idx / total_batches_per_epoch,
                 'samples': batch_size * (batch_idx + total_batches_per_epoch * epoch),}
    return timestamp


def save_models(models, cfg, prefix='best'):
    # Create directory if not exists
    path = os.path.join(cfg['save_path'], 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    # Save model parameters
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            fn = os.path.join(path, f'{prefix}_{name}.pth')
            torch.save(model.state_dict(), fn)
            print(f"Saving parameters to {fn}")


def load_models(models, cfg, prefix='best'):
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_{name}.pth')
            model.load_state_dict(torch.load(fn, map_location=cfg['device']))

def get_validation_results(dataset, models, training_pipeline, cfg):
    output = CustomSummaryTracker()
    performance = CustomSummaryTracker()
    for batch in dataset['valloader']:
        model_output = training_pipeline['forward'](batch, models, cfg, to_cpu=True)
        save_output = {key: model_output[key] for key in cfg['save_output']}
        output.update(save_output)
        loss = training_pipeline['compound_loss_func'](model_output)
        performance.update(training_pipeline['compound_loss_func'].items())
    performance = pd.DataFrame(performance.get())
    return output, performance

def save_validation_results(output, performance, cfg):
    path = os.path.join(cfg['save_path'], 'validation_results')
    print(f'Saving validation results to {path}')
    if output is not None:
        output = {k: torch.cat(v) for k, v in output.get().items()}  # concatenate batches
        save_pickle(output, path)
    performance.to_csv(os.path.join(path, 'validation_performance.csv'))
    performance.describe().to_csv(os.path.join(path, 'performance_summary.csv'))

def save_output_history(logging, cfg):
    path = os.path.join(cfg['save_path'], 'output_history')
    all_output = logging['example_output'].get()
    output = {key: val for key, val in all_output.items() if key in cfg['save_output']}
    save_pickle(output, path)


def save_training_summary(logging, cfg):
    # Write training and validation summary
    for label in ['training', 'validation']:
        fn = os.path.join(cfg['save_path'], f'{label}_summary.csv')
        data = pd.DataFrame(logging[f'{label}_summary'].get())
        data['label'] = label
        data.to_csv(fn, index=False)

def main(args):
    """"Initialize components and run training"""

    # Initialize training
    cfg = load_config(args.config)
    models = init_training.get_models(cfg)
    dataset = init_training.get_dataset(cfg)
    training_pipeline = init_training.get_training_pipeline(cfg)
    logging = init_training.get_logging(cfg)
    train(dataset, models, training_pipeline, logging, cfg)
    save_models(models, cfg, prefix='final')

    # Save the results
    load_models(models, cfg, prefix='best')
    save_output_history(logging, cfg)
    save_training_summary(logging, cfg)
    output, performance = get_validation_results(dataset, models, training_pipeline, cfg)
    if not args.save_output:
        output = None
    save_validation_results(output, performance, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", type=str, default=None,
                       help="filename of config file (yaml) with the training configurations: e.g. '_config.yaml' ")
    # group.add_argument("-l", "--specs-list", type=str, default=None,
    #                     help="filename of specs file (csv) with the list of model specifications") # Todo
    parser.add_argument('-s', '--save-output', action='store_true',
                        help="save the processed validation images after training")


    args = parser.parse_args()
    main(args)