import os
import torch
import json

def save_json(checkpoint_dir, config, name='config.json'):
    with open(os.path.join(checkpoint_dir, name), 'w') as handle:
        json.dump(config, handle, indent=2)

def save_checkpoint(checkpoint_dir, iteration, model, optimizer):
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
    torch.save(checkpoint, filename)

def resume_checkpoint(resume_path, model, optimizer):
    checkpoint = torch.load(resume_path)

    iteration = checkpoint['iteration'] + 1
    model.load_state_dict(checkpoint['model'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    return iteration