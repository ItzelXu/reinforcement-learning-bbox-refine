import os
import math
import copy
import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque, namedtuple
from tqdm import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange
from accelerate import Accelerator

from reward import build_reward_model
from datasets.coco import CocoDetection
import datasets.transforms as T
import util.misc as utils
from util.misc import NestedTensor


class Actor(nn.Module):
    # actor model
    def __init__(self, model, postprocessor=None, device='cuda'):
        super(Actor, self).__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.model.to(device)
        if self.postprocessor is not None:
            self.postprocessor.to(device)
    
    def forward(self, x):
        x = self.model(x)
        if self.postprocessor is not None:
            x = self.postprocessor(x)
        return x


class Critic(nn.Module):
    # critic model
    def __init__(self, model, model_dim=501*2094, device='cuda'):
        super(Critic, self).__init__()
        self.model = model
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(model_dim, 1),
            Rearrange('... 1 -> ...')
        )
        self.model.to(device)
        self.value_head.to(device)

        nn.init.zeros_(self.value_head[1].bias)
        nn.init.orthogonal_(self.value_head[1].weight, gain = math.sqrt(2))
    
    def forward(self, x):
        x = self.model(x)
        x = x['pred_seq_logits']
        #print(x.shape)
        return self.value_head(x)


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [480, 512, 544, 576, 608, 640]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=640),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=640),
            normalize,
        ])
    if image_set == 'ppo':
        return T.Compose([
            #T.RandomResize([480], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def detach_to_cpu_(tensor):
    if isinstance(tensor, list):
        return [detach_to_cpu_(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: detach_to_cpu_(v) for k, v in tensor.items()}
    else:
        tensor = tensor.detach().cpu()
        if tensor.shape[0] == 1 and len(tensor.shape) > 3:
            return rearrange(tensor, '1 ... -> ...')
        else:
            return tensor


def to_device(tensor, device):
    if isinstance(tensor, list):
        return [to_device(t, device) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: v.to(device) for k, v in tensor.items()}
    else:
        return tensor.to(device)

Memory = namedtuple(
    'Memory',
    ['state', 'reward', 'a_prob', 'a_log_prob', 'value']
)


class ExperienceDataset(Dataset):
    def __init__(self, data, device=None):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return to_device(self.data[ind], device=self.device)

def create_dataloader(data, batch_size, shuffle=True, device=None, **kwargs):
    ds = ExperienceDataset(data, device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


class PPO():
    # PPO class
    def __init__(self, model, critic_model=None, device='cuda:0'):
        self.accelerate = Accelerator()
        self.device = self.accelerate.device
        # get actor and critic model
        #self.device = torch.device(device)
        self.actor_net = Actor(model, device=self.device)
        if critic_model is None:
            critic_model = copy.deepcopy(model)
        self.critic_net = Critic(critic_model, device=self.device)
        
        # get reward model
        self.reward_model = build_reward_model('recall')
        
        # set up parameters
        self.memory_buffer = deque([])
        self.counter = 0
        self.training_step = 0
        
        # configuration parameters
        self.buffer_capacity = 1000
        self.num_episodes = 50000
        self.minibatch_size = 1
        self.epochs = 1
        self.kl_div_loss_weight = 0.1
        self.eps_clip = 0.1
        self.beta_s = 0.01
        self.value_clip = 0.4

        # get optimizer
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-6)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 4e-6)
        
        # accelerate with multi-GPU
        self.actor_net, self.critic_net, self.actor_optimizer, self.critic_optimizer = \
            self.accelerate.prepare(self.actor_net, self.critic_net, self.actor_optimizer, self.critic_optimizer) 

        # set up dataset
        img_folder = 'coco/train2017'
        ann_file = 'coco/annotations/instances_train2017.json'
        dataset_train = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms('train'),
            return_masks=False,
            large_scale_jitter=False,
            image_set='train'
        )
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, 3, drop_last=True)

        self.data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn, num_workers=2)
        # test
        #for sample, target in self.data_loader_train:
        #    print(sample)
        #    print(target)
        #    break

    @torch.no_grad()
    def generate(self, state):
        # generate actions
        # state: batched image tensors + target (None)
        actions = self.actor_net(state)
        action_logits = actions['pred_seq_logits']
        action_probs = action_logits.softmax(dim=-1)
        action_log_probs = torch.log(action_probs.clamp(min=1e-20))
        # actions = self.postprocessor(action_logits)
        # reward = self.reward_model(actions, state)
        return actions, action_probs, action_log_probs
    
    @torch.no_grad()
    def get_value(self, state):
        value = self.critic_net(state)
        return value

    def save(self, save_dir='weights'):
        actor_save_path = os.path.join(save_dir, 'actor_net_{}.pth'.format(self.training_step))
        critic_save_path = os.path.join(save_dir, 'critic_net_{}.pth'.format(self.training_step))
        torch.save(self.actor_net.state_dict(), actor_save_path)
        torch.save(self.critic_net.state_dict(), critic_save_path)

    def load(self, load_dir='weights'):
        actor_load_path = os.path.join(load_dir, 'actor_net_{}.pth'.format(self.training_step))
        critic_load_path = os.path.join(load_dir, 'critic_net_{}.pth'.format(self.training_step))
        checkpoint = torch.load(actor_load_path, map_location='cpu')
        self.actor_net.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(critic_load_path, map_location='cpu')
        self.critic_net.load_state_dict(checkpoint['model'])

    def store_transition(self, transition):
        self.memory_buffer.append(transition)
        self.counter += 1
        print('stored to memories: {}/{}'.format(self.counter % self.buffer_capacity, self.buffer_capacity), end='\r')
        return self.counter % self.buffer_capacity == 0

    def update(self):
        # update model when buffer capacity meets max
        self.training_step += 1
        self.actor_net.train()
        self.critic_net.train()
        
        # get from buffer
        data = [[t.state, t.reward, t.a_prob, t.a_log_prob, t.value] for t in self.memory_buffer]
        #print('='*50)
        #print(data[0][0][0].shape)
        #print(data[0][0][1].shape)
        #print(data[0][0][2])
        #print(data[0][1].shape)
        #print(data[0][2].shape)
        #print(data[0][3].shape)
        #print(data[0][4].shape)
        #dl = create_dataloader(data, self.minibatch_size, device = self.device)
        #print(data)
        #print('='*50)
        random.shuffle(data)
        #print(data)
        for _ in range(self.epochs):
            #for (state, rewards, old_action_probs, old_action_log_probs, old_values) in dl:
            #for list_data in data:
            tbar = tqdm(total=len(data))
            for list_data in data:
                #tbar.set_description('updating policy:')
                list_data = to_device(list_data, self.device)
                state, rewards, old_action_probs, old_action_log_probs, old_values = list_data
                img, mask, targets = state
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                if len(mask.shape) == 4:
                    mask = mask.squeeze(0)
                #print(img.shape)
                #print(mask.shape)
                #print(targets)
                #print(rewards.shape)
                #print(old_action_probs.shape)
                #print(old_action_log_probs.shape)
                #print(old_values.shape)
                samples = NestedTensor(img, mask)
                state = [samples, targets]
                #print(img.shape, mask.shape)
                action_logits = self.actor_net(state)['pred_seq_logits']
                #print(action_logits.shape)
                #values = self.critic_net(state)
                action_probs = action_logits.softmax(dim=-1)
                action_log_probs = torch.log(action_probs.clamp(min=1e-20))
                #print(action_log_probs.shape)
                # entropy
                entropies = (action_probs * action_log_probs).sum(dim=-1).mean()
                #print(entropies.shape)
                # kl div
                kl_penalty = 0.
                if self.kl_div_loss_weight > 0:
                    kl_penalty = (old_action_probs * (old_action_log_probs - action_log_probs)).sum(dim = -1).mean() * self.kl_div_loss_weight
                # subtract kl penalty from the rewards
                rewards = rewards - kl_penalty
                #print(rewards.shape)
                # calculate clipped surrogate objective, classic PPO loss
                ratios = (action_log_probs - old_action_log_probs).exp()
                #print(ratios.shape)
                advantages = rewards - old_values
                #print(old_values.shape)
                #print(advantages)
                # normalize
                mean_advantages = advantages.mean(dim=-1, keepdim=True)
                centered_advantages = advantages - mean_advantages
                var_advantages = (centered_advantages ** 2).mean(dim=-1, keepdim=True)
                advantages = centered_advantages * var_advantages.clamp(min=1e-5).rsqrt()
                # compute policy loss
                advantages = advantages.reshape((-1, 1, 1))
                #print(advantages)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                #print(advantages)
                #print(entropies)
                #print(torch.min(ratios))
                #print(torch.max(ratios))
                #print(ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip))
                #print(torch.min(surr1))
                #print(torch.min(surr2))
                #policy_loss = - surr2 - self.beta_s * entropies
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropies
                loss = policy_loss.mean()
                #print(torch.max(policy_loss))
                #print(loss)
                # train actor model
                self.actor_optimizer.zero_grad()
                self.accelerate.backward(loss)
                # loss.backward()
                self.actor_optimizer.step()
                # self.actor_optimizer.zero_grad()
                
                # compute value loss
                #values, rewards.detach(), old_values, self.value_clip
                values = self.critic_net(state)
                value_clipped = old_values + (values - old_values).clamp(-self.value_clip, self.value_clip)
                value_loss_1 = (value_clipped.flatten() - rewards.detach()) ** 2
                value_loss_2 = (values.flatten() - rewards.detach()) ** 2
                value_loss = torch.mean(torch.max(value_loss_1, value_loss_2)).mean()
                # train critic model
                self.critic_optimizer.zero_grad()
                self.accelerate.backward(value_loss)
                #value_loss.backward()
                self.critic_optimizer.step()
                # self.critic_optimizer.zero_grad()
                tbar.set_postfix(loss=float(loss), value_loss=float(value_loss))
                tbar.update(1)
    def train(self):
        # main loop
        for eps in range(self.num_episodes):
            print('running episode: {}/{}'.format(eps, self.num_episodes))
            #generate samples for reinforce learning
            for samples, targets in self.data_loader_train:
                with torch.no_grad():
                    samples = samples.to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    state = [samples, targets]
                    actions, action_probs, action_log_probs = self.generate(state)
                    rewards = self.reward_model(actions, targets)
                    #print('rewards:', rewards)
                    values = self.get_value(state)
                # print(actions, rewards, action_probs, action_log_probs, values)
                # trans = Transition(state, actions, rewards, action_probs, action_log_probs, values)
                # detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')
                #print(type(state[0]), type(state[1]))
                #print(type(actions))
                #print(type(rewards))
                #print(type(action_probs))
                #print(type(action_log_probs))
                #print(type(values))
                img, mask = samples.decompose()
                #print(img.shape)
                #print(mask.shape)
                #print(targets)
                #print(rewards.shape)
                #print(action_probs.shape)
                #print(action_log_probs.shape)
                #print(values.shape)
                state = [img, mask, targets]
                trans = Memory(*map(detach_to_cpu_, (
                    state,
                    rewards,
                    action_probs,
                    action_log_probs,
                    values
                )))
                signal = self.store_transition(trans)
                if signal:
                    print('\nstart to update...')
                    self.update()
                    self.memory_buffer.clear()
                    print('saving weights...')
            self.save()
        print('RLHF training complete!')
        

if __name__ == "__main__":
    import argparse
    from common import get_args_parser
    from playground import build_all_model
    import loralib as lora
    parser = argparse.ArgumentParser('Pix2Seq PPO script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, criterion, postprocessors = build_all_model[args.model](args)
    lora.mark_only_lora_as_trainable(model)
    if not args.resume:
        raise RuntimeError('Please specify model weight path using --resume')
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    critic_model = copy.deepcopy(model)
    ppo = PPO(model=model, critic_model=critic_model)
    ppo.train()



