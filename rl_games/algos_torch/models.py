import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.algos_torch.torch_ext import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs


class BaseModel():
    def __init__(self, model_class):
        self.model_class = model_class

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)

class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape[0] - 32) #TODO 16 is z_size
                #self.running_mean_std_depth = RunningMeanStd(32*32) #TODO 16 is z_size
                #self.running_mean_std = RunningMeanStd(obs_shape[0] - 3 + 32*32) #TODO 16 is z_size

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def norm_obs_depth(self, observation):
        with torch.no_grad():
            return self.running_mean_std_depth(observation) if self.normalize_input else observation

    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

class ModelA2C(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()            

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)

            if is_train:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                return result
            else:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
                return  result

class ModelA2CMultiDiscrete(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                return result
            else:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]                
                
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
                }
                return  result

class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, sigma, value, states = self.a2c_network(input_dict)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return  result          


class ApproxRepSet(torch.nn.Module):

    def __init__(self, n_hidden_sets, n_elements, d, n_classes, device):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        
        self.Wc = nn.Parameter(torch.FloatTensor(d, n_hidden_sets*n_elements))
        self.fc1 = nn.Linear(n_hidden_sets, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t,_ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.relu(self.fc1(t))
        out = self.fc2(t)

        return F.log_softmax(out, dim=1)


class PosQuatPredictor(nn.Module):
    def __init__(self, hidden, visual_emb, use_q=True) -> None:
        super().__init__()

        self.use_q = use_q
        self.project_visual_emb = nn.Linear(visual_emb, hidden)

        inp = hidden + 8 if self.use_q else hidden
        self.predict_pos_quat = nn.Sequential(nn.Linear(inp, hidden*2), nn.SELU(), nn.Linear(hidden*2, 7))

    def forward(self, visuals, q=None):
        visuals = self.project_visual_emb(visuals)
        inp = torch.cat((visuals, q), -1) if self.use_q else visuals
        return self.predict_pos_quat(inp)
        
class Encoder_small(nn.Module):
    def __init__(self, hidden_size, vae=False, predict_cubeA_state=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vae = vae
        D = 8
        final_dim = hidden_size-7 if predict_cubeA_state else hidden_size
        self.conv1 = nn.Conv2d(1, D, kernel_size=4, stride=2, padding=1) #(batch_size, 8, 64, 64)
        self.conv2 = nn.Conv2d(D, D*2, kernel_size=4, stride=2, padding=1) #(batch_size, 16, 32, 32)
        #self.bn2 = nn.BatchNorm2d(D*2)
        #self.bn = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(2*D, 4*D, kernel_size=4, stride=2, padding=1) 
        #self.bn3 = nn.BatchNorm2d(D*4)
        self.conv4 = nn.Conv2d(4*D, final_dim, kernel_size=8, stride=1, padding=0)  # (batch_size, 32, 16, 16)
        #self.act = nn.ReLU()
        self.act = nn.ELU()

        #self.predict_cubeA_state = nn.Linear(hidden_size-7, 7) if predict_cubeA_state else None

        self.predict_cubeA_state = PosQuatPredictor(hidden=24, visual_emb=hidden_size-7) if predict_cubeA_state else None
        self.cubeA_state_prediction = None
           



        if self.vae:
            self.mu = nn.Sequential(nn.Linear(hidden_size, hidden_size))
            self.log_var = nn.Sequential(nn.Linear(hidden_size, hidden_size))
            self.KLD = None

    def forward(self, x, q=None):

        x  = self.act(self.conv1(x))
        

        x  = self.act(self.conv2(x))
        #x = self.bn2(x)

        x = self.act(self.conv3(x))
        #x = self.bn3(x)

        z = self.act(self.conv4(x)).reshape(x.size(0), -1)
        if self.predict_cubeA_state is not None:
            self.cubeA_state_prediction = self.predict_cubeA_state(z, q)
            z = torch.cat((z, self.cubeA_state_prediction), dim=-1)
        else:
            self.cubeA_state_prediction = None
        return z

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space.
        In order for the back-propagation to work, we need to be able to calculate the gradient.
        This reparameterization trick first generates a normal distribution, then shapes the distribution
        with the mu and variance from the encoder.

        This way, we can can calculate the gradient parameterized by this particular random instance.
        """
        eps = torch.rand_like(log_var)
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    def __init__(self, hidden_size, vae=False, predict_cubeA_state=False, use_q=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.vae = vae
        D = 8
        final_dim = hidden_size - 7 if predict_cubeA_state else hidden_size
        # self.conv1 = nn.Conv2d(1, D, kernel_size=3, stride=2, padding=1) #(batch_size, 8, 64, 64)
        # self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(1, D*2, kernel_size=3, stride=2, padding=1) #(batch_size, 16, 32, 32)
        self.conv2_rgb = nn.Conv2d(1, D*2, kernel_size=3, stride=2, padding=1)
        #self.bn2 = nn.BatchNorm2d(D*2)
        self.conv3 = nn.Conv2d(D*4, D * 4, kernel_size=3, stride=2, padding=1)  # (batch_size, 32, 16, 16)
        self.bn3 = nn.BatchNorm2d(D * 4)
        self.conv4 = nn.Conv2d(D * 4, D * 8, kernel_size=3, stride=2, padding=1)  # (batch_size, 64, 8, 8)
        self.bn4 = nn.BatchNorm2d(D * 8)
        self.conv5 = nn.Conv2d(D * 8, D * 16, kernel_size=3, stride=2, padding=1)  # (batch_size, 128, 4, 4)
        self.bn5 = nn.BatchNorm2d(D * 16)
        self.conv6 = nn.Conv2d(D * 16, D * 32, kernel_size=3, stride=2, padding=1)  # (batch_size, 256, 2, 2)
        self.bn6 = nn.BatchNorm2d(D * 32)
        self.conv7 = nn.Conv2d(D * 32, final_dim, kernel_size=2, stride=2, padding=0)  # (batch_size, 512, 1, 1))
        #self.bn7 = nn.BatchNorm2d(hidden_size)
        self.act = nn.ELU()

        # self.skip1 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(1, D, kernel_size=1, stride=1))

        # self.skip2 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(4, 2*D, kernel_size=1, stride=1))

        # self.skip3 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(4*D, 4 * D, kernel_size=1, stride=1))

        # self.skip4 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(4 * D, 8 * D, kernel_size=1, stride=1))

        # self.skip5 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(8 * D, 16 * D, kernel_size=1, stride=1))

        # self.skip6 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(16 * D, 32 * D, kernel_size=1, stride=1))

        # self.skip7 = nn.Sequential(nn.AvgPool2d(2),
        #                            nn.Conv2d(32 * D, hidden_size, kernel_size=1, stride=1))

        
        self.predict_cubeA_state = PosQuatPredictor(hidden=24, visual_emb=final_dim, use_q=use_q) if predict_cubeA_state else None
        self.cubeA_state_prediction = None

        # self.conv_64 = nn.Sequential(nn.Conv2d(1, D, kernel_size=3, stride=1, padding=1), nn.ReLU())

        if self.vae:
            self.mu = nn.Sequential(nn.Linear(hidden_size, hidden_size))
            self.log_var = nn.Sequential(nn.Linear(hidden_size, hidden_size))
            self.KLD = None

    def forward(self, x, q=None):
        # x_s = self.skip1(x)
        # x  = self.act(self.bn1(self.conv1(x)))
        # x = x + x_s

        #x_s = self.skip2(x)
        x1  = self.act(self.conv2(x[:,0:1,:,:]))
        x2  = self.act(self.conv2_rgb(x[:,1:,:,:]))
        #print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        #x = x + x_s


        #x_s = self.skip3(x)
        x = self.act(self.bn3(self.conv3(x)))
        #x = x + x_s

        #x_s = self.skip4(x)
        x = self.act(self.bn4(self.conv4(x)))
        #x = x + x_s

        #x_s = self.skip5(x)
        x = self.act(self.bn5(self.conv5(x)))
        #x = x + x_s

        #x_s = self.skip6(x)
        x = self.act(self.bn6(self.conv6(x)))
        #x = x + x_s

        #x_s = self.skip7(x)
        x = self.act(self.conv7(x))
        #x = x + x_s

        if self.vae:
            mu = self.mu(x.view(x.size(0), -1))
            log_var = self.log_var(x.view(x.size(0), -1))

            z = self.reparameterize(mu, log_var)

            KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            self.KLD = torch.sum(KLD_element).mul_(-0.5)
        else:
            z = x.reshape(x.size(0), -1)


        if self.predict_cubeA_state is not None:
            self.cubeA_state_prediction = self.predict_cubeA_state(z, q)
            z = torch.cat((z, self.cubeA_state_prediction), dim=-1)
        else:
            self.cubeA_state_prediction = None
        return z
    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space.
        In order for the back-propagation to work, we need to be able to calculate the gradient.
        This reparameterization trick first generates a normal distribution, then shapes the distribution
        with the mu and variance from the encoder.

        This way, we can can calculate the gradient parameterized by this particular random instance.
        """
        eps = torch.rand_like(log_var)
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

class Decoder(nn.Module): 
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        D = 8
        self.conv1 = nn.ConvTranspose2d(hidden_size, D*4, kernel_size=2, stride=2, output_padding=0)
        #self.bn1 = nn.BatchNorm2d(D*4)
        self.conv2 = nn.ConvTranspose2d(D*4, D*8, kernel_size=4, stride=2, padding=0, output_padding=0) #(batch_size, 16, 32, 32)
        #self.bn2 = nn.BatchNorm2d(D*8)
        self.conv3 = nn.ConvTranspose2d(D*8, D*16, kernel_size=4, stride=2, padding=0, output_padding=0) #(batch_size, 32, 16, 16)
        #self.bn3 = nn.BatchNorm2d(D*16)
        self.conv4 = nn.ConvTranspose2d(D*16, D*32, kernel_size=4, stride=2, padding=0, output_padding=0) #(batch_size, 64, 8, 8)
        #self.bn4 = nn.BatchNorm2d(D*32)
        self.conv5 = nn.ConvTranspose2d(D*32, 2, kernel_size=8, stride=2, padding=1, output_padding=0) #(batch_size, 128, 4, 4)
        # self.bn5 = nn.BatchNorm2d(D*2)
        # self.conv6 = nn.ConvTranspose2d(D*2, D, kernel_size=3, stride=2, padding=1, output_padding=1) #(batch_size, 256, 2, 2)
        # self.bn6 = nn.BatchNorm2d(D)
        # self.conv7 = nn.ConvTranspose2d(D, 1, kernel_size=3, stride=2, padding=1, output_padding=1) #(batch_size, 512, 1, 1))
        self.act = nn.SELU()
    
    def forward(self, x):
        x  = self.act(self.conv1(x.view(-1, self.hidden_size, 1, 1)))
        x  = self.act(self.conv2(x))
        x  = self.act(self.conv3(x))
        x  = self.act(self.conv4(x))
        x  = self.conv5(x)  #self.act(self.bn5(
        #x  = self.act(self.bn6(self.conv6(x)))
        #x  = self.conv7(x)

        return x


class AutoencoderDepth(nn.Module):
    def __init__(self, hidden_size, img_size=32, vae=False, predict_cubeA_state=False, use_q_for_cubeA_prediction=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.running_mean = RunningMeanStd(insize=(2, self.img_size*self.img_size), per_channel=True)  

        D = 16
        self.encoder = Encoder(hidden_size, vae=vae, predict_cubeA_state=predict_cubeA_state, use_q=use_q_for_cubeA_prediction)
        #self.decoder = Decoder(hidden_size)
        self.reconstruction_loss = None
        # self.post_decoder = nn.Sequential(nn.Conv2d(1, D, kernel_size=3, stride=1, padding=1),
        #                                   nn.ELU(),
        #                                   nn.Conv2d(D, 2 * D, kernel_size=3, stride=1, padding=1),
        #                                   nn.ELU(),
        #                                   nn.Conv2d(2 * D, 1, kernel_size=3, stride=1, padding=1))

    def encode(self, x, mask=None, q=None):
        x = self.running_mean(x.view(x.shape[0], 2, -1), mask=mask)
        x = x.reshape(-1, 2, self.img_size, self.img_size) # * mask
        # x = torchvision.transforms.functional.resize(x, (128,128))
        # for debugging:
        # plt.imshow(x.view(-1,128,128)[-1].detach().cpu().numpy())
        # plt.savefig('./depth_natural2.png')F
        z = self.encoder(x, q) #, q
        #x_hat = self.decoder(z)
        #self.reconstruction_loss = nn.functional.mse_loss(x_hat, x)
        return z

    # def decode(self, x):
    #     x = x.view(-1, self.hidden_size, 1, 1)
    #     x = self.decoder(x)
    #     post_x_hat = self.post_decoder(x) + x.detach()
    #     return post_x_hat.view(-1, self.img_size * self.img_size), x.view(-1, self.img_size * self.img_size)
    #
    # def forward(self, x):
    #     z = self.encode(x)
    #     post_x_hat, x_hat = self.decode(z)
    #
    #     return post_x_hat, x_hat, z



class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # self.auto_encoder = AutoencoderDepth(16) #


        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)

            if torch.all(input_dict['obs'] == 0):
               input_dict['obs'] = input_dict['obs'][:, :14] #12 15 is the size of the original observations (not including depth and etc)...
            # if torch.all(input_dict['obs'] == 0):
            #    input_dict['obs'] = input_dict['obs'][:, :1] #15 is the size of the original observations (not including depth and etc)...

            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            #input_dict['depth_images'] = self.norm_obs_depth(input_dict['depth_images'].view(input_dict['depth_images'].shape[0], -1)).view(input_dict['depth_images'].shape[0], 32,32)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    # 'depth_features' : depth_encoding,
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    # 'depth_features': depth_encoding,

                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)


class ModelCentralValue(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            return None # or throw exception?

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            value, states = self.a2c_network(input_dict)
            if not is_train:
                value = self.unnorm_value(value)

            result = {
                'values': value,
                'rnn_states': states
            }
            return result



class ModelSACContinuous(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'sac')
        self.network_builder = network
    
    class Network(BaseModelNetwork):
        def __init__(self, sac_network,**kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.sac_network = sac_network

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)
        
        def is_rnn(self):
            return False

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist



