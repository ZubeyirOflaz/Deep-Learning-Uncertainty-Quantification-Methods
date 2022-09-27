import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


def optuna_ford_a_experimental(trial):
    layers = []
    num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 3, 3)
    num_dense_nodes = trial.suggest_categorical('num_dense_nodes',
                                                [64, 128, 512, 1024])
    dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor',
                                                    [2, 4, 8])
    drop_out = trial.suggest_discrete_uniform('drop_out', 0.1, 0.5, 0.1)
    drop_out_cnn = trial.suggest_discrete_uniform('drop_out_cnn', 0.1, 0.5, 0.1)

    num_filter_last = trial.suggest_categorical('out_channels', [32, 64, 128, 256])

    dict_params = {'num_cnn_blocks': num_cnn_blocks,
                   'num_dense_nodes': num_dense_nodes,
                   'dense_nodes_divisor': dense_nodes_divisor,
                   'drop_out': drop_out}
    input_channels = 1
    for i in range(dict_params['num_cnn_blocks']):
        filter_base = [4, 8, 16, 32, 64]
        filter_selections = [y * (i + 1) for y in filter_base]
        num_filters = trial.suggest_categorical(f'num_filters_{i}', filter_selections)
        kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 6)
        # if len(layers) == 0:
        #    input_channels = 1
        # else:
        #    input_channels = layers[-2].out_channels
        layers.append(nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size))
        input_channels = num_filters
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
        layers.append(nn.Dropout(drop_out_cnn))
    layers.append(nn.AdaptiveMaxPool1d(128))
    layers.append(nn.Conv1d(input_channels, num_filter_last, kernel_size=3))
    layers.append(nn.Flatten())
    linear_input = 126 * num_filter_last
    layers.append(nn.Linear(linear_input, dict_params['num_dense_nodes']))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dict_params['drop_out']))
    layers.append(nn.Linear(dict_params['num_dense_nodes'],
                            int(dict_params['num_dense_nodes'] / dict_params['dense_nodes_divisor'])))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dict_params['drop_out']))
    layers.append(nn.Linear(int(dict_params['num_dense_nodes'] / dict_params['dense_nodes_divisor']), 2))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


def optuna_ford_a(trial):
    layers = []
    num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 3, 5)
    num_dense_nodes = trial.suggest_categorical('num_dense_nodes',
                                                [64, 128, 512, 1024])
    dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor',
                                                    [2, 4, 8])
    drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)
    drop_out_cnn = trial.suggest_discrete_uniform('drop_out_cnn', 0.05, 0.5, 0.05)

    num_filter_last = trial.suggest_categorical('out_channels', [32, 64, 128])

    dict_params = {'num_cnn_blocks': num_cnn_blocks,
                   'num_dense_nodes': num_dense_nodes,
                   'dense_nodes_divisor': dense_nodes_divisor,
                   'drop_out': drop_out}
    input_channels = 1
    for i in range(dict_params['num_cnn_blocks']):
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        # if len(layers) == 0:
        #    input_channels = 1
        # else:
        #    input_channels = layers[-2].out_channels
        layers.append(nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size))
        input_channels = num_filters
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
        layers.append(nn.Dropout(drop_out_cnn))
    layers.append(nn.AdaptiveMaxPool1d(128))
    layers.append(nn.Conv1d(input_channels, num_filter_last, kernel_size=3))
    layers.append(nn.Flatten())
    linear_input = 126 * num_filter_last
    layers.append(nn.Linear(linear_input, dict_params['num_dense_nodes']))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dict_params['drop_out']))
    layers.append(nn.Linear(dict_params['num_dense_nodes'],
                            int(dict_params['num_dense_nodes'] / dict_params['dense_nodes_divisor'])))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dict_params['drop_out']))
    layers.append(nn.Linear(int(dict_params['num_dense_nodes'] / dict_params['dense_nodes_divisor']), 2))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


def optuna_ford_a_mimo(trial, trial_parameters):
    ensemble_num = trial_parameters['ensemble_num']
    device = trial_parameters['device']

    # study_name = trial_parameters ['study_name']

    class MimoCnnModel(nn.Module):
        def __init__(self, ensemble_num: int, num_categories: int):
            super(MimoCnnModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 512)
            self.num_channels = trial.suggest_int('num_channels', 128, 256)
            self.final_img_resolution = 32 * ensemble_num
            self.input_dim = self.num_channels * ((self.final_img_resolution) - 2)
            self.conv_module = ConvModule(self.num_channels, self.final_img_resolution, ensemble_num)
            self.linear_module = LinearModule(self.input_dim, self.output_dim)
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            batch_size = input_tensor.size()[0]
            conv_result = self.conv_module(input_tensor)
            # print(self.input_dim)
            # print(conv_result.size())
            # print(conv_result.reshape(batch_size, -1).size())
            output = self.linear_module(conv_result.reshape(batch_size, -1))
            # print('tensor shapes')
            # print(output.size())
            output = self.output_layer(output)
            # print(output.size())
            output = output.reshape(
                batch_size, ensemble_num, -1
            )  # (batch_size, ensemble_num, num_categories, ensemble_num)
            # print(output.size())
            # output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)
            # print(output.size())
            output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, num_categories)
            # print(output.size())
            return output

    class ConvModule(nn.Module):
        def __init__(self, num_channels: int, final_img_resolution: int, ensemble_num: int):
            super(ConvModule, self).__init__()
            layers = []
            num_layers = trial.suggest_int('num_cnn_layers', 5, 5)
            drop_out_cnn = trial.suggest_discrete_uniform('drop_out_cnn', 0.05, 0.5, 0.05)
            input_channels = 1
            for i in range(num_layers):
                filter_base = [4, 8, 16]
                filter_selections = [y * (num_layers + 1) for y in filter_base]
                num_filters = trial.suggest_categorical(f'num_filters_{i}', filter_selections)
                kernel_size = trial.suggest_int(f'kernel_size_{i}', 2, 5)
                dilation_amount = trial.suggest_int(f'dilation_{i}', 1, 2)
                layers.append(nn.Conv1d(input_channels, num_filters, kernel_size, dilation=dilation_amount))
                layers.append(nn.ReLU())
                if i < 3:
                    pool_stride = 3
                else:
                    pool_stride = 2
                if i != num_layers - 1:
                    layers.append(nn.MaxPool1d(3, pool_stride))
                    layers.append(nn.Dropout(drop_out_cnn))
                input_channels = num_filters
            layers.append(nn.AdaptiveMaxPool1d(final_img_resolution))
            layers.append(nn.Conv1d(input_channels, num_channels, 3))
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            output = self.module(input_tensor)
            return output

    class LinearModule(nn.Module):
        def __init__(self, input_dimension: int, output_dimension: int):
            super(LinearModule, self).__init__()
            layers = []
            in_features = input_dimension
            num_layers = 2  # trial.suggest_int('num_layers', 1, 3)
            for i in range(num_layers):
                out_dim = trial.suggest_int('n_units_l{}'.format(i), 128, 2048)
                layers.append(nn.Linear(in_features, out_dim))
                layers.append(nn.ReLU())
                dropout_rate = trial.suggest_float('dr_rate_l{}'.format(i), 0.0, 0.5)
                if dropout_rate > 0.05:
                    layers.append(nn.Dropout(dropout_rate))
                in_features = out_dim
            layers.append(nn.Linear(in_features, output_dimension))
            layers.append(nn.ReLU())
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = self.module(x)
            return output

    mimo_optuna = MimoCnnModel(ensemble_num=ensemble_num, num_categories=2)
    return mimo_optuna


def optuna_ford_a_mimo_experimental(trial, trial_parameters):
    ensemble_num = trial_parameters['ensemble_num']
    device = trial_parameters['device']

    # study_name = trial_parameters ['study_name']

    class MimoCnnModel(nn.Module):
        def __init__(self, ensemble_num: int, num_categories: int):
            super(MimoCnnModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 512)
            self.num_channels = trial.suggest_int('num_channels', 128, 256)
            self.final_img_resolution = 32 * ensemble_num
            self.input_dim = self.num_channels * ((self.final_img_resolution) - 2)
            self.conv_module = ConvModule(self.num_channels, self.final_img_resolution, ensemble_num)
            self.linear_module = LinearModule(self.input_dim, self.output_dim)
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            batch_size = input_tensor.size()[0]
            conv_result = self.conv_module(input_tensor)
            # print(self.input_dim)
            # print(conv_result.size())
            # print(conv_result.reshape(batch_size, -1).size())
            output = self.linear_module(conv_result.reshape(batch_size, -1))
            # print('tensor shapes')
            # print(output.size())
            output = self.output_layer(output)
            # print(output.size())
            output = output.reshape(
                batch_size, ensemble_num, -1
            )  # (batch_size, ensemble_num, num_categories, ensemble_num)
            # print(output.size())
            # output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)
            # print(output.size())
            output = nn.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, num_categories)
            # print(output.size())
            return output

    class ConvModule(nn.Module):
        def __init__(self, num_channels: int, final_img_resolution: int, ensemble_num: int):
            super(ConvModule, self).__init__()
            layers = []
            num_layers = trial.suggest_int('num_cnn_layers', 5, 5)
            drop_out_cnn = trial.suggest_discrete_uniform('drop_out_cnn', 0.05, 0.5, 0.05)
            input_channels = 1
            for i in range(num_layers):
                filter_base = [4, 8, 16, 32]
                filter_selections = [y * (i + 1) for y in filter_base]
                num_filters = trial.suggest_categorical(f'num_filters_{i}', filter_selections)
                kernel_size = trial.suggest_int(f'kernel_size_{i}', 2, 5)
                dilation_amount = trial.suggest_int(f'dilation_{i}', 1, 2)
                layers.append(nn.Conv1d(input_channels, num_filters, kernel_size, dilation=dilation_amount))
                layers.append(nn.ReLU())
                if i < 3:
                    pool_stride = 3
                else:
                    pool_stride = 2
                if i != num_layers - 1:
                    layers.append(nn.MaxPool1d(3, pool_stride))
                    layers.append(nn.Dropout(drop_out_cnn))
                input_channels = num_filters
            layers.append(nn.AdaptiveMaxPool1d(final_img_resolution))
            layers.append(nn.Conv1d(input_channels, num_channels, 3))
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            output = self.module(input_tensor)
            return output

    class LinearModule(nn.Module):
        def __init__(self, input_dimension: int, output_dimension: int):
            super(LinearModule, self).__init__()
            layers = []
            in_features = input_dimension
            num_layers = 2  # trial.suggest_int('num_layers', 1, 3)
            for i in range(num_layers):
                out_dim = trial.suggest_int('n_units_l{}'.format(i), 128, 2048)
                layers.append(nn.Linear(in_features, out_dim))
                layers.append(nn.ReLU())
                dropout_rate = trial.suggest_float('dr_rate_l{}'.format(i), 0.0, 0.5)
                if dropout_rate > 0.05:
                    layers.append(nn.Dropout(dropout_rate))
                in_features = out_dim
            layers.append(nn.Linear(in_features, output_dimension))
            layers.append(nn.ReLU())
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = self.module(x)
            return output

    mimo_optuna = MimoCnnModel(ensemble_num=ensemble_num, num_categories=2)
    return mimo_optuna
