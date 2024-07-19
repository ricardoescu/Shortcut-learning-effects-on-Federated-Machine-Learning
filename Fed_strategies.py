import torch
import copy


def FedAvg(params):
    """
    Average the parameters from each client to update the global model
    :param params: list of parameters from each client's model
    :return global_params: average of parameters from each client
    """
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        for param in params[1:]:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params

