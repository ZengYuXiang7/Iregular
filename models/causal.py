from castle.algorithms import ICALiNGAM
from castle.algorithms import DirectLiNGAM
from castle.algorithms import PC
from castle.algorithms import Notears
from castle.algorithms import GraNDAG
from castle.algorithms import GOLEM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import time

from models.hcd import HCD
from models.sada import SADA


def get_causal_matrix(data, method, config):
    data = data.astype(np.float32)
    data = StandardScaler().fit_transform(data)
    start = time.time()
    if method == "PC":
        model = PC(alpha=config.pc_alpha)
    if method == "ICALiNGAM":
        model = ICALiNGAM(thresh=config.thresh)
    if method == "DirectLiNGAM":
        model = DirectLiNGAM(thresh=config.thresh)

    if method == "Notears":
        model = Notears(w_threshold=config.thresh)
    if method == "GraNDAG":
        model = GraNDAG(input_dim=data.shape[1], device_type="gpu")
    if method == "GOLEM":
        model = GOLEM(
            num_iter=config.golem_epoch,
            graph_thres=config.thresh,
            device_type="gpu",
            learning_rate=config.lr,
        )

    if method == "SADA":
        model = SADA(
            theta=10,
            alpha=0.05,
            k=10,
            max_cond=3,
            sub_method="pc",
            thresh=config.thresh,
            pc_alpha=config.pc_alpha,
        )
    if method == "HCD":
        model = HCD(
            pre_gate=config.pre_gate, thresh=config.thresh, method=config.sub_method
        )

    model.learn(data)
    end = time.time()

    if method == "HCD":
        execute_time = model.avg_time
    else:
        execute_time = end - start

    print(f"Method {method} Done. Execution time = {execute_time}")
    return model.causal_matrix, execute_time
