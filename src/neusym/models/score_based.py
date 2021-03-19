# -*- coding: utf-8 -*-


import numpy as np
import scipy.optimize as sopt
import torch
from scipy.special import expit as sigmoid

from .base import BaseModel
from .utils import MLP, LBFGSBScipy


"""
NoTears models are adopted from
https://github.com/xunzheng/notears
"""


class NoTearsLinear(BaseModel):
    def __init__(self, lower, upper):
        super(NoTearsLinear, self).__init__(lower, upper)
    
    def train(self, context):
        # list all objects, including the blicket machine
        objs = []
        for view in context:
            for obj in view["objects"]:
                obj_id = "+".join([obj["shape"], obj["color"], obj["material"]])
                objs.append(obj_id)
        objs = list(set(objs))
        objs.append("blicket machine")
        self.objs = objs[:]
        
        # prepare the data matrix
        data_matrix = []
        for view in context:
            row = [0] * (len(objs))
            view_light_state = view["light_state"]
            view_objs = []
            if view_light_state == "on":
                row[-1] = 1
            for obj in view["objects"]:
                obj_id = "+".join([obj["shape"], obj["color"], obj["material"]])
                # one row of the data matrix
                row[objs.index(obj_id)] = 1
            data_matrix.append(row)
        data_matrix = np.array(data_matrix)
        self.W_est = self.fit(data_matrix, "logistic")

    def fit(self, X, loss_type, lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
        """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        Args:
            X (np.ndarray): [n, d] sample matrix
            lambda1 (float): l1 penalty parameter
            loss_type (str): l2, logistic, poisson
            max_iter (int): max num of dual ascent steps
            h_tol (float): exit if |h(w_est)| <= htol
            rho_max (float): exit if rho >= rho_max
            w_threshold (float): drop edge if |weight| < threshold

        Returns:
            W_est (np.ndarray): [d, d] estimated DAG
        """
        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            #     E = slin.expm(W * W)  # (Zheng et al. 2018)
            #     h = np.trace(E) - d
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
            return obj, g_obj

        n, d = X.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        if loss_type == 'l2':
            X = X - np.mean(X, axis=0, keepdims=True)
        for _ in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break
        W_est = _adj(w_est)
        # as we are only concerned about prediction, there is no need to zero-out links with small weight
        # W_est[np.abs(W_est) < w_threshold] = 0
        return W_est

    def test(self, query):
        """
        Treat the state of blicket machine as a variable and optimize it
        """
        cases = []
        for view in query:
            case = [0] * (len(self.objs) - 1)
            for obj in view["objects"]:
                obj_id = "+".join([obj["shape"], obj["color"], obj["material"]])
                if obj_id in self.objs:
                    case[self.objs.index(obj_id)] = 1
            cases.append(case)
        cases = np.array(cases)
        probs = self.solve(cases, "logistics")
        pred = [self.predict(prob) for prob in probs.tolist()]
        return pred
    
    def solve(self, query, loss_type):
        n = query.shape[0]
        var = 0.5 * np.ones(n)
        def _func(var):
            if loss_type == "logistics":
                X = np.concatenate((query, var[:, np.newaxis]), axis=1)
                M = X @ self.W_est
                loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / n * ((1.0 - 1.0 / (1.0 + np.exp(M)) - X) @ self.W_est[-1, :] - M[:, -1])
            else:
                raise NotImplementedError
            return loss, G_loss
        bounds = [(0, 1)] * n
        sol = sopt.minimize(_func, var, method='L-BFGS-B', jac=True, bounds=bounds)
        return sol.x


class NoTearsMLP(NoTearsLinear):
    def __init__(self, lower, upper):
        super(NoTearsMLP, self).__init__(lower, upper)
        torch.set_default_dtype(torch.double)
    
    def fit(self, X, loss_type, lambda1=0.01, lambda2=0.01, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):

        def dual_ascent_step(loss_func, model, X, lambda1, lambda2, rho, alpha, h, rho_max):
            """Perform one step of dual ascent in augmented Lagrangian."""
            h_new = None
            optimizer = LBFGSBScipy(model.parameters())
            X_torch = torch.from_numpy(X)
            while rho < rho_max:
                def closure():
                    optimizer.zero_grad()
                    X_hat = model(X_torch)
                    loss = loss_func(X_hat, X_torch)
                    h_val = model.h_func()
                    penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                    l2_reg = 0.5 * lambda2 * model.l2_reg()
                    l1_reg = lambda1 * model.fc1_l1_reg()
                    primal_obj = loss + penalty + l2_reg + l1_reg
                    primal_obj.backward()
                    return primal_obj
                optimizer.step(closure)  # NOTE: updates model in-place
                with torch.no_grad():
                    h_new = model.h_func().item()
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            alpha += rho * h_new
            return rho, alpha, h_new

        def squared_loss(output, target):
            n = target.shape[0]
            loss = 0.5 / n * torch.sum((output - target) ** 2)
            return loss

        def log_loss(output, target):
            n = target.shape[0]
            sigmoid = torch.sigmoid(output)
            loss = -1.0 / n * torch.sum(target * torch.log(sigmoid) + (1.0 - target) * torch.log(1.0 - sigmoid))
            return loss

        if loss_type == "l2":
            loss_func = squared_loss
        elif loss_type == "logistic":
            loss_func = log_loss
        else:
            raise ValueError("unknown loss type")

        X = X.astype(np.double)
        d = X.shape[1]
        model = MLP(dims=[d, 10, 1], bias=True)
        rho, alpha, h = 1.0, 0.0, np.inf
        for _ in range(max_iter):
            rho, alpha, h = dual_ascent_step(loss_func, model, X, lambda1, lambda2,
                                             rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        W_est = model.fc1_to_adj()
        # as we are only concerned about prediction, there is no need to zero-out links with small weight
        # W_est[np.abs(W_est) < w_threshold] = 0
        self.model = model
        return W_est

    def solve(self, query, loss_type):
        n = query.shape[0]
        var = 0.5 * np.ones(n)
        def _func(var):
            if loss_type == "logistics":
                query_torch = torch.from_numpy(query).to(torch.get_default_dtype())
                var_torch = torch.from_numpy(var[:, np.newaxis]).to(torch.get_default_dtype())
                var_torch.requires_grad_(True)
                X = torch.cat([query_torch, var_torch], dim=1)
                X_hat = self.model(X)
                # loss
                sigmoid = torch.sigmoid(X_hat)
                loss = -1.0 / n * torch.sum(X * torch.log(sigmoid) + (1.0 - X) * torch.log(1.0 - sigmoid))
                G_loss = torch.autograd.grad(loss, var_torch)[0]
                loss = loss.item()
                G_loss = G_loss.cpu().detach().numpy().flatten()
            else:
                raise NotImplementedError
            return loss, G_loss
        bounds = [(0, 1)] * n
        sol = sopt.minimize(_func, var, method='L-BFGS-B', jac=True, bounds=bounds)
        return sol.x