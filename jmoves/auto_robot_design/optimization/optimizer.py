import os
import dill

import numpy as np

class PymooOptimizer:
    def __init__(self, problem, algortihm, saver=None) -> None:
        self.history = {"X": [], "F": [], "Fs": [], "Mean":[]}
        self.problem = problem
        self.algorithm = algortihm
        self.saver = saver

    def load_history(self, path):
        with open(os.path.join(path, "history.pkl"), "rb") as f:
            self.history.update(dill.load(f))

    def run(self, checkpoint=False, **opt_params):

        self.algorithm.setup(self.problem, **opt_params)
        while self.algorithm.has_next():

            pop = self.algorithm.ask()
            self.algorithm.evaluator.eval(self.problem, pop)

            arrs_F = np.zeros((len(pop), self.problem.n_obj))
            for pop_id, p in enumerate(pop):
                self.history["X"].append(p.X)
                self.history["F"].append(p.F)
                self.history["Fs"].append(p.get("Fs"))
                arrs_F[pop_id] = p.F

            self.algorithm.tell(infills=pop)
            self.history["Mean"].append(np.mean(arrs_F, axis=0))

            if checkpoint:
                assert self.saver is not None
                self.saver.save_history(self.history)
                with open(os.path.join(self.saver.path, "checkpoint.pkl"), "wb") as f:
                    dill.dump(self.algorithm, f)

        res = self.algorithm.result()
        return res
