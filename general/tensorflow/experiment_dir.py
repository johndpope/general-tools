import time
class ExperimentDir():

    def __init__(self, base_dir, model_name, name=None):
        if name is not None:
            self.experiment_name = name
        else:
            self.experiment_name = str(int(time.time()))

        self.base_dir = base_dir
        self.model_name = model_name

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        pass

    def _concat_params(self, params):
        if not params:
            return ""
        return ",".join("=".join([str(k),str(v)]) for k,v in params.items())

    def get_dir(self,params={}):
        return "{}/{}/{}/{}".format(self.base_dir, self.model_name, self.experiment_name, self._concat_params(params))
